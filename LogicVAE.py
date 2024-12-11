import math
import torch
from torch import nn
from torch.nn import functional as F

import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist
from gat_layer import GAT
import contrastive_gen as cont
import conversions as conv
from torch.linalg import pinv
from data_utils import get_structure_info_flattened
import kernel as kernel
from traj_measure import BaseMeasure
import parameters_fine_tuning as pft
from memory_profiler import profile

class DAG_GNN(nn.Module):
    def __init__(self, nvar, v_types, arity=None, start_idx=0, end_idx=1, h_dim=250, z_dim=56, heads=None, bidirectional=True, 
                 layers=3, v_id=False,conditional=False, semantic_length=100, semantic_encoding=False, eig= False, xfit= False, kfit = False,
                 semantic_approx=None, device=None):
        super(DAG_GNN, self).__init__()
        
        self.v_types = v_types  # Number of different node types
        self.type_arity = [1, 0, 2, 2, 1, 1, 1 ,2, 0] if arity is None else arity
        self.nvar = nvar
        self.start_idx = start_idx  # Type index of synthetic start and end nodes
        self.end_idx = end_idx
        self.hidden_size = h_dim  # Hidden dimension
        self.latent_size = z_dim  # Latent dimension
        self.v_id = v_id  # Whether vertex should be given a unique identifier
        self.bidirectional = bidirectional  # Whether to use bidirectional message passing
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.conditional=conditional
        self.layers = layers
        self.vertex_state_size = self.hidden_size 
        self.types = 9 #number of effective node type
        self.topology = 12 #length of the topology vector
        self.semantic_length=semantic_length
        self.semantic_approx = semantic_approx #network for semantic approximation of the embeddings
        self.measure=BaseMeasure()
        self.kernel = kernel.StlKernel(self.measure,samples=2000)
        self.semantic_encoding=semantic_encoding #wheter insert semantic in the encoding

        self.conditional = conditional

        #dynamically create GRU layers for each layer index of the GNN
        self.gru_layers_forward = nn.ModuleList([nn.GRU(v_types if i == 0 else self.hidden_size, self.hidden_size, 1) for i in range(layers)])
        self.gru_layers_backward = nn.ModuleList([nn.GRU(self.hidden_size, self.hidden_size, 1) for i in range(layers)])
    
    
        heads = [1] * (layers - 1) + [1] if heads is None else heads
        assert len(heads) == layers
        self.gat_forward = GAT(layers, heads, [v_types] + [h_dim] * layers, device, reverse=False).gat_layers
        self.gat_reverse = GAT(layers, heads, [h_dim] * (layers + 1), device, reverse=True).gat_layers

        #GRU for variables reading 
        self.GRU_var = nn.GRU(self.hidden_size, self.hidden_size, 1)

        #encoding final steps
        self.hidden_graph_size=self.hidden_size*self.nvar if self.semantic_encoding==False else self.hidden_size*self.nvar+ self.semantic_length
        self.mlp_mean = nn.Linear(self.hidden_graph_size, self.latent_size)
        self.mlp_mean_top =nn.Linear(self.latent_size + self.topology, self.latent_size)
        self.mlp_std = nn.Linear(self.hidden_graph_size, self.latent_size)
        self.mlp_std_top =nn.Linear(self.latent_size + self.topology, self.latent_size)

        #DECODER
        self.gru_dec = nn.GRUCell(self.v_types, self.hidden_size)
        from_latent_input_dim = self.latent_size if not self.conditional else self.latent_size + self.semantic_length
        self.from_latent = nn.Linear(from_latent_input_dim, self.hidden_size)
        #simple type prediction
        #self.vertex_type = nn.Sequential(nn.Linear(self.hidden_size * 2 + self.semantic_length, self.hidden_size * 4), nn.ReLU(),
        #                                 nn.Linear(self.hidden_size * 4, self.types-2))

        self.decoding_prediction_size= self.hidden_size * 2 + self.semantic_length if conditional else self.hidden_size * 2
        #complex type prediction
        self.vertex_type = nn.Sequential(
            nn.Linear(self.decoding_prediction_size, self.hidden_size * 5),   
            nn.ReLU(),
            nn.Linear(self.hidden_size * 5, self.hidden_size * 5),   
            nn.ReLU(),
            nn.Linear(self.hidden_size * 5, self.hidden_size * 4),  
            nn.ReLU(),
            nn.Linear(self.hidden_size * 4, self.types - 2)       
        )

        #MLP ofr intervalm threshold an variable prediction
        self.interval_pred = nn.Sequential(nn.Linear(self.decoding_prediction_size, self.hidden_size * 5), nn.ReLU(),
                                         nn.Linear(self.hidden_size * 5, 2))
        self.threshold_pred =nn.Sequential(nn.Linear(self.decoding_prediction_size, self.hidden_size * 5), nn.ReLU(),
                                         nn.Linear(self.hidden_size * 5, 1))
        self.variable_guess = nn.Sequential(nn.Linear(self.decoding_prediction_size, self.hidden_size * 5), nn.ReLU(),
                                         nn.Linear(self.hidden_size * 5, 3)) 
        
        #GRU to keep track of general graph evolution
        self.gru_total_graph = nn.GRUCell(15,self.hidden_size)
        
        self.tanh = nn.Tanh()
        self.logsoftmax = nn.LogSoftmax(1)
        self.negLogLikelihood = nn.NLLLoss(reduction='sum')
        self.gate_forward = nn.Sequential(nn.Linear(self.vertex_state_size, self.hidden_size), nn.Sigmoid())
        self.map_forward = nn.Sequential(nn.Linear(self.vertex_state_size, self.hidden_size, bias=False), )



      
    
    def forward(self, g_in):
        mu, sigma = self.encode(g_in)
        loss, _, _ = self.loss(mu, sigma, g_in)
        return loss

    def encode(self, g_in, g_degree, g_relatives, g_topology):
        #ensure g_in is a list of graphs
        if type(g_in) is not list:
            g_in = [g_in]
        n_vars = [g[0].shape[0] for g in g_in]
        prop_order = range(max(n_vars))
        hidden_single_forward = [torch.zeros(n, self.hidden_size).to(self.device) for i, n in enumerate(n_vars)]
        
        #variables output
        variables_output = torch.zeros((len(g_in), self.layers, self.nvar, self.hidden_size))  # Shape: [num_graphs, layers, 3 vars, encoding_size]
        #variables_position
        p_1,p_2,p_3 = self.get_var_pos(g_in)      
        
        for layer in range(self.layers):
            for v in prop_order:
                hidden_single_forward = self._conv_propagate_to(g_in, g_degree, g_relatives, v, hidden_single_forward, n_vars, layer)
                for idx, hidden in enumerate(hidden_single_forward):
                    if torch.isnan(hidden).any() or torch.isinf(hidden).any():
                        print(f"NaNs/Infs detected in hidden_single_forward[{idx}] after _conv_propagate_to")
                        print(layer)
                        raise ValueError(f"Invalid values in hidden_single_forward[{idx}]")

            #set variables_output layer
            for graph_idx in range(len(g_in)):
                for var_idx, p_var in enumerate([p_1, p_2, p_3]):
                    pos = p_var[graph_idx]
                    
                    if pos != -1:
                        variables_output[graph_idx, layer, var_idx] = hidden_single_forward[graph_idx][pos]
                    else: #no variable of type var_idx found
                        variables_output[graph_idx, layer, var_idx] = torch.zeros(self.hidden_size, device=self.device, dtype=hidden_single_forward[graph_idx].dtype)

            #Reverse propagation only if not at the last layer
            if self.bidirectional and layer != self.layers - 1:
                for w in prop_order:
                    hidden_single_forward = self._conv_propagate_to(g_in, g_degree, g_relatives, max(n_vars) - w - 1, hidden_single_forward, n_vars, layer, reverse=True)
        
        for i, n in enumerate(n_vars):
            hidden_single_forward[i] = hidden_single_forward[i][:n, :self.hidden_size]
        
        #GRU between each single variable through layers (weights shared between variables)
        hidden_graph = self.apply_gru_to_variables(variables_output)
        if self.semantic_encoding:
            graphs_sem= torch.stack([g[2] for g in g_in])
            hidden_graph=torch.cat((hidden_graph,graphs_sem),dim=1)
        mu, sigma = self.mlp_mean(hidden_graph.float()), self.mlp_std(hidden_graph.float())
        tensor_top = torch.stack(g_topology, dim=0).float()
        mu_top = torch.cat((mu, tensor_top), dim=1)
        sigma_top = torch.cat((sigma, tensor_top), dim=1)
        mu_1,sigma_1 = self.mlp_mean_top(mu_top.float()), self.mlp_std_top(sigma_top.float())
        for hs in hidden_single_forward:
            del hs  #to save GPU memory
        return mu_1, sigma_1
    
   
    def get_vertex_feature(self, g, v_idx, hidden_single_forward_g=None, layer=0):
        if layer == 0:
            feat = g[1][v_idx, :].to(self.device)
            if feat.shape[0] < self.v_types:
                feat = torch.cat([feat] + [torch.zeros(self.v_types - feat.shape[0]).to(self.device)])
        else:
            assert (hidden_single_forward_g is not None)
            feat = hidden_single_forward_g[v_idx, :]
        return feat.unsqueeze(0).to(self.device)
    
    def get_vertexes_state(self, g_in, v_idx, hidden_single):
        hidden_v = []
        for i, g in enumerate(g_in):
            hv = torch.zeros(self.hidden_size).to(self.device) if v_idx[i] >= g[0].shape[0] \
                else hidden_single[i][v_idx[i], :]
            hidden_v.append(hv)
        hidden_v = torch.cat(hidden_v, 0).reshape(-1, self.hidden_size)
        return hidden_v


    def _conv_propagate_to(self, g_in, g_degree, g_relatives, v_idx, hidden_single_forward, n_vert, layer=0,reverse=False):
        #send messages to v_idx and update its hidden state
        graphs_idx = [i for i, _ in enumerate(g_in) if n_vert[i] > v_idx]
        graphs = [g_in[i] for i in graphs_idx]
        graphs_deg = [g_degree[i] for i in graphs_idx]
        graphs_rel = [g_relatives[i] for i in graphs_idx]

        if len(graphs) == 0:
            return None, hidden_single_forward
        v_info = [sublist[v_idx] for sublist in graphs_deg]
        v_predecessors = [torch.tensor(sublist[v_idx][0], dtype=torch.int64) for sublist in graphs_rel]    
        v_children = [torch.tensor(sublist[v_idx][1], dtype=torch.int64) for sublist in graphs_rel]   
        v_neigh = [torch.cat([v_pred, v_child]) for v_pred, v_child in zip(v_predecessors, v_children)]
        neigh_info = [[graphs_deg[i][n] for n in v_neigh[i]] for i, g in enumerate(graphs)]
        children_info=[[graphs_deg[i][p] for p in v_children[i]] for i, g in enumerate(graphs)]
        pred_info = [[graphs_deg[i][p] for p in v_predecessors[i]] for i, g in enumerate(graphs)]

        if self.bidirectional:  #accept messages also from children
            if reverse==False:
                if layer==0:
                    h_neigh = [torch.cat([graphs[i][1][n_idx].unsqueeze(0) for n_idx in v_predecessors[i]], dim=0)
                                    if len(v_predecessors[i]) > 0
                                    else torch.zeros((1, graphs[i][1].size(1)))  
                                    for i in range(len(graphs))]
                    h_node = torch.cat([graphs[i][1][v_idx].unsqueeze(0) for i in range(len(graphs))], dim=0)

                else:       #layer > 0
                    h_neigh = [torch.cat([hidden_single_forward[graphs_idx[i]][p_idx].unsqueeze(0) for p_idx in v_predecessors[i]], dim=0) 
                                    if len(v_predecessors[i]) > 0
                                    else torch.zeros((1,hidden_single_forward[graphs_idx[i]][1].size(0)))  
                                    for i in range(len(graphs))]

                    h_node = torch.cat([hidden_single_forward[graphs_idx[i]][v_idx].unsqueeze(0) for i in range(len(graphs))], dim=0)
            else: #backward part
                if layer==0:
                    h_neigh = [torch.cat([hidden_single_forward[graphs_idx[i]][n_idx].unsqueeze(0) for n_idx in v_children[i]], dim=0) 
                                    if len(v_children[i]) > 0
                                    else torch.zeros((1,hidden_single_forward[graphs_idx[i]][1].size(0)))    
                                    for i in range(len(graphs))]
                    
                    h_node = torch.cat([hidden_single_forward[graphs_idx[i]][v_idx].unsqueeze(0) for i in range(len(graphs))], dim=0)
                    


                else:   #layer > 0
                    h_neigh = [torch.cat([hidden_single_forward[graphs_idx[i]][n_idx].unsqueeze(0) for n_idx in v_children[i]], dim=0) 
                                    if len(v_children[i]) > 0
                                    else torch.zeros((1,hidden_single_forward[graphs_idx[i]][1].size(0)))  
                                    for i in range(len(graphs))]
                    h_node = torch.cat([hidden_single_forward[graphs_idx[i]][v_idx].unsqueeze(0) for i in range(len(graphs))], dim=0)

                
        else:  # monodirectional
            if layer==0:
                h_neigh = [torch.cat([graphs[i][1][n_idx].unsqueeze(0) for n_idx in v_predecessors[i]], dim=0)
                                if len(v_predecessors[i]) > 0
                                else torch.zeros((1, graphs[i][1].size(1)))  
                                for i in range(len(graphs))]
                h_node = torch.cat([graphs[i][1][v_idx].unsqueeze(0) for i in range(len(graphs))], dim=0)

            else:       #layer > 0
                h_neigh = [torch.cat([hidden_single_forward[graphs_idx[i]][p_idx].unsqueeze(0) for p_idx in v_predecessors[i]], dim=0) 
                                if len(v_predecessors[i]) > 0
                                else torch.zeros((1,hidden_single_forward[graphs_idx[i]][1].size(0)))  
                                for i in range(len(graphs))]

                h_node = torch.cat([hidden_single_forward[graphs_idx[i]][v_idx].unsqueeze(0) for i in range(len(graphs))], dim=0)
        
        
        h_self = [torch.cat([self.get_vertex_feature(g, v_idx, hidden_single_forward[graphs_idx[i]], layer + reverse)] *
                                h_neigh[i].shape[0], 0) for i, g in enumerate(graphs)]
        max_n_neigh = max([n.shape[0] for n in h_neigh])
        h_self = [torch.cat([h_s.to(self.device)] + [torch.zeros(max_n_neigh - len(h_s), h_s.shape[1]).to(
                self.device)], 0).unsqueeze(0) for h_s in h_self]
        h_self = torch.cat(h_self, 0).to(self.device)  # [batch, max_n_neigh, n_types]
        h_neigh = [torch.cat([h_n.to(self.device)] + [torch.zeros(max_n_neigh - len(h_n), h_n.shape[1]).to(
                self.device)], 0).unsqueeze(0) for h_n in h_neigh]
        h_neigh = torch.cat(h_neigh, 0).to(self.device)  # [batch, max_n_neigh, n_types]
        
        if reverse:
            h_v1 = self.gat_reverse[layer](h_self, h_neigh)
        else:
            h_v1 = self.gat_forward[layer](h_self, h_neigh)
        
        h_v = torch.mean(h_v1, dim=1)
        h_v = h_v.unsqueeze(0)
        h_node=h_node.unsqueeze(0)

        if reverse == False:
            h_v, hidden = self.gru_layers_forward[layer](h_node, h_v)
        else:
            h_v, hidden = self.gru_layers_backward[layer](h_node, h_v)

        
        for i, g in enumerate(graphs):
            hv = h_v[0, i, :].unsqueeze(0)  

            hv_shape_2 = hv.shape[1]  #get the feature dimension
            hidden_shape_1 = hidden_single_forward[graphs_idx[i]].shape[1]  #get the expected hidden dimension

            if hv_shape_2 > hidden_shape_1:
                #eventual pad
                pad = nn.ConstantPad1d((0, hv_shape_2 - hidden_shape_1), 0) 
                hidden_single_forward[graphs_idx[i]] = pad(hidden_single_forward[graphs_idx[i]])
            elif hv_shape_2 < hidden_shape_1:
                pad = nn.ConstantPad1d((0, hidden_shape_1 - hv_shape_2), 0) 
                hv = pad(hv)  
            
            hidden_single_forward[graphs_idx[i]][v_idx, :] = hv.squeeze(0) 
   
        return hidden_single_forward


    
    def name_model(self):
        direction = 'bidirectional' if self.bidirectional else 'monodirectional'
        name = 'DAG GNN' 
        print(f'{name} {direction}')
    

    def get_graph_state(self, g_in, hidden_single_forward, hidden_single_backward=None, intermediate=False,
                        decode=False):
        semantic_vects = []
        hidden_graph = []
        for g, graph in enumerate(g_in):
            # take hidden state of last node (e.g. end or last added)
            hidden_g = hidden_single_forward[g][-1, :]
            if self.bidirectional and not decode:
                assert (hidden_single_backward is not None)
                hidden_g_back = hidden_single_backward[g][0, :]
                hidden_g = torch.cat([hidden_g, hidden_g_back], 0)
            hidden_graph.append(hidden_g.unsqueeze(0))
            if self.conditional and not intermediate:
                semantic_vects.append(graph[2].reshape(1, -1))
        hidden_graph = torch.cat(hidden_graph, 0)
        if self.bidirectional and not decode:
            hidden_graph = self.unify_hidden_graph(hidden_graph)
        hidden_graph = hidden_graph.reshape(-1, self.hidden_size)
        # during decoding or loss computation we don't use the semantic information
        if self.conditional and not intermediate:
            semantic_vects = torch.cat(semantic_vects, 0)
            hidden_graph = torch.cat([hidden_graph, semantic_vects], 1)
        return hidden_graph

    def get_var_pos(self, g_in):
        '''Return the node index in the graph that stands for the three variables (-1 if a variable is not present)'''
        #initialize the lists for positions of each variable
        positions_first = []
        positions_second = []
        positions_third = []

        #target variables
        first_variable = torch.tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                    0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000], device=self.device)
        second_variable = torch.tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                        0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000], device=self.device)
        third_variable = torch.tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                    0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000], device=self.device)

        for i, couple in enumerate(g_in):
            second_tensor = couple[1]
            
            #variable are found
            found_first = False
            found_second = False
            found_third = False

            for row_index in range(second_tensor.shape[0]):
                if torch.equal(second_tensor[row_index], first_variable) and not found_first:
                    positions_first.append(row_index)
                    found_first = True  #mark as found to prevent multiple entries, only one variable

                if torch.equal(second_tensor[row_index], second_variable) and not found_second:
                    positions_second.append(row_index)
                    found_second = True

                if torch.equal(second_tensor[row_index], third_variable) and not found_third:
                    positions_third.append(row_index)
                    found_third = True

            # If a variable wasn't found in any row, append -1 as a placeholder
            if not found_first:
                positions_first.append(-1)
            if not found_second:
                positions_second.append(-1)
            if not found_third:
                positions_third.append(-1)
            
        return positions_first, positions_second, positions_third


    def apply_gru_to_variables(self, encoded_values):
        """
        Apply GRU on the encoded values for each variable across the layers, 
        and concatenate the final hidden states of the variables for each graph.
        Returns atensor of shape [batch_size, hidden_size * num_vars], where the final
        GRU-encoded hidden states for each variable are concatenated.
        """
        encoded_values = encoded_values.to(self.device)
        batch_size, num_layers, num_vars, hidden_size = encoded_values.shape

        final_hidden_states = []

        #iteration over the three variables
        for var_idx in range(num_vars):
            h = encoded_values[:, 0, var_idx].unsqueeze(0).contiguous()  #first layer's encoding as initial hidden state

            for layer_idx in range(1, num_layers):
                #current input for the GRU is the next layer's encoded value for this variable
                input_val = encoded_values[:, layer_idx, var_idx].unsqueeze(0).contiguous()    # [1, batch_size, hidden_size]
                #pass the input and previous hidden state through the GRU
                _, h = self.GRU_var(input_val, h)  #the second return value is the hidden state

            #storing
            final_hidden_states.append(h.squeeze(0))  # [batch_size, hidden_size]

        #Concatenate the hidden states for all variables along the last dimension
        #final_hidden_states will be a list of tensors of shape [batch_size, hidden_size], one for each variable.
        final_hidden_states = torch.cat(final_hidden_states, dim=1)

        return final_hidden_states  # [batch_size, hidden_size * num_vars]



    def reparameterize(self, mu, sigma, eps=0.01):
        if self.training:
            return (torch.randn_like(sigma) * eps).mul(sigma.mul(0.5).exp_()).add_(mu)
        else:
            return mu



    def loss(self, mu, sigma, g_true, g_complete, beta=0.1, alpha=0.92,accuracy=False): 
        
        if torch.isnan(mu).any() or torch.isinf(mu).any():
            print(f"Invalid values detected in mu")
            raise ValueError("NaNs or Infs in mu")
        if torch.isnan(sigma).any() or torch.isinf(sigma).any():
            print(f"Invalid values detected in sigma")
            raise ValueError("NaNs or Infs in sigma")
        #teacher forcing setup
        z = self.reparameterize(mu.float(), sigma.float())
        n_graphs = len(z)
        if self.conditional: #concatenate z with the semantic vector
            y = torch.cat([g[2].reshape(1, -1) for g in g_true], 0)
            combined_z = torch.cat([z, y], dim=1)  
        else:
            combined_z = z
            
        #transformation via MLP to size hidden_size
        trans_z = self.tanh(self.from_latent(combined_z.float()))
        hidden_zero = trans_z
        if torch.isnan(hidden_zero).any() or torch.isinf(hidden_zero).any():
            print("NaN or Inf detected in hidden_zero")
            raise ValueError("NaN or Inf in hidden_zero")

        #initialize graph reconstruction state
        g_batch = [[torch.zeros(1, 1).float().to(self.device), torch.zeros(1, self.v_types).float().to(self.device)]
                for _ in range(n_graphs)]
        completed = [False for _ in range(n_graphs)]
        for g in g_batch:
            g[1][0, 0] = 1  #start node type
        correct_nodes=[0] * len(g_true)
        hidden_single = [torch.zeros(1, self.hidden_size).to(self.device) for _ in range(n_graphs)] #[batch_size,node_index,hidden_size]
        hidden_v, hidden_single = self.message_to(g_batch, 0, self.gru_dec, hidden_single=hidden_single,
                                                hidden_agg=hidden_zero)
            
        ll = 0  # log-likelihood
        n_vert = [g[0].shape[0] for g in g_true]
        complete_vertexes, current_nodes_in_g = [[[] for _ in range(n_graphs)], [1 for _ in range(n_graphs)]]
        interval_loss, threshold_loss, variable_ll = 0, 0, 0
        depths = [[0] for _ in range(n_graphs)]  #depth of each node, starting with root node at depth 0

        #loop over the node index, generating nodes
        for v_true in range(1, max(n_vert)):
            if sum(completed) == n_graphs:
                break
            
            true_types = [torch.nonzero(g[1][v_true, :9]).flatten() if v_true < n_vert[i] else self.start_idx
                            for i, g in enumerate(g_true)] #list of true types of v_true for each graph in the batch
                
            hidden_g = self.get_graph_state(g_batch, hidden_single, intermediate=True, decode=True)
            v_pred = [np.setdiff1d(np.arange(cur), np.array(com)) for cur, com in zip(current_nodes_in_g, complete_vertexes)]
            v_pred = list(map(lambda v: np.max(v) if len(v) > 0 else -1, v_pred)) #index of predecessor of v_true
                
            h_v_pred = self.get_vertexes_state(g_batch, v_pred, hidden_single)
            if self.conditional:
                h_pred_type = torch.cat([y, h_v_pred,hidden_zero], 1)
            else:
                h_pred_type = torch.cat([h_v_pred,hidden_zero], 1)
        

                
            intervals, thresholds = [], []
            for i, g in enumerate(g_batch):
                true_type = true_types[i]
                if isinstance(true_type, torch.Tensor):
                    is_a_temp = torch.any(true_type == torch.tensor([5, 6, 7], device=true_type.device)).float()
                    if is_a_temp: #TEMPORAL OPERATOR
                        a, b = self.interval_pred(h_pred_type[i]) #interval prediction
                        a,b = abs(a), abs(b)
                        interval = (a, a + b + 1)
                        
                    else:
                        interval = (0, 0)
                    is_type_8 = (true_type == 8).float()
                    if is_type_8: #THRESHOLD
                        threshold = self.threshold_pred(h_pred_type[i]) #threshold prediction
                    else:
                        threshold = 0
                else:
                    interval = np.NaN
                    threshold = np.NaN
                intervals.append(interval)
                thresholds.append(threshold)

            types_score = self.vertex_type(h_pred_type) #guess the node type, return a tensor [types]
            new_attr_rows = [torch.zeros(self.v_types).float().to(self.device) for _ in range(n_graphs)]
            vll = self.logsoftmax(types_score)
                

            #loop over the graphs, about the current node generated
            for i, g in enumerate(g_batch):
                if completed[i]:
                    continue
                if true_types[i] != self.end_idx and true_types[i] != self.start_idx:
                    #retrieve depth of parent node and calculate the child node's depth
                    parent_depth = depths[i][v_pred[i]] if v_pred[i] != -1 else 0
                    node_depth = parent_depth + 1
                    depths[i].append(node_depth)
                    depth_weight =alpha ** node_depth #depth weight influences the loss

                    nll_loss = self.negLogLikelihood(
                        vll[i].unsqueeze(0),
                        torch.Tensor([true_types[i] - 2]).type(torch.LongTensor).to(self.device)
                    ) #negative log_likelihood
                    if accuracy==True: #calculate number of times the most probable type is the true type
                        pred_node_t=torch.argmax(types_score[i])
                        if pred_node_t==true_types[i] - 2:
                            correct_nodes[i]+=1

                    ll += depth_weight * nll_loss

                    current_nodes_in_g[i] += 1
                    if true_types[i] == 8:
                        complete_vertexes[i].append(v_true)
                        complete_vertexes[i] = list(set(complete_vertexes[i]))
                    new_attr_rows[i][true_types[i]] = 1
                    g[0] = self.add_node_adj(g[0])
                    g[0][v_pred[i], v_true] = 1 if v_pred[i] != -1 else 0

                    type_v_pred  = torch.nonzero(g[1][v_pred[i], :9], as_tuple=True)[0]
                    n_edge_v_pred = len(torch.nonzero(g[0][v_pred[i], :]).flatten())
                    if n_edge_v_pred == self.type_arity[type_v_pred]:
                        complete_vertexes[i].append(v_pred[i])
                        complete_vertexes[i] = list(set(complete_vertexes[i]))

                    if true_types[i] in [5, 6, 7]:
                        new_attr_rows[i][12], new_attr_rows[i][13] = g_true[i][1][v_true][12], g_true[i][1][v_true][13]
                        interval_loss += (g_true[i][1][v_true][12] - intervals[i][0])**2 + (g_true[i][1][v_true][13] - intervals[i][1])**2
                    if true_types[i] == 8:
                        new_attr_rows[i][14] = g_true[i][1][v_true][14]
                        threshold_loss += (g_true[i][1][v_true][14] - thresholds[i])**2

                    g[1] = self.add_node_attr(g[1], new_attr_rows[i])

                    
                if current_nodes_in_g[i] == len(complete_vertexes[i]) and len(complete_vertexes[i]) > 1:
                    completed[i] = True

            hidden_single = self.add_zeros_row_batch(hidden_single)
            hidden_v, hidden_single = self.message_to(g_batch, v_true, self.gru_dec, hidden_single=hidden_single)

            #variable prediction for type == 8
            for i, g in enumerate(g_batch):
                if true_types[i] == 8:
                    var_predecessor=hidden_single[i][-1]
                    if self.conditional:
                        variable_predictor = torch.cat([var_predecessor, y[i],hidden_zero[i]], 0)
                    else:
                        variable_predictor = torch.cat([var_predecessor,hidden_zero[i]], 0)
                    var_dist = self.variable_guess(variable_predictor) #VARIABLE PREDICTION
                    varll = F.log_softmax(var_dist, dim=0)
                    true_variable = self.find_true_variable(new_attr_rows[i], g_complete[i])
                    indices = torch.arange(true_variable.shape[0], device=true_variable.device, dtype=true_variable.dtype)
                    true_index = torch.sum(true_variable * indices)
                    true_index = true_index.long()
                    varll_loss = self.negLogLikelihood(varll.unsqueeze(0), true_index.unsqueeze(0))
                    variable_ll += varll_loss
                    
                    #update threshold like after the variable node
                    temp_att=torch.zeros(self.v_types).float().to(self.device)
                    temp_att[9+true_index]=1
                    temp_att = F.pad(temp_att, pad=(0, self.v_types - len(temp_att)), mode='constant', value=0) if self.v_types - len(temp_att) > 0 else temp_att
                    x = temp_att.unsqueeze(0) 
                    h_pred = hidden_single[i][-1].unsqueeze(0) 
                    hidden_agg = (self.gate_forward(h_pred) * self.map_forward(h_pred)).sum(0).unsqueeze(0)
                    new_hidden_v = self.gru_dec(x.to(self.device), hidden_agg.to(self.device))
                    updated_hidden = hidden_single[i].clone()
                    updated_hidden[-1, :] = new_hidden_v
                    hidden_single[i] = updated_hidden


            current_hidden_single = torch.stack([tensor[-1, :] for tensor in hidden_single])
            new_attr_rows_tensor = torch.stack(new_attr_rows)  # Shape: [batch_size, input_size]
            hidden_zero= self.gru_total_graph(new_attr_rows_tensor,hidden_zero)
            if torch.isnan(hidden_zero).any() or torch.isinf(hidden_zero).any():
                print("NaN or Inf detected in hidden_zero")
                raise ValueError("NaN or Inf in hidden_zero")

            
            
        if torch.isnan(sigma).any() or torch.isinf(sigma).any():
            print("Invalid values detected in sigma before exp().")
            print("Sigma:", sigma)
            raise ValueError("NaNs or Infs in sigma")
            
 
        # KL divergence
        kld = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
            
        if accuracy==True:
            new_list = [correct_nodes[i] / len(g_true[i][1]) for i in range(len(g_true))]
            mean_value = sum(new_list) / len(new_list)

        #eventual weight penalties to contrast overfitting
        l2_penalty = 0.0
        for param in self.vertex_type.parameters():
            l2_penalty += torch.sum(param.pow(2))
        l3_penalty=0.0    
        for param in self.gru_total_graph.parameters():
            l3_penalty += torch.sum(param.pow(2))
        for param in self.gru_dec.parameters():
            l3_penalty += torch.sum(param.pow(2))
        for param in self.gate_forward.parameters():
            l3_penalty += torch.sum(param.pow(2))
        for param in self.map_forward.parameters():
            l3_penalty += torch.sum(param.pow(2))
        l4_penalty=0.0  
        for param in self.from_latent.parameters():
            l4_penalty += torch.sum(param.pow(2)) 
        l5_penalty=0.0
        for param in self.variable_guess.parameters():
            l5_penalty += torch.sum(param.pow(2))
        for param in self.threshold_pred.parameters():
            l5_penalty += torch.sum(param.pow(2))
        for param in self.interval_pred.parameters():
            l5_penalty += torch.sum(param.pow(2))   

                
            
        return ll + beta * kld + 0.01 * interval_loss + threshold_loss + variable_ll+ 0.05*l2_penalty+0.1*l4_penalty+0.0001*(l3_penalty+4*l5_penalty), ll, kld, interval_loss, threshold_loss, variable_ll, mean_value





    def message_to(self, g_in, v_idx, prop_net, hidden_single, hidden_agg=None, reverse=False):
        # send all messages to vertex v from predecessors
        # g_in is a list of graphs (i.e. of list [adj, attr])
        n_vert = [g[0].shape[0] for g in g_in]
        graphs_idx = [i for i, _ in enumerate(g_in) if n_vert[i] > v_idx]
        graphs = [g for i, g in enumerate(g_in) if i in graphs_idx]
        if len(graphs) == 0:
            return None, hidden_single
        # extract adjacency and feature matrix for each graph in the list
        adj = [g[0] for g in graphs]
        type_v = [g[1][v_idx, :] for g in graphs]
        # zero-padding for consistent dimensionality
        type_v = [F.pad(t, pad=(0, self.v_types - len(t)), mode='constant', value=0) if self.v_types - len(t) > 0 else t
                  for t in type_v]
        x = torch.cat(type_v, 0).reshape(len(type_v), -1)
        v_ids = None
        if hidden_agg is not None:
            # original size: [#graphs, hidden_dim]
            hidden_agg = hidden_agg[graphs_idx]
        if reverse:
            # find successor of current node for each graph in the list
            succ_idx = [(a[v_idx, :] == 1).nonzero().type(torch.LongTensor) for a in adj]
            # h_pred has size [#graphs, #succ, hidden_dim]
            h_pred = [[hidden_single[g_idx][s_idx, :] for s_idx in succ_idx[i]] for i, g_idx in enumerate(graphs_idx)]
            if self.v_id:
                succs = [succ.unsqueeze(0).t() if len(succ.shape) < 2 else succ.t() for succ in succ_idx]
                v_ids = [torch.zeros((len(h_pred[i]), self.max_n_vert)).scatter_(1, succ, 1).to(self.device)
                         for i, succ in enumerate(succs)]
            # gated sum of messages
            gate, mapper = self.gate_backward, self.map_backward
        else:
            # find predecessor of current node for each graph in the list
            pred_idx = [(a[:, v_idx] == 1).nonzero().type(torch.LongTensor) for a in adj]
            # h_pred has size [#graphs, #pred, hidden_dim]
            h_pred = [[hidden_single[g_idx][p_idx, :] for p_idx in pred_idx[i]] for i, g_idx in enumerate(graphs_idx)]
            if self.v_id:
                preds = [pred.unsqueeze(0).t() if len(pred.shape) < 2 else pred.t() for pred in pred_idx]
                v_ids = [torch.zeros((len(h_pred[i]), self.max_n_vert)).scatter_(1, pred, 1).to(self.device)
                         for i, pred in enumerate(preds)]
            gate, mapper = self.gate_forward, self.map_forward
        if self.v_id:
            h_pred = [[torch.cat([x[i], y[i:i + 1]], 1) for i in range(len(x))] for x, y in zip(h_pred, v_ids)]
        if hidden_agg is None:
            max_pred = max([len(p) for p in h_pred])
            if max_pred == 0:
                hidden_agg = torch.zeros(len(graphs), self.hidden_size).to(self.device)
            else:
                h_pred = [torch.cat(h_p + [torch.zeros(max_pred - len(h_p), self.vertex_state_size).to(self.device)],
                                    0).unsqueeze(0) for h_p in h_pred]
                h_pred = torch.cat(h_pred, 0)
                hidden_agg = (gate(h_pred) * mapper(h_pred)).sum(1)
        new_hidden_v = prop_net(x.to(self.device), hidden_agg.to(self.device))
        for i, g_idx in enumerate(graphs_idx):
            hidden_single[g_idx][v_idx, :] = new_hidden_v[i:i + 1]
        return new_hidden_v, hidden_single       
    

    def add_node_adj(self, adj):
        new_adj = torch.zeros(adj.shape[0] + 1, adj.shape[1] + 1).to(self.device)
        new_adj[0:-1, 0:-1] = adj
        return new_adj

    @staticmethod
    def add_node_attr(attr, new_row):
        return torch.cat([attr, new_row.unsqueeze(0)], 0)

    def add_zeros_row_batch(self, h):
        return [torch.cat([h[i], torch.zeros(1, self.hidden_size).to(self.device)], 0) for i in range(len(h))]

    def add_zeros_hidden(self, h):
        return torch.cat([h, torch.zeros(1, self.hidden_size).to(self.device)], 0)
    
    @staticmethod
    def find_true_variable(attribute_tensor, graph):
        '''attribute tensor is to be find in the (complete) graph and the relative variable index is extracted'''
        threshold = attribute_tensor[-1]
        
        #first row that contains the threshold
        index = -1  
        for i, row in enumerate(graph[1]):
            if threshold in row:
                index = i
                break  #pick the first row that matches
        
        if index == -1:
            raise ValueError(f"Threshold {threshold} not found in graph[1]")

        #get the adjacency matrix and find the child nodes of the current node (index)
        adjacency_matrix = graph[0] 
        node_row = adjacency_matrix[index] 
        child_indices = torch.nonzero(node_row, as_tuple=False).squeeze()

        #if there are multiple children, handle the filtering
        if child_indices.numel() > 1:
            #find the parent of the current node (exactly one parent)
            parent_indices = torch.nonzero(adjacency_matrix[:, index], as_tuple=False).squeeze()
            
            #error: correct parent condition (should check if parent_indices.numel() is not 1)
            if parent_indices.numel() != 1:
                raise ValueError(f"Node at index {index} does not have exactly one parent, found {parent_indices.numel()}")
            
            parent_index = int(parent_indices)              
            parent_children = torch.nonzero(adjacency_matrix[parent_index], as_tuple=False).squeeze()

            #find the common node between the current node's children and the parent's children
            common_node_mask = torch.isin(child_indices, parent_children)
            common_node = child_indices[common_node_mask]

            if common_node.numel() != 1:
                raise ValueError(f"Expected exactly one common node between the current node's children and the parent's children, but found {common_node.numel()}")

            #filter out the common child
            child_indices = child_indices[child_indices != common_node]

        #ensure child_indices is not a scalar (0-dim tensor)
        if child_indices.numel() == 1:
            child_index = int(child_indices) 
        else:
            child_index = int(child_indices[0]) 

        #extract var_node_attr
        var_node_attr = graph[1][child_index][9:12]

        return var_node_attr
    
    

    def encode_decode(self,g_in, g_degree, g_relatives, g_topology,prior=False):
        if prior==False:
            mu, sigma = self.encode(g_in, g_degree, g_relatives, g_topology)
            z = self.reparameterize(mu, sigma)
        else:
            z = torch.randn(len(g_in), self.latent_size, device=self.device)
        if self.conditional:
            y = torch.cat([g[2].reshape(1, -1) for g in g_in], 0)
            return self.decode(z,y)
        else:
            return self.decode(z)
    
    def decode(self, z, y=None, stochastic=True, max_vertices=50):
        if self.conditional:
            assert (y is not None)
            combined_z = torch.cat([z, y], dim=1)  
        else:
            combined_z = z
        
        trans_z = self.tanh(self.from_latent(combined_z.float()))
        hidden_zero = trans_z.detach().clone()
        

        n_graphs = len(z)
        #first node generated is the starting node
        g_batch = [[torch.zeros(1, 1).float().to(self.device), torch.zeros(1, self.v_types).float().to(self.device)]
                for _ in range(n_graphs)]
        
        for g in g_batch:
            g[1][0, 0] = 1  # start node type
        
        hidden_single = [torch.zeros(1, self.hidden_size).to(self.device) for _ in range(n_graphs)]
        hidden_v, hidden_single = self.message_to(g_batch, 0, self.gru_dec, hidden_single=hidden_single,
                                                hidden_agg=hidden_zero)
        
        completed = [False for _ in range(n_graphs)]
        complete_vertexes, current_nodes_in_g = [[[] for _ in range(n_graphs)], [1 for _ in range(n_graphs)]]
        
        v_idx = 1  #vertex I'm adding now
        #list to store variable associated with threshold
        v_1=[[] for _ in range(n_graphs)] 
        v_2=[[] for _ in range(n_graphs)] 
        v_3=[[] for _ in range(n_graphs)] 

        #generate vertexes and sample their type
        while min(current_nodes_in_g) < max_vertices - 1:
            if sum(completed) == n_graphs:
                break
                        
            current_graph_state = self.get_graph_state(g_batch, hidden_single, intermediate=True, decode=True)

            v_preds = [np.setdiff1d(np.arange(cur), np.array(com)) for cur, com in zip(current_nodes_in_g, complete_vertexes)]
            v_preds = list(map(lambda v: np.max(v) if len(v) > 0 else -1, v_preds))
           
            h_v_preds = self.get_vertexes_state(g_batch, v_preds, hidden_single)

            if self.conditional:
                h_pred_type = torch.cat([y, h_v_preds, hidden_zero], 1)
            else:
                h_pred_type = torch.cat([h_v_preds, hidden_zero], 1)
            
            v_type_scores = self.vertex_type(h_pred_type)
            #NODE TYPE PREDICTION
            if stochastic:
                v_type_probs = F.softmax(v_type_scores, 1).detach()
                v_type = [torch.multinomial(v_type_probs[i], 1) + 2 for i in range(n_graphs)]
            else:
                v_type = torch.argmax(v_type_scores, 1) + 2
                v_type = v_type.flatten().tolist()
            
            new_attr_rows = [torch.zeros(self.v_types).float().to(self.device) for _ in range(n_graphs)]
            thresholds = torch.full((len(g_batch),), float('nan'))
            for i, g in enumerate(g_batch):
                if not completed[i]:
                    if current_nodes_in_g[i] == len(complete_vertexes[i]) and len(complete_vertexes[i]) > 1:
                        v_type[i] = self.end_idx
                    if current_nodes_in_g[i] == max_vertices - 2:
                        v_type[i] = self.end_idx
                    
                    if v_type[i] in [5, 6, 7]:  #node types that require intervals
                        a, b = self.interval_pred(h_pred_type[i]) #INTERVAL PREDICTION
                        interval = (int(abs(a)), int((abs(a)) +int(abs(b))) + 1)
                        new_attr_rows[i][12], new_attr_rows[i][13] = interval[0], interval[1]
                        
                    
                    if v_type[i] == 8:  #node type that requires threshold
                        threshold = self.threshold_pred(h_pred_type[i]) #THRESHOLD PREDICTION
                        new_attr_rows[i][14] = threshold
                        thresholds[i]= threshold
                        

            for i, g in enumerate(g_batch):     
                if not completed[i]:  
                    #update the graph with the new node and its attributes
                    new_attr_rows[i][v_type[i]] = 1
                    g[1] = self.add_node_attr(g[1], new_attr_rows[i])  #add the new node attributes
                    g[0] = self.add_node_adj(g[0])
                    g[0][v_preds[i], v_idx] = 1 if v_preds[i] != -1 else 0
                    current_nodes_in_g[i] += 1
                    
                    if v_type[i] >= len(self.type_arity) - 1: #threshold
                        complete_vertexes[i].append(v_idx)
                        complete_vertexes[i] = list(set(complete_vertexes[i]))
                    
                    type_v_p = torch.nonzero(g[1][v_preds[i], :9]).item()
                    type_v_pred = len(self.type_arity) - 1 if type_v_p >= len(self.type_arity) - 1 else type_v_p
                    n_edge_v_pred = len(torch.nonzero(g[0][v_preds[i], :]).flatten())
                    if n_edge_v_pred == self.type_arity[type_v_pred]:
                        complete_vertexes[i].append(v_preds[i])
                        complete_vertexes[i] = list(set(complete_vertexes[i]))
                    
                    
                    if v_type[i] == self.end_idx:  # complete the graph
                        completed[i] = True
            hidden_single = self.add_zeros_row_batch(hidden_single)
            hidden_v, hidden_single = self.message_to(g_batch, v_idx, self.gru_dec, hidden_single=hidden_single)
            v_idx += 1

        
            for i, g in enumerate(g_batch):   
                if not torch.isnan(thresholds[i].clone().detach()): #threshold node
                    var_predecessor=hidden_single[i][-1]
                    if self.conditional:
                        variable_predictor=torch.cat([var_predecessor, y[i],hidden_zero[i]], 0)
                    else:
                        variable_predictor=torch.cat([var_predecessor,hidden_zero[i]], 0)
                    var_dist=self.variable_guess(variable_predictor) #VARIABLE PREDICTION
                    varll = F.softmax(var_dist, dim=0)
                    var_type = torch.multinomial(varll, 1).item()
                    #store the variable predicted
                    if var_type==0:
                        v_1[i].append(v_idx-1)
                    if var_type==1:
                        v_2[i].append(v_idx-1)
                    if var_type==2:
                        v_3[i].append(v_idx-1)
                    #upgrade as if variable node is added
                    temp_att=torch.zeros(self.v_types).float().to(self.device)
                    temp_att[9+var_type]=1
                    temp_att = F.pad(temp_att, pad=(0, self.v_types - len(temp_att)), mode='constant', value=0) if self.v_types - len(temp_att) > 0 else temp_att
                    x = temp_att.unsqueeze(0) 
                    h_pred = hidden_single[i][-1].unsqueeze(0) 
                    hidden_agg = (self.gate_forward(h_pred) * self.map_forward(h_pred)).sum(0).unsqueeze(0)
                    new_hidden_v = self.gru_dec(x.to(self.device), hidden_agg.to(self.device))
                    hidden_single[i][-1, :] = new_hidden_v.squeeze(0)

                    complete_vertexes[i].append(v_idx-1)
                    complete_vertexes[i] = list(set(complete_vertexes[i]))
           
            current_hidden_single = torch.stack([tensor[-1, :] for tensor in hidden_single])
            new_attr_rows_tensor = torch.stack(new_attr_rows)  # Shape: [batch_size, input_size]
            hidden_zero= self.gru_total_graph(new_attr_rows_tensor,hidden_zero)


        #ADD VARIABLE NODES
        v_1_att = torch.zeros(1, self.v_types).float().to(self.device)
        v_1_att[0, 9] = 1
        v_2_att = torch.zeros(1, self.v_types).float().to(self.device)
        v_2_att[0, 10] = 1
        v_3_att = torch.zeros(1, self.v_types).float().to(self.device)
        v_3_att[0, 11] = 1
        for i, g in enumerate(g_batch):
            g[1] = torch.cat([g[1][: -1], v_1_att, v_2_att, v_3_att, g[1][-1:]], dim=0)
            for _ in range(3):
                g[0] = self.add_node_adj(g[0])
            
            #edges with the respective thresholds
            g[0][v_1[i], -4] = 1
            g[0][v_2[i], -3] = 1
            g[0][v_3[i], -2] = 1
            
            #edges with the end node
            g[0][-4, -1] = 1
            g[0][-3, -1] = 1
            g[0][-2, -1] = 1
            
            g_batch[i]=conv.trim(g) #adds until edge and eliminates the not connected nodes
               
        for hs in hidden_single:
            del hs  # free memory
        return g_batch
    
    


    def semantic_loss(self, g_true, true_encoding, g_degree, g_relatives, g_topology):
        '''Semantic loss that uses semantic approximation disatance as loss, no teacher forcing'''
        g_reconstructed= self.encode_decode(g_true, g_degree, g_relatives, g_topology)
        rec_degree, rec_relatives = get_structure_info_flattened(g_reconstructed, self.device)
        semantic_loss = self.semantic_approx.forward(g_reconstructed,rec_degree, rec_relatives,len(g_reconstructed),true_encoding,[])
        return torch.sum((semantic_loss-true_encoding)**2)
    
    


    

            
    
    