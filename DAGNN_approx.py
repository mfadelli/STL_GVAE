import math
import torch
from torch import nn
from torch.nn import functional as F
from gat_layer import GAT
from torch.linalg import pinv

                    ################################################
                    #   Network to approximate kernel embeddings   #
                    #    NEEDS RETRAINING on correct embeddings    #
                    #   weight in at encoding_approximating_GNN.pt #
                    ################################################


#SAME FUNDAMENTAL STRUCTURE OF THE ENCODING OF THE GRAPH-VAE 

class DAG_GNN(nn.Module):
    def __init__(self, nvar, v_types, start_idx=0, end_idx=1, h_dim=200, z_dim=100, heads=None, bidirectional=True, layers=3, v_id=False, device=None):
        super(DAG_GNN, self).__init__()
        
        self.v_types = v_types  # Number of different node types
        self.nvar = nvar
        self.start_idx = start_idx  # Type index of synthetic start and end nodes
        self.end_idx = end_idx
        self.hidden_size = h_dim  # Hidden dimension
        self.latent_size = z_dim  # Latent dimension
        self.v_id = v_id  # Whether vertex should be given a unique identifier
        self.bidirectional = bidirectional  # Whether to use bidirectional message passing
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.layers = layers

        #dynamically create GRU layers for each layer index
        self.gru_layers_forward = nn.ModuleList([nn.GRU(v_types if i == 0 else self.hidden_size, self.hidden_size, 1) for i in range(layers)])
        self.gru_layers_backward = nn.ModuleList([nn.GRU(self.hidden_size, self.hidden_size, 1) for i in range(layers)])

        #GRU for vertex
        self.GRU_var = nn.GRU(self.hidden_size, self.hidden_size, 1)
        
    
        heads = [1] * (layers - 1) + [1] if heads is None else heads
        assert len(heads) == layers
        self.gat_forward = GAT(layers, heads, [v_types] + [h_dim] * layers, device, reverse=False).gat_layers
        self.gat_reverse = GAT(layers, heads, [h_dim] * (layers + 1), device, reverse=True).gat_layers
    
        
        self.mlp_encoding = nn.Linear(self.hidden_size*self.nvar, self.latent_size)  # Latent space MLP


    
    def cont_loss(self, graph_encodings, b_size, kernel_embeddings):
        '''Contrastive Loss (Global_Global)'''
        # Stack list to tensor if it's a list
        if isinstance(graph_encodings, list):
            graph_encodings = torch.stack(graph_encodings)
        # Compute loss - Element-wise squared difference between matrices
        loss_gg = torch.sum((graph_encodings - kernel_embeddings) ** 2)
        return loss_gg
    
    
    def forward(self, g_in, g_degree, g_relatives, batch_size, kernel_embeddings, reg_solutions, printing=False):
        '''Contrastive forward with Global_Global loss'''
        g_encodings = self.encode(g_in, g_degree, g_relatives)

        # Ensure no in-place modification of `g_encodings`
        g_encodings = [g_encoding.clone().detach().requires_grad_(True) for g_encoding in g_encodings]

        # Compute loss - Element-wise squared difference
        loss = self.cont_loss(g_encodings, batch_size, kernel_embeddings)
        
        if printing:
            return loss, g_encodings
        else:
            return loss
        

    def encode(self, g_in, g_degree, g_relatives):
        # Ensure g_in is a list of graphs
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
                #print(hidden_single_forward)

            #set variables_output layer
            for graph_idx in range(len(g_in)):
                # Collect the encoded values for the three variables (p_1, p_2, p_3)
                for var_idx, p_var in enumerate([p_1, p_2, p_3]):
                    pos = p_var[graph_idx]  # Get the position for the current graph

                    if pos != -1:
                        # Extract the row corresponding to the position from hidden_single_forward
                        variables_output[graph_idx, layer, var_idx] = hidden_single_forward[graph_idx][pos]
                    else:
                        # If position is -1, assign a zero vector (to handle missing variables)
                        variables_output[graph_idx, layer, var_idx] = torch.zeros(self.hidden_size)

            # Reverse propagation only if not at the last layer
            if self.bidirectional and layer != self.layers - 1:
                for w in prop_order:
                    hidden_single_forward = self._conv_propagate_to(g_in, g_degree, g_relatives, max(n_vars) - w - 1, hidden_single_forward, n_vars, layer, reverse=True)
        
        for i, n in enumerate(n_vars):
            hidden_single_forward[i] = hidden_single_forward[i][:n, :self.hidden_size]
        
        #GRU between each single variable through layers (weights shared between variables)    
        hidden_graph = self.apply_gru_to_variables(variables_output)
        g_encoded = self.mlp_encoding(hidden_graph)
        #last_node_features = self.get_last_node_state(g_in, hidden_single_forward)
        #g_encoded = self.mlp_encoding(last_node_features)
        return g_encoded
    
   
    def get_vertex_feature(self, g, v_idx, hidden_single_forward_g=None, layer=0):
        if layer == 0:
            feat = g[1][v_idx, :].to(self.device)
            if feat.shape[0] < self.v_types:
                feat = torch.cat([feat] + [torch.zeros(self.v_types - feat.shape[0]).to(self.device)])
        else:
            assert (hidden_single_forward_g is not None)
            feat = hidden_single_forward_g[v_idx, :]
        return feat.unsqueeze(0).to(self.device)


    def _conv_propagate_to(self, g_in, g_degree, g_relatives, v_idx, hidden_single_forward, n_vert, layer=0,reverse=False):
        # send messages to v_idx and update its hidden state
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

        if self.bidirectional:  # accept messages also from children
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
            # Extract the relevant h_v for the current graph.
            # Since h_v has the shape (1, 4, 200), we need to index properly.
            hv = h_v[0, i, :].unsqueeze(0)  # Now hv has shape (1, 200)

            hv_shape_2 = hv.shape[1]  # Get the feature dimension (200)
            hidden_shape_1 = hidden_single_forward[graphs_idx[i]].shape[1]  # Get the expected hidden dimension

            # Adjust the shapes of hv and hidden_single_forward as needed
            if hv_shape_2 > hidden_shape_1:
                # If hv is larger, pad hidden_single_forward
                pad = nn.ConstantPad1d((0, hv_shape_2 - hidden_shape_1), 0)  # Pad only the last dimension
                hidden_single_forward[graphs_idx[i]] = pad(hidden_single_forward[graphs_idx[i]])
            elif hv_shape_2 < hidden_shape_1:
                # If hv is smaller, pad hv
                pad = nn.ConstantPad1d((0, hidden_shape_1 - hv_shape_2), 0)  # Pad only the last dimension
                hv = pad(hv)  # hv will be of shape (1, hidden_shape_1)
            
            # Assign hv to hidden_single_forward with proper indexing
            hidden_single_forward[graphs_idx[i]][v_idx, :] = hv.squeeze(0) 
   
        return hidden_single_forward


    
    def name_model(self):
        direction = 'bidirectional' if self.bidirectional else 'monodirectional'
        name = 'DAG GNN' 
        print(f'{name} {direction}')
    
    def get_graph_state(self, g_in, hidden_single_forward, decode=False, start=0, offset=0):
        hidden_graph = []
        max_n_nodes = max([g[0].shape[0] for g in g_in])
        
        for i, g in enumerate(g_in):
            n_nodes_g = g[0].shape[0]
            
            if len(hidden_single_forward[i].shape) > 2:
                hidden_single_forward[i] = hidden_single_forward[i].squeeze(0)
            hidden_g = torch.cat([hidden_single_forward[i][v_idx, :].unsqueeze(0) for v_idx in range(start, n_nodes_g - offset)]).unsqueeze(0)
            
            if n_nodes_g < max_n_nodes:
                hidden_g = F.pad(hidden_g, (0, 0, 0, max_n_nodes - n_nodes_g, 0, 0), "constant", 0) #gpt
                # hidden_g = torch.cat([hidden_g, torch.zeros(1, max_n_nodes - n_nodes_g, hidden_g.shape[2]).to(self.device)], 1)  # [1, max_n_nodes, hidden_size]
                
            hidden_graph.append(hidden_g)
        # use as graph state the sum of node states
        hidden_graph = torch.cat(hidden_graph, 0).sum(1).to(self.device)  # [n_batch, hidden_size]
        return hidden_graph

    def get_var_pos(self, g_in):
    # Initialize the lists for positions of each variable
        positions_first = []
        positions_second = []
        positions_third = []

        # Define the target variables
        first_variable = torch.tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                    0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000])
        second_variable = torch.tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                        0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000])
        third_variable = torch.tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                    0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000])

        # Loop through each tensor couple
        for i, couple in enumerate(g_in):
            second_tensor = couple[1]
            
            # Track whether the variable was found
            found_first = False
            found_second = False
            found_third = False

            # Check each row in the second tensor for the first, second, and third variables
            for row_index in range(second_tensor.shape[0]):
                if torch.equal(second_tensor[row_index], first_variable) and not found_first:
                    positions_first.append(row_index)
                    found_first = True  # Mark as found to prevent multiple entries

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

        Args:
        - self: The class instance (with `self.GRU_var` defined).
        - encoded_values: A tensor of shape [batch_size, num_layers, 3, hidden_size].
                        This contains the encoded values for each graph, for each layer,
                        and for each variable.

        Returns:
        - A tensor of shape [batch_size, hidden_size * num_vars], where the final
        GRU-encoded hidden states for each variable are concatenated.
        """
        batch_size, num_layers, num_vars, hidden_size = encoded_values.shape

        #initialize a list to collect the final GRU-encoded hidden states for each variable.
        final_hidden_states = []

        #iterate over the three variables (0: var1, 1: var2, 2: var3)
        for var_idx in range(num_vars):
            #for each variable, we iteratively apply the GRU across the layers.
            
            #initial hidden state for GRU (we start with the first layer's encoding)
            #shape: [1 (num_layers in GRU), batch_size, hidden_size]
            h = encoded_values[:, 0, var_idx].unsqueeze(0)  # Use the first layer's encoding as initial hidden state

            #iterate through layers (starting from the second layer) to apply the GRU
            for layer_idx in range(1, num_layers):
                #current input for the GRU is the next layer's encoded value for this variable
                input_val = encoded_values[:, layer_idx, var_idx].unsqueeze(0)  # [1, batch_size, hidden_size]

                #pass the input and previous hidden state through the GRU
                _, h = self.GRU_var(input_val, h)  # The second return value is the hidden state

            #after processing all layers, store the final hidden state for this variable
            final_hidden_states.append(h.squeeze(0))  # [batch_size, hidden_size]

        #concatenate the hidden states for all variables along the last dimension
        #final_hidden_states will be a list of tensors of shape [batch_size, hidden_size], one for each variable.
        final_hidden_states = torch.cat(final_hidden_states, dim=1)  # Concatenate along hidden_size dimension

        return final_hidden_states  # [batch_size, hidden_size * num_vars]



