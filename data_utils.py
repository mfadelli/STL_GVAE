#import networkx as nx
#import numpy as np
import torch
#import copy
import os
import pickle
import random
import math
#import itertools
#import stl
import phis_generator
import stl_dataset
import contrastive_gen as cont
import conversions as conv

class StlFormulaeDataset:
    def __init__(self, n_graphs, nvars,device,save_path,leaf_prob= 0.45, inner_node_prob=[0.166, 0.166, 0.166, 0.17, 0.166, 0.166],
                 threshold_mean=0.0,threshold_sd=1.0,unbound_prob=0.1,right_unbound_prob=0.2,
                 time_bound_max_range=20, adaptive_unbound_temporal_ops=True,max_timespan=100, max_depth=10):
        self.n_graphs = n_graphs
        self.nvars = nvars
        self.leaf_prob = leaf_prob
        self.inner_node_prob = inner_node_prob
        self.threshold_mean = threshold_mean
        self.threshold_sd = threshold_sd
        self.unbound_prob = unbound_prob
        self.right_unbound_prob = right_unbound_prob
        self.time_bound_max_range = time_bound_max_range
        self.adaptive_unbound_temporal_ops = adaptive_unbound_temporal_ops
        self.max_timespan = max_timespan
        self.max_depth = max_depth
        self.dataset_folder = save_path
        self.device = device
        self.generator = phis_generator.StlGenerator(leaf_prob,inner_node_prob,threshold_mean,threshold_sd,unbound_prob,
                                                    right_unbound_prob,time_bound_max_range, adaptive_unbound_temporal_ops,
                                                    max_timespan, max_depth)
        self.simple_generator = phis_generator.StlGenerator(0.7,inner_node_prob,threshold_mean,threshold_sd,unbound_prob,
                                                    right_unbound_prob,time_bound_max_range, adaptive_unbound_temporal_ops,
                                                    max_timespan, 2)

    @property
    def _n_graphs(self):
        return self.n_graphs

    @_n_graphs.setter
    def _n_graphs(self, ng):
        self.n_graphs = ng

    @staticmethod
    def build_DAG(formula):
        graph, label_dict = stl_dataset.build_dag(formula,representation='Main')
        return graph, label_dict

    @staticmethod
    
    def get_matrices(graph,n_vars, label_dict):
        adj,att,feat = stl_dataset.get_matrices(graph, n_vars, label_dict, representation='Main')
        tot_att = torch.cat((att, feat), dim=1)
        return adj, tot_att

    @staticmethod
    def get_input(input_formula, n_vars):
        g, lab_dic = StlFormulaeDataset.build_DAG(input_formula)
        return StlFormulaeDataset.get_matrices(g, n_vars, lab_dic)

    def generate_dataset(self, file_name, save=True):
        ''''
        Generate the dataset.
        100 formulae are sampled with fewer nodes, 
        '''
        graphs = []
        formulae=[]
        n_gen_var = 0
        while n_gen_var < self.n_graphs:
            phi = self.generate_formula(graphs, formulae)
            g = StlFormulaeDataset.get_input(phi, self.nvars)
            graphs.append(g)
            formulae.append(phi)
            n_gen_var+=1
        if save:
            with open(self.dataset_folder + os.path.sep + file_name, 'wb') as f:
                pickle.dump(graphs, f)
        return graphs

    def generate_formula(self, graphs, formulae):
        ''''
        Generate a formula for the dataset 
        '''
        if len(graphs) < 100:
            phi = self.simple_generator.sample(self.nvars)  #first 100 formulae have fewer nodes, simpler sampler
        elif len(graphs) >= 100 and len(graphs) <= (self.n_graphs *0.7): #formulae sampled from the more complex sample
            phi = self.generator.sample(self.nvars)  
        else: #data augmentation
            ran_index = torch.randint(0, min(len(graphs), math.floor(self.n_graphs *0.7)),(1,)) #contrastive on a 'clean' formula
            rand_contrastive = torch.randint(0, 10,(1,))
            if rand_contrastive < 3: #20% positive instance
                phi = cont.positive_instance(formulae[ran_index], 'random', self.nvars)
                
            else: #80% negative instance
                phi = cont.negative_instance(formulae[ran_index], 'random', self.nvars)
                
        return phi

        


class StlFormulaeLoader:
    def __init__(self, n_graphs, nvars,device,save_path,leaf_prob= 0.3, inner_node_prob=[0.166, 0.166, 0.166, 0.17, 0.166, 0.166],
                 threshold_mean=0.0,threshold_sd=1.0,unbound_prob=0.1,right_unbound_prob=0.2,
                 time_bound_max_range=20, adaptive_unbound_temporal_ops=True, max_timespan=100, max_depth=6):
        self.n_graphs = n_graphs
        self.nvars = nvars
        self.leaf_prob = leaf_prob
        self.inner_node_prob = inner_node_prob
        self.threshold_mean = threshold_mean
        self.threshold_sd = threshold_sd
        self.unbound_prob = unbound_prob
        self.right_unbound_prob = right_unbound_prob
        self.time_bound_max_range = time_bound_max_range
        self.adaptive_unbound_temporal_ops = adaptive_unbound_temporal_ops
        self.max_timespan = max_timespan
        self.max_depth = max_depth
        self.dataset_folder = save_path
        self.device = device
        self.generator = StlFormulaeDataset(self.n_graphs,self.nvars,self.device,self.dataset_folder,leaf_prob,inner_node_prob,threshold_mean,threshold_sd,unbound_prob,
                                                    right_unbound_prob,time_bound_max_range, adaptive_unbound_temporal_ops,
                                                    max_timespan, max_depth,)
        
    def get_data(self, kind=None, save=True, dvae=True, min_depth=0):
        if not os.path.exists(self.dataset_folder):
            os.makedirs(self.dataset_folder)
        files = os.listdir(self.dataset_folder)
        train_list, test_list, val_list = [[i for i in files if i.startswith(j)] for j in
                                           ['training', 'test', 'validation']]
        if kind == 'train':
            if train_list:
                with open(self.dataset_folder + os.path.sep + train_list[0], 'rb') as f:
                    data = pickle.load(f)
            else:
                # generate name
                train_name = 'training_data_p={}_max-depth={}.pickle'.format(self.leaf_prob, self.max_depth)
                val_name = 'validation_data_p={}_max-depth={}.pickle'.format(self.leaf_prob, self.max_depth)
                #test_name = 'test_data_p={}_max-depth={}.pickle'.format(self.leaf_prob, self.max_depth)
                all_data = self.generator.generate_dataset(False,False)
                random.shuffle(all_data)
                
                #VALIDATION SIZE IS 2.5% of the whole dataset
                train_end_idx = int(0.975*len(all_data)) 
                #val_end_idx = int(0.90*len(all_data)) if (int(0.90*len(all_data)) - train_end_idx) < 100 \
                #    else train_end_idx + 100
                data = all_data[:train_end_idx]
                validation_data = all_data[train_end_idx+1:]
                #test_data = all_data[train_end_idx+1:]
                if save:
                    with open(self.dataset_folder + os.path.sep + train_name, 'wb') as f:
                        pickle.dump(data, f)
                with open(self.dataset_folder + os.path.sep + val_name, 'wb') as f:
                    pickle.dump(validation_data, f)
                #with open(self.dataset_folder + os.path.sep + test_name, 'wb') as f:
                #    pickle.dump(test_data, f)
                
        elif kind == 'test':
            if test_list:
                with open(self.dataset_folder + test_list[0], 'rb') as f:
                    data = pickle.load(f)
            else:
                self.generator.n_graphs = 100
                # generate name
                name = 'test_data_p={}_max-depth={}.pickle'.format(self.leaf_prob, self.max_depth)
                data = self.generator.generate_dataset(name,True)
                
        elif kind == 'validation':
            if val_list:
                with open(self.dataset_folder + os.path.sep + val_list[0], 'rb') as f:
                    data = pickle.load(f)
            else:
                ng = 100
                self.generator.n_graphs = ng
                # generate name
                name = 'validation_data_p={}_max-depth={}.pickle'.format(self.leaf_prob, self.max_depth)
                data = self.generator.generate_dataset(name, save=True)
        #print(data)        
        attr_idx = 1 
        data = [[d[0].to(self.device), d[attr_idx].to(self.device)] for d in data]
        #print([data[0]])
        if dvae and kind in ['train','test', 'validation']:
            data = StlFormulaeLoader.get_dvae_input(data)
        return data

    @staticmethod
    def get_dvae_input(data):
        dvae_data = []
        for d in data:
            adj, attr = [d[0], d[1]]
            n_types = attr.shape[1] + 2  # need to add start node and end node
            root_row_idx = torch.where(~torch.any(adj, dim=0))[0]
            if len(root_row_idx) > 1:  # unique root node
                root_row_idx = root_row_idx[0]
            leaf_row_idx = torch.where(~torch.any(adj, dim=1))[0] + 1  # + 1 to take into account start node index
            # adjust adjacency matrix
            new_adj = torch.zeros((adj.shape[0] + 2, adj.shape[1] + 2))  # start and end nodes
            new_adj[1:-1, 1:-1] = torch.clone(adj)  # start node is first row, end node is last row
            new_adj[0, root_row_idx + 1] = 1
            new_adj[leaf_row_idx, -1] = 1  # leaves are connected to end node
            # adjust feature matrix
            new_attr = torch.zeros((new_adj.shape[0], n_types))
            new_attr[1:-1, 2:] = torch.clone(attr)  # 0-th type is start node, 1-th type is end node
            new_attr[0, 0] = 1  # 0-th type is start type (row 0)
            new_attr[-1, 1] = 1  # 1-th type is end type  (last row)
            dvae_data.append([new_adj, new_attr])
        return dvae_data

    @staticmethod
    def divide_batches(dataset, batch_size, n_data):
        batches = []
        for b in range(len(dataset) // batch_size):
            batches.append(dataset[b * batch_size:b * batch_size + batch_size])
        last_idx = (len(dataset) // batch_size - 1) * batch_size + batch_size
        if last_idx < n_data - 1:
            batches.append(dataset[last_idx:])
        return batches

    def load_batches(self, arg, batch_size, save=True, dvae=True, min_depth=0):
        if not os.path.exists(self.dataset_folder):
            os.makedirs(self.dataset_folder)
        dataset = self.get_data('train',True,True,0)
        #files = os.listdir(self.dataset_folder)
        #dataset_prefix = [f for f in files if f.startswith('train')]
        #dataset_name = dataset_prefix[0] if len(dataset_prefix) > 0 else 'train_current'
        n_data = len(dataset)
        batch_size = batch_size #arg.batch_size
        #if dvae:
        #    dataset = StlFormulaeLoader.get_dvae_input(dataset)
        batches = StlFormulaeLoader.divide_batches(dataset, batch_size, n_data)
        return batches, n_data
    




#UTILITY TO GENERATE NEIGHBOUR AND DEGREE INFO ON VERTICES
def get_structure_info_flattened(graphs, device):
    '''
    Given a list of graphs returns a couple of list
    the first contains for each node of each graphs (output_degree, input_degree, total_degree)
    the second contains (parent_nodes_indices, child_node_indices)
    '''

    degree = []
    relatives = []
    
    for graph in graphs:
        adj_matrix = graph[0].to(device)  #move the adjacency matrix to GPU
        
        #calculate out-degrees and in-degrees
        out_degrees = torch.sum(adj_matrix, dim=1).int()
        in_degrees = torch.sum(adj_matrix, dim=0).int()
        degrees = (out_degrees + in_degrees).int()
        
        #prepare degree information
        degree_v = list(zip(out_degrees.cpu().tolist(), in_degrees.cpu().tolist(), degrees.cpu().tolist()))
        degree.append(degree_v)
        
        #calculate parents and children
        parents = (adj_matrix == 1).nonzero(as_tuple=True)
        children = (adj_matrix.t() == 1).nonzero(as_tuple=True)
        
        relatives_v = []
        for row in range(adj_matrix.size(0)):
            parent_nodes = parents[0][parents[1] == row].cpu().tolist()
            child_nodes = children[0][children[1] == row].cpu().tolist()
            relatives_v.append((parent_nodes, child_nodes))
        
        relatives.append(relatives_v)
    
    return degree, relatives


