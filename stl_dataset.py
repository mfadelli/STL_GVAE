import networkx as nx
import torch

import stl
from phis_generator import StlGenerator


def get_name_given_type(formula):
    '''
    Returns the type of node (as a string) of the top node of the formula/subformula 
    '''
    name_dict = {stl.And: 'and', stl.Or: 'or', stl.Not: 'not', stl.Eventually: 'F', stl.Globally: 'G', stl.Until: 'U',
                 stl.Atom: 'x'}
    return name_dict[type(formula)]


def get_id(child_name, name, label_dict, idx):
    '''
    Get unique identifier for a node
    '''
    while child_name in label_dict.keys():  #if the name is already present
        idx += 1
        child_name = name + "(" + str(idx) + ")"
    return child_name, idx                  #returns both the child name and the identifier 


# TODO: maybe change right unbound case (?)
def get_temporal_list(temporal_node, variant, num_arg):
    '''
    Returns the features vector for temporal nodes (the two bounds of the temporal interval)
    Variant and num_arg modify the length of the list to return (3,4 or 5)
    '''
    left = float(temporal_node.left_time_bound) if temporal_node.unbound is False else 0.
    right = float(temporal_node.right_time_bound) if (temporal_node.unbound is False and
                                                      temporal_node.right_unbound is False) else -1.
    if variant == 'threshold-sign' or variant == 'all-in-var': 
        vector_l=[left, right, 0.]      #third slot for sign and fourth for threshold
    if variant == 'original':
        vector_l = [left, right, 0.]        #third slot for threshold 
    if num_arg==True:
        vector_l.append(0)                  #add an additional slot for argument number
    return vector_l

def add_internal_child(current_child, current_idx, label_dict):
    child_name = get_name_given_type(current_child) + '(' + str(current_idx) + ')'
    child_name, current_idx = get_id(child_name, get_name_given_type(current_child), label_dict, current_idx)
    return child_name, current_idx


def add_leaf_child(node, name, label_dict, idx, variant, shared_var, num_arg, until_right=False):
    ''' 
    Add the edges and update the label_dictionary and the identifier count for a leaf node (variable)
    variant = ['original', 'threshold-sign', 'all-in-var']
    shared_var = [True, False] denotes if shared variables for all the DAG or single variables (tree-like)
    num_arg = [True, False] if true argument number is one-hot encoded in the feature vector
    until_right is a flag to detect when the argument number encoding should be 1 
    '''
    new_e = []
    if variant == 'threshold-sign' or variant == 'all-in-var':
        label_dict[name] = [0., 0., 0.]     #temp_left, temp_right, threshold, sign
    if variant == 'original':
        label_dict[name] = [0., 0., 0.]         #temp_left, temp_right, threshold
    if num_arg== True:                      
        label_dict[name].append(0.)             #add another slot for argument index
    if shared_var==True:
        atom_idx = str(node).split()[0]
    if shared_var==False:
        atom_idx =str(node).split()[0] +  '(' + str(idx) + ')'  
        #different names for the same variables (e.g. x_1(5), x_1(8)) 
        idx += 1   
    if atom_idx not in label_dict.keys():
        if variant == 'threshold-sign' or variant == 'all-in-var':
            label_dict[atom_idx] = [0., 0., 0.]
        if variant == 'original':
            label_dict[atom_idx] = [0., 0., 0.]
        if num_arg== True:
            label_dict[atom_idx].append(0.)
    if variant== 'threshold-sign':
        atom_sign_t =  't' + '(' + str(idx) + ')'  + ' ' + str(node).split()[1]
        if atom_sign_t not in label_dict.keys():
            if str(node).split()[1]=='<=':
                label_dict[atom_sign_t] = [0., 0., round(node.threshold, 4)] #0 encode <=
            else:
                label_dict[atom_sign_t] = [0., 0., round(node.threshold, 4)] #1 encode >=
            if num_arg== True:
                label_dict[atom_sign_t].append(0.)
        new_e.append([name, atom_sign_t])
        new_e.append([atom_sign_t, atom_idx])
        #in the threshold-sign case we have a node [name] as a placeholder, connected to a node
        #for threshold and sign [atom_sign_t], connected with the variable node [atom_idx]
    if variant == 'original': 
        new_e.append([name, atom_idx])
        atom_sign = atom_idx + ' ' + str(node).split()[1]
        if atom_sign not in label_dict.keys():
            label_dict[atom_sign] = [0., 0., 0.]
            if num_arg== True:
                label_dict[atom_sign].append(0.)
            new_e.append([atom_idx, atom_sign])
        new_e.append([name, atom_sign])
        atom_thresh = 't' + '(' + str(idx) + ')'
        atom_thresh, current_idx = get_id(atom_thresh, 't', label_dict, idx)
        new_e.append([name, atom_thresh])
        new_e.append([atom_sign, atom_thresh])
        label_dict[atom_thresh] = [0., 0., round(node.threshold, 4)]
        #in the original case we have a node [name] as a placeholder, connected to a node
        #for sign [atom_sign], a node for the threshold [atom_thresh] and a node for the variable [atom_idx]
        #[atom_idx] is connected with [atom_sign] which is connected to [atom_thresh]
    if variant == 'all-in-var':
        if str(node).split()[1]=='<=':
            label_dict[name] = [0., 0., round(node.threshold, 4)]
        else:
            label_dict[name] = [0., 0., round(node.threshold, 4)]
        if num_arg== True:
            label_dict[name].append(0.)
        new_e.append([name, atom_idx])
        #in the all-in-var case we only have two nodes: the placeholder [name] with all the features 
        #connected with the variable node 
    if num_arg==True and until_right==True:
        label_dict[name][-1] = 1            #right child of an until node only case of argument slot not 0
    return new_e, label_dict, idx+1


def traverse_formula(formula, idx, label_dict, variant, shared_var, num_arg,until_flag=False):
    """
        DFS traverse of the AST of the formula

        Parameters
        ----------
        formula : formula on which to perform the DFS
        
        idx : counter for unique identifiers
            it should start from 0 outside of recursive calls
        
        label_dict : dictionary for the nodes and their features
            it should start from {} outside of recursive calls
        
        variant : ['original','threshold-sign','all-in-var']
            three different variants, see the comments in add_leaf_child for more info

        shared_var :  shared variable nodes
            if false it creates a tree-like structure with different variable nodes

        num_arg : whether to keep track of the argument number with edges (False) or with nodes (True)
            if False edges between first and second argument, if True only encoded in the node
        
        until_flag : flag for right child of until
            useful only in the case num_arg is True      

        Returns
        -------
        A list of the edges, the dictionary of the node features

        """
    current_node = formula
    edges = []
    if type(current_node) is not stl.Atom:
        current_name = get_name_given_type(current_node) + '(' + str(idx) + ')'
        if (type(current_node) is stl.And) or (type(current_node) is stl.Or) or (type(current_node) is stl.Not):
            if variant == 'threshold-sign' or variant == 'all-in-var':
                label_dict[current_name] = [0., 0., 0. ] #temp_left, temp_right, threshold, sign (eliminated)
            if variant == 'original':
                label_dict[current_name] = [0., 0., 0.]     #temp_left, temp_right, threshold
            if num_arg== True:
                label_dict[current_name].append(0.)         #slot for argument number
        else:
            label_dict[current_name] = get_temporal_list(current_node, variant, num_arg)
        if num_arg==True and until_flag==True:
            label_dict[current_name][-1]=1              #right child of until
        if (type(current_node) is stl.And) or (type(current_node) is stl.Or) or (type(current_node) is stl.Until):
            left_child_name, current_idx = add_internal_child(current_node.left_child, idx + 1, label_dict)
            edges.append([current_name, left_child_name])
            if type(current_node.left_child) is stl.Atom:
                e, d, current_idx = add_leaf_child(current_node.left_child, left_child_name, label_dict, current_idx+1, variant, shared_var, num_arg)
                edges += e
                label_dict.update(d)
            e, d = traverse_formula(current_node.left_child, current_idx, label_dict, variant, shared_var, num_arg)
            edges += e
            label_dict.update(d)
            right_child_name, current_idx = add_internal_child(current_node.right_child, current_idx + 1, label_dict)
            if type(current_node) is stl.Until:
                if num_arg==False:
                    edges.append([left_child_name, right_child_name])
                else:
                    until_flag=True
            edges.append([current_name, right_child_name])
            if type(current_node.right_child) is stl.Atom:
                e, d, current_idx = add_leaf_child(current_node.right_child, right_child_name, label_dict,
                                                   current_idx+1,variant,shared_var, num_arg,until_right=until_flag)
                edges += e
                label_dict.update(d)
            e, d = traverse_formula(current_node.right_child, current_idx, label_dict, variant, shared_var, num_arg, until_flag=until_flag)
            edges += e
            label_dict.update(d)
        else:
            # eventually, globally, not
            child_name, current_idx = add_internal_child(current_node.child, idx + 1, label_dict)
            edges.append([current_name, child_name])
            if type(current_node.child) is stl.Atom:
                e, d, current_idx = add_leaf_child(current_node.child, child_name, label_dict, current_idx+1,  variant, shared_var, num_arg)
                edges += e
                label_dict.update(d)
            e, d = traverse_formula(current_node.child, current_idx, label_dict,  variant, shared_var, num_arg)
            edges += e
            label_dict.update(d)
    return edges, label_dict


def build_dag(formula, representation, variant='all-in-var', shared_var=True, num_arg=False):
    """
        Build a DAG from a formula

        Parameters
        ----------
        formula : formula to be represented by a DAG

        representation = ['original', 'T1', 'T2', 'DAG1', 'DAG2']
            shortcut for some specific representations
        
        variant, shared_var, num_arg : parameters for tree representation
            chosen automatically with the representation. Can be chosen manually too

        Returns
        -------
        The graph, the dictionary of the nodes and their features

        """
    variant, shared_var, edge_feats, num_var = rep_to_pars(representation)  
    edges, label_dict = traverse_formula(formula, 0, {}, variant, shared_var, num_arg)
    graph = nx.from_edgelist(edges, create_using=nx.DiGraph)
    assert(nx.is_directed_acyclic_graph(graph))
    return graph, label_dict


def get_matrices(graph, n_vars, feat_dict, representation, variant='all-in-var', shared_var=True, edge_feats='single'):
    '''
    Given a graph and the node features it outputs the adjacency matrix, the attribute matrix and the feature tensor
    
    Parameters
    ----------
    graph : nx graph to be taken as input

    n_vars : number of variables (indices) to be encoded

    feat_dict : dictionary for node features

    representation = ['original', 'T1', 'T2', 'DAG1', 'DAG2']
            shortcut for some specific representations
    
    variant, shared_var, edge_feats : can be manually set but automatically obtained by specifying representation

    '''
    def find_index(d, s):
        result = []
        keys_length = len(d.keys())
        for ss in s:
            if ss in d:
                result.append(d[ss])
            else:
                result.append(keys_length + ss)
        return result

    def get_name(n):
        if '(' in n:
            if '_' in n and n.index('_') < n.index('('):
                prefix = n.partition('(')[0]
                return int(prefix.partition('_')[2])
            return n.partition('(')[0]
        elif '<' in n:
            return '<'
        elif '>' in n:
            return '>'
        else:
            return int(n.partition('_')[2])
    #obtains the parameters for the graph building from the representation chosen
    if rep_to_pars(representation) is not None:
        variant, shared_var, edge_feats, num_var = rep_to_pars(representation)  
    adj = torch.from_numpy(nx.to_numpy_array(graph))
    #different variants have different attribute matrices
    if variant=='original':
        one_hot_dict_idx = {'and': 0, 'or': 1, 'not': 2, 'F': 3, 'G': 4, 'U': 5, 'x': 6, 't': 7, '<': 8, '>': 9}
    if variant=='threshold-sign':
        one_hot_dict_idx = {'and': 0, 'or': 1, 'not': 2, 'F': 3, 'G': 4, 'U': 5, 'x': 6, 't': 7}
    if variant=='all-in-var':
        one_hot_dict_idx = {'and': 0, 'or': 1, 'not': 2, 'F': 3, 'G': 4, 'U': 5, 'x': 6}
    names = list(map(get_name, graph.nodes()))
    feat = torch.cat([torch.tensor(feat_dict[n]).unsqueeze(0) for n in graph.nodes()], dim=0)
    idx_list = find_index(one_hot_dict_idx, names)
    attr = torch.zeros((adj.shape[0], len(list(one_hot_dict_idx.keys())) + n_vars))
    attr[torch.arange(attr.shape[0]), idx_list] = 1
    #create eventual edge "matrix"
    if edge_feats == 'cumulative':
        edg = get_edge_attr(adj,attr,feat,True)
        return adj, attr, feat, edg
    elif edge_feats == 'single':
        edg = get_edge_attr(adj,attr,feat,False)
        return adj, attr, feat, edg
    elif edge_feats == False:
        return adj, attr, feat

def rep_to_pars(representation):
    '''
    representation in ['original', 'T1', 'T2', 'DAG1', 'DAG2']
    '''
    if representation == 'original':
        variant, shared_var, edge_feats, num_var = 'original', True, False, False
    elif representation == 'T1':
        variant, shared_var, edge_feats, num_var = 'all-in-var', False, False, True
    elif representation == 'T2':
        variant, shared_var, edge_feats, num_var = 'all-in-var', False, 'single', True
    elif representation == 'DAG1':
        variant, shared_var, edge_feats, num_var = 'all-in-var', False, False, False
    elif representation == 'DAG2':
        variant, shared_var, edge_feats, num_var = 'all-in-var', True, 'single', True
    elif representation == 'Main':
        variant, shared_var, edge_feats, num_var = 'all-in-var', True, False, False
    else:
        return None
    return variant, shared_var, edge_feats, num_var


def generate_edge_features(row, col, att_par, att_child, feat_par, feat_child, edge_matrix, cum_time):
    '''Generate the edge features vector for an edge of the graph'''
    e_feat = [0., 0.]
    par_type = int(torch.nonzero(att_par, as_tuple=False))
    if cum_time == False:
        if par_type in [3, 4, 5]:
            feat_par_list = feat_par[:2].tolist()
            e_feat = feat_par_list
    elif cum_time == True:
        par_edge_index = next((i for i, row in enumerate(edge_matrix) if row[col]), None)
        par_edge_index_2 = next((i for i, row in enumerate(edge_matrix) if row[par_edge_index]), None)
        if par_edge_index_2 is not None:
            e_feat = edge_matrix[par_edge_index_2][par_edge_index][:2]
        if par_type in [3, 4, 5]:
            feat_par_list = feat_par[:2].tolist()
            e_feat = [x + y for x, y in zip(e_feat, feat_par_list)]
    return e_feat

def get_edge_attr(adj, attr, feat, cum_time=False):
    '''
    Generate the edge matrix given adjacency, attribute and feature matrices
    
    cum_time : cumulative time
        if True times for temporal interval is summed. All edges will have an interval.
    '''
    edge_features = [[[] for _ in range(adj.size(1))] for _ in range(adj.size(0))]
    for j in range(adj.size(1)):
        for i in range(adj.size(0)):
            #loop column-wise        
            if adj[i, j] == 1:
                edge_features[i][j]=[0.,0.]
                edge_features[i][j] = generate_edge_features(
                i,j,attr[i],attr[j],feat[i],feat[j],edge_features, cum_time)
    return edge_features

def main_representation(formula,nvar):
    '''
    Main representation.
    The formula is already in 'only <=' form.
    Returns the attribute and feature matrices
    '''
    dag, nodes_d = build_dag(formula,representation='Main')
    adj,att,feat = get_matrices(dag, nvar, nodes_d, representation='Main')
    return adj,att,feat, dag,

#TODO: double check all the parameters combinations
#TODO: in 'cumulative' what to assign to the edge between the two until? How to detect it? Also is the
#parent correctly chosen between the two? (probably not)
#TODO: add argument for number argument as edge feature


