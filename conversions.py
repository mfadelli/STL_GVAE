import torch
import networkx as nx
import stl
import copy



                        ######################################################################
                        #                  Utilities for manipulating matrices               #                  
                        ######################################################################


def matrices_to_formula(adj_mat,attr_mat, n_vars, dvae=True):
    '''Convert a couple (adjacency_matrix,attribute_matrix) to an STL formula. 
    dvae specifies if the matrices contains beginning and end node'''
    adj = adj_mat.clone()  
    attr = attr_mat.clone()  
    if dvae == True:
        #remove the beginning and end node
        attr = attr[1:-1, :]
        adj = adj[1:-1, 1:-1]
    #locate first node
    column_sums = torch.sum(torch.abs(adj), dim=0)
    first_node = torch.nonzero(column_sums == 0.0).squeeze().tolist()
    if isinstance(first_node,int) == False:
        first_node= first_node[0]
    #create a dictionary {'node1':[stl.And/None,['node_2','node_3']]}
    nodes = {}   
    vars = [] #contains the row of the variables
    var_indexes=[] #contains the respective indexes of the variable
    for i in range(adj.size(0)): #rows
        if torch.all(adj[i] == 0): # variable vectors
            vars.append(i)
            for variable in range(0,n_vars):
                if attr[i][9+variable]==1.0:
                    var_indexes.append(variable)
                    break 
    all_signs=[]
    for j in range(0,len(vars)): #for each variable
        variable = vars[j]
        signs = torch.nonzero(adj[:,variable] == 1.0).squeeze().tolist() #list of all the signs connected to the variable
        if type(signs)==int:
            signs=[signs]
        for k in range(0,len(signs)): #for each sign-threshold for the given variable
            threshold = float(attr[signs[k]][-1])
            sign = 1
            node = stl.Atom(var_indexes[j],threshold,sign)
            #add the node to the dictionary
            key = f'node_{signs[k]}' #name is node_q where q is the row in the matrices
            nodes[key] = [[True],node]
        all_signs+=[signs]
    #correctly generates the atoms
    #all_signs is a list of lists (first element is rows of all thresholds of the first variable)
    nodes_list=list(range(0,adj.size(0)))
    flattened_signs = [item for sublist in all_signs for item in sublist]
    n_list = [item for item in nodes_list if item not in vars and item not in flattened_signs]
    #n_list contains all the rows that are not variable or thresholds
    for i in range(0,len(n_list)):
        ind = n_list[i]
        key = f'node_{ind}'
        children =  find_real_children(ind,adj,attr)
        if isinstance(children,list) and len(children)==1:
            children=int(children[0])
        nodes[key] = [children, None]
    #created dictionary of all nodes
    it=0
    while any(value[1] is None for value in nodes.values()):
        it+=1
        if it>50:
            print('error in conversion')
            break
        for key, value in nodes.items():
            i = int(key.split('_')[1]) #number of the node
            if type(value[0])== int: #one children
                child = f'node_{value[0]}'
                if value[1] is None and nodes[child][1] is not None: #children has already a node
                    if attr[i][4]==1.0: #NOT
                        value[1]=stl.Not(nodes[child][1])
                    if attr[i][5]==1.0: #eventually
                        int_1 = int(attr[i][-3])
                        int_2 = int(attr[i] [-2])-1
                        value[1]=stl.Eventually(nodes[child][1],False,False,int_1,int_2)
                    if attr[i][6]==1.0: #globally
                        int_1 = int(attr[i] [-3])
                        int_2 = int(attr[i] [-2])-1
                        value[1]=stl.Globally(nodes[child][1],False,False,int_1,int_2)

            elif len(value[0])==2: #two children
                left_child = f'node_{value[0][0]}'
                right_child = f'node_{value[0][1]}'
                if value[1] is None and nodes[right_child][1] is not None and nodes[left_child][1] is not None:
                    if attr[i][2]==1.0: #AND
                        value[1]=stl.And(nodes[left_child][1],nodes[right_child][1])
                    if attr[i][3]==1.0: #OR
                        value[1]=stl.Or(nodes[left_child][1],nodes[right_child][1])
                    if attr[i][7]==1.0: #UNTIL
                        int_1 = int(attr[i] [-3])
                        int_2 = int(attr[i] [-2])-1
                        if adj[value[0][0]][value[0][1]] == 1.0:
                            value[1]=stl.Until(nodes[left_child][1],nodes[right_child][1],False,False,int_1,int_2)
                        elif adj[value[0][1]][value[0][0]] == 1.0:
                            value[1]=stl.Until(nodes[right_child][1],nodes[left_child][1],False,False,int_1,int_2)
    f_node = f'node_{first_node}'
    return nodes[f_node][1]


def find_real_children(node_index,adj,attr):
    '''find real children of a node for an until node'''
    children =  torch.nonzero(adj[node_index] == 1.0).squeeze().tolist()
    #it may contain right child of an until (a sibling)
    parent_node = find_parent_1(node_index,adj,attr)
    if parent_node is None:
        return children
    elif attr[parent_node][7]==1.0: #parent is an Until 
        if type(children)==int:
            return [children]
        else:
            for c in children:
                parent_c=find_parent_1(c,adj,attr)
                if parent_node == parent_c:
                    children.remove(c)
    return children


def find_parent_1(sel_point,adj,attr):
    '''find teh parent of a node (in the case of an until node)'''
    parent=None
    parents=(adj[:, sel_point] == 1.0).nonzero(as_tuple=True)[0].tolist()
    if len(parents)==1:
        parent=parents[0]
    else:
        for j in parents:
            if attr[j][7]==1.0:
                parent=j
                break
    return parent




def topological_ordering(adj,attr):
    '''Rewrite the matrices in order to have a correct topological ordering'''
    if check_top_ordering(adj) == False:
        new_adj = copy.deepcopy(adj)
        new_attr = copy.deepcopy(attr)
        adj_numpy = adj.numpy()
        G = nx.from_numpy_array(adj_numpy, create_using=nx.DiGraph)
        try:
            order = list(nx.topological_sort(G))
        except nx.NetworkXUnfeasible:
            order = list(nx.reverse_cuthill_mckee_ordering(G))
        

        order = torch.tensor(order, dtype=torch.long)
        
        new_adj = new_adj[order][:, order]
        new_attr = new_attr[order]

        return new_adj, new_attr    
    
    else: #matrices already ordered
        return adj, attr

def check_top_ordering(adj):
    '''check if a graph is topologically ordered'''
    lower_triangular = torch.tril(adj)
    return torch.all(lower_triangular == 0).item()





def tree_pruning(graph,remove_until=True,remove_final=True, remove_variables=True):
    '''Remove variable nodes, until edge, initial and final node to produce cut tree to be reconstructed'''
    adj, attr = graph[0], graph[1]

    #removing variable nodes
    if remove_variables==True:
        mask = (attr[:, 9] == 1) | (attr[:, 10] == 1) | (attr[:, 11] == 1)
        keep_nodes = torch.where(~mask)[0]
        adj_matrix_reduced = adj[keep_nodes][:, keep_nodes]
        feature_matrix_reduced = attr[keep_nodes]
    else:
        adj_matrix_reduced = adj
        feature_matrix_reduced= attr

    #remove final
    final_node = torch.where(feature_matrix_reduced[:, 0] == 1)[0].item()
    if remove_final == True:
        mask = (feature_matrix_reduced[:, 1] == 1)
        keep_nodes = torch.where(~mask)[0]
        adj_matrix_reduced = adj_matrix_reduced[keep_nodes][:, keep_nodes]
        feature_matrix_reduced = feature_matrix_reduced[keep_nodes]
    else:
        no_children_mask = adj_matrix_reduced.sum(dim=1) == 0
        no_children_nodes = torch.where(no_children_mask)[0]
        #connect threshold nodes to the final node
        for node in no_children_nodes:
            if node != final_node:
                # Connect the isolated node to the final node
                adj_matrix_reduced[node, final_node] = 1
                adj_matrix_reduced[final_node, node] = 1
    
    #remove until
    if remove_until == True:
        until_nodes = torch.where(feature_matrix_reduced[:, 7] == 1)[0]
        for until_node in until_nodes:
            # Find the children of the 'until' node (nodes connected via outgoing edges)
            children = torch.where(adj_matrix_reduced[until_node] == 1)[0]
            
            # Ensure the node has exactly two children
            if len(children) == 2:
                first_child, second_child = children[0], children[1]

                # Remove the edge from the first child to the second child
                adj_matrix_reduced[first_child, second_child] = 0
                adj_matrix_reduced[second_child, first_child] = 0
    return [adj_matrix_reduced,feature_matrix_reduced]





def topological_info(graph):
    '''Generates a list of topological properties of a graph'''
    adj, attr = graph[0], graph[1]
    
    # Move adjacency matrix to CPU and convert to NumPy
    adj_matrix_np = adj.cpu().numpy()
    G = nx.DiGraph(adj_matrix_np)

    # 1. Number of nodes
    t_1 = G.number_of_nodes()
    
    # 2. Depth
    root_nodes = [node for node, in_degree in dict(G.in_degree()).items() if in_degree == 0]
    if nx.is_directed_acyclic_graph(G):
        t_2 = nx.dag_longest_path_length(G) 
    else:
        t_2 = 0
    
    # 3. Threshold nodes
    eighth_column = attr[:, 8].cpu()  # Move to CPU
    t_3 = torch.count_nonzero(eighth_column).item()
    
    # 4. Logical operators
    selected_columns = attr[:, 2:5].cpu()  # Move to CPU
    non_zero_any_column = torch.any(selected_columns != 0, dim=1)
    t_4 = torch.count_nonzero(non_zero_any_column).item()
    
    # 5. Temporal operators
    selected_columns = attr[:, 5:8].cpu()  # Move to CPU
    non_zero_any_column = torch.any(selected_columns != 0, dim=1)
    t_5 = torch.count_nonzero(non_zero_any_column).item()
    
    # 6-7-8. Threshold for each variable
    variables_columns = attr[:, 9:12].cpu()  # Move to CPU
    variable_1_node = (variables_columns[:, 0] == 1).nonzero(as_tuple=True)[0]
    variable_2_node = (variables_columns[:, 1] == 1).nonzero(as_tuple=True)[0]
    variable_3_node = (variables_columns[:, 2] == 1).nonzero(as_tuple=True)[0]
    t_6 = count_predecessors(adj.cpu(), variable_1_node)  # Move `adj` to CPU
    t_7 = count_predecessors(adj.cpu(), variable_2_node)  # Move `adj` to CPU
    t_8 = count_predecessors(adj.cpu(), variable_3_node)  # Move `adj` to CPU
    
    # 9-10-11. Average depth for each variable
    t_9 = average_path_length_to_variable(adj.cpu(), root_nodes, variable_1_node) + 1  # Move `adj` to CPU
    t_10 = average_path_length_to_variable(adj.cpu(), root_nodes, variable_2_node) + 1  # Move `adj` to CPU
    t_11 = average_path_length_to_variable(adj.cpu(), root_nodes, variable_3_node) + 1  # Move `adj` to CPU
    
    # 12. Imbalance index
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    imbalance_index = sum(abs(in_degrees[node] - out_degrees[node]) for node in G.nodes()) / G.number_of_nodes()
    t_12 = imbalance_index
    
    return [t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8, t_9, t_10, t_11, t_12]

def count_predecessors(adj_matrix, node_idx):
    if node_idx.numel() == 0:
        return 0  
    node_idx = node_idx.item()  
    return torch.sum(adj_matrix[:, node_idx]).item() 

def find_paths_lengths(adj_matrix, root, target, current_length=0):
    if root == target:
        return [current_length]  
    paths = []
    for i in range(adj_matrix.size(1)):  #traverse all possible edges
        if adj_matrix[root, i] == 1:  
            paths.extend(find_paths_lengths(adj_matrix, i, target, current_length + 1))
    return paths


def average_path_length_to_variable(adj_matrix, root_nodes, variable_node):
    '''Function to calculate the average path length from roots to a variable node'''
    if variable_node.numel() == 0:
        return 0.0  #if the variable node doesn't exist, return 0
    variable_node = variable_node.item()
    path_lengths = []
    for root in root_nodes:
        path_lengths.extend(find_paths_lengths(adj_matrix, root, variable_node))
    if len(path_lengths) == 0:
        return float('inf')  #no paths found, return inf
    return sum(path_lengths) / len(path_lengths)  



def dfs_reorder(adj_matrix, start_node=0):
    """
    Performs a DFS traversal on the graph represented by the adjacency matrix
    and returns the node visit order. It starts from the given start_node.
    Returns a list of nodes in the order they were visited during DFS.
    """
    num_nodes = adj_matrix.shape[0]
    visited = [False] * num_nodes
    dfs_order = []

    def dfs(node):
        visited[node] = True
        dfs_order.append(node)
        #visit all neighbors (successors) of the current node
        for neighbor in range(num_nodes):
            if adj_matrix[node, neighbor] == 1 and not visited[neighbor]:
                dfs(neighbor)

    #start DFS from the specified start_node
    dfs(start_node)

    #in case there are other components or isolated nodes, continue DFS
    for node in range(num_nodes):
        if not visited[node]:
            dfs(node)

    return dfs_order

def dfs_ordered(g_true, start_node=0):
    """
    Reorders the adjacency matrix and attribute matrix of g_true based on the DFS traversal order.
    Returns the reordered adjacency and attribute matrices based on DFS order.
    """
    adj_matrix, attr_matrix = g_true[0], g_true[1]
    #perform DFS on the adjacency matrix to get the DFS order
    dfs_order = dfs_reorder(adj_matrix, start_node)

    # Reorder the adjacency and the attribute matrices based on DFS order
    adj_matrix_dfs = adj_matrix[dfs_order, :][:, dfs_order]
    attr_matrix_dfs = attr_matrix[dfs_order, :]

    return [adj_matrix_dfs, attr_matrix_dfs]




def trim(graph,add_until=True):
    '''Eliminates unused variables'''
    adj, attr = graph 
    n = adj.size(0)   #number of nodes 

    if add_until==True:
        for i in range(n):
            if attr[i][7]==1: #until node
                until_children=torch.nonzero(adj[i] == 1.0).squeeze().tolist()
                if type(until_children)!= list:
                    print('Matrix generated too big')
                    continue

                if len(until_children)==2:
                    adj[until_children[0],until_children[1]]=1
                else:
                    if len(until_children)==3:
                        until_parent = torch.nonzero(adj[:, i] == 1.0).squeeze().item()
                        until_parent_children = torch.nonzero(adj[until_parent] == 1.0).squeeze().tolist()
                        until_children = list(set(until_children) - set(until_parent_children))
                        adj[until_children[0],until_children[1]]=1
                    else:
                        print('too many children',len(until_children))
 
    #remove isolated nodes
    isolated_nodes = []
    for i in range(1,n):
        if torch.sum(adj[:, i]) == 0:
            isolated_nodes.append(i)
    
    if isolated_nodes:  
        mask = torch.ones(n, dtype=torch.bool)
        mask[isolated_nodes] = False 

        # Use the mask to index rows and columns for the adjacency matrix and attribute matrix
        adj_trimmed = adj[mask][:, mask]
        attr_trimmed = attr[mask]  
        
        return adj_trimmed, attr_trimmed
    else:
        
        return adj, attr

