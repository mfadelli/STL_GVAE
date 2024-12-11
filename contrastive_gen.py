import torch
import stl
import copy
import random
import conversions as conv


                #################################################################################
                #  Functions for generation of positive and negative instances of STL formulae  #
                #              for contrastive learning or data augmentation                    #  
                #          in two versions [directly on matrices/on stl formulae)               #                     
                #################################################################################


#VERSION THAT ACTS DIRECTLY ON MATRICES


def pos_instance_matrices(adj,attr,nvar,type='random'):
    '''
    Given a labelled graph (adj, attr) generates a positive instance (adj,attr), possible instances in 
    ['de_morgan', 'double_neg', 'insert_tautology', 'ev_glo_into_until'].
    A positive instance is a formula with the same semantics of the original formula.
    '''
    pos_adj = copy.deepcopy(adj)
    pos_attr = copy.deepcopy(attr)
   
    if type == 'random':
        type = random.choice(['de_morgan', 'double_neg', 'insert_tautology', 'ev_glo_into_until'])
    
    if type == 'de_morgan':
        list_and = [i for i, row in enumerate(attr) if row[2] == 1.0]   #indices of AND rows
        list_or = [i for i, row in enumerate(attr) if row[3] == 1.0]    #indices of OR rows 
        sel_type = random.choice([0,1]) #select if we modify an OR (0) or a AND (1)

        #check whether one of the two lists is empty (or both)
        if len(list_and)==0 and len(list_or)==0:
            return pos_instance_matrices(adj,attr,nvar,type='random')
                
        elif len(list_and)==0 or (len(list_and)!= 0 and len(list_or)!= 0 and sel_type==1): #change OR-->AND
            random_index = random.randint(0, len(list_or) - 1)
            sel_node = list_or[random_index]
            pos_attr[sel_node][3]=0.0
            pos_attr[sel_node][2]=1.0

        elif len(list_or)==0 or (len(list_and)!= 0 and len(list_or)!= 0 and sel_type==0): #change AND-->OR
            random_index = random.randint(0, len(list_and) - 1)
            sel_node = list_and[random_index]
            pos_attr[sel_node][3]=1.0
            pos_attr[sel_node][2]=0.0

        pos_adj,pos_attr = add_not(sel_node,adj,attr,pos_adj,pos_attr,nvar)
        pos_adj,pos_attr = conv.topological_ordering(pos_adj,pos_attr)
        return pos_adj,pos_attr
    
       
    if type == 'double_neg':
        list_possible = [i for i in range(1, attr.size(0) - 1) if not torch.any(attr[i, 9:9 + nvar] == 1.0)]
        sel_point = random.choice(list_possible)
        #print(sel_point)
        pos_adj,pos_attr = add_not(sel_point,adj,attr,pos_adj,pos_attr,nvar)
        new_point=pos_attr.size(0) - 2
        pos_adj,pos_attr = add_not(new_point,pos_adj,pos_attr,pos_adj,pos_attr,nvar)
        pos_adj,pos_attr = conv.topological_ordering(pos_adj,pos_attr)
        return pos_adj,pos_attr 

    if type =='insert_tautology':
        list_possible = [i for i in range(1, attr.size(0) - 1) if not torch.any(attr[i, 9:9 + nvar] == 1.0)] 
        sel_point = random.choice(list_possible)
        and_attr_row = torch.zeros(12+nvar)
        and_attr_row[2]=1.0 #AND
        or_attr_row = torch.zeros(12+nvar)
        or_attr_row[3] = 1.0 #OR
        threshold = random.gauss(0,1)
        t1_attr_row= torch.zeros(12+nvar)
        t1_attr_row[8]=1.0 #threshold-sign
        t1_attr_row[-1] = threshold
        not_attr_row =torch.zeros(12+nvar)
        not_attr_row[4] = 1.0 #NOT
        t2_attr_row= torch.zeros(12+nvar)
        t2_attr_row[8]=1.0 #threshold-sign
        t2_attr_row[-1] = threshold
        #update attribute matrix
        pos_attr = torch.cat((pos_attr[:-1], and_attr_row.unsqueeze(0), or_attr_row.unsqueeze(0),t1_attr_row.unsqueeze(0) ,not_attr_row.unsqueeze(0), t2_attr_row.unsqueeze(0), pos_attr[-1:]), dim=0)
        #update adjacency matrix
        rows_to_insert = torch.zeros((5, pos_adj.size(1)))
        pos_adj = torch.cat((pos_adj[: -1], rows_to_insert, pos_adj[-1:]), dim=0)
        cols_to_insert = torch.zeros((pos_adj.size(0), 5))
        pos_adj = torch.cat((pos_adj[:, : -1], cols_to_insert, pos_adj[:, -1:]), dim=1)
        parent= find_parent(sel_point,adj,attr)
        pos_adj[parent][sel_point], pos_adj[parent][-6]= 0.0, 1.0
        #if pos_attr[parent][7]== 1.0: TODO fix if the parent is an until
        pos_adj[-6][-5], pos_adj[-6][sel_point]= 1.0, 1.0 #AND
        pos_adj[-5][-4], pos_adj[-5][-3] = 1.0, 1.0 #OR
        pos_adj[-3][-2]= 1.0
        var_nodes = torch.nonzero((attr[:, 9:(8 + nvar + 1)] == 1.0).any(dim=1), as_tuple=True)[0]
        var_chosen = random.choice(var_nodes)
        pos_adj[-4][var_chosen]=1.0 #first child
        pos_adj[-2][var_chosen]=1.0 #second child
        #print(pos_adj,pos_attr)
        pos_adj,pos_attr = conv.topological_ordering(pos_adj,pos_attr)
        return pos_adj,pos_attr


    if type =='ev_glo_into_until':
        globs = [i for i, row in enumerate(attr) if row[6] == 1.0] 
        eves = [i for i, row in enumerate(attr) if row[5] == 1.0] 
        sel_type = random.choice([0,1]) #select if we modify a Globally (0) or an Eventually (1)
        #check whether one of the two lists is empty (or both)
        if len(eves)==0 and len(globs)==0:
            return pos_instance_matrices(adj,attr,nvar,type='random')
        
        elif len(globs)==0 or (len(globs)!= 0 and len(eves)!= 0 and sel_type==1):
            #replace eventually[a,b](phi) with True Until[a,b] phi
            random_index = random.randint(0, len(eves) - 1)
            sel_node = eves[random_index]
            pos_attr[sel_node][5], pos_attr[sel_node][7] = 0.0, 1.0 #changed from eventually to until
            #add the tautology
            or_attr_row = torch.zeros(12+nvar)
            or_attr_row[3] = 1.0
            threshold = random.gauss(0,1)
            t1_attr_row= torch.zeros(12+nvar)
            t1_attr_row[8]=1.0
            t1_attr_row[-1] = threshold
            not_attr_row =torch.zeros(12+nvar)
            not_attr_row[4] = 1.0
            t2_attr_row= torch.zeros(12+nvar)
            t2_attr_row[8]=1.0
            t2_attr_row[-1] = threshold
            #update attribute matrix
            pos_attr = torch.cat((pos_attr[:-1], or_attr_row.unsqueeze(0),t1_attr_row.unsqueeze(0) ,not_attr_row.unsqueeze(0), t2_attr_row.unsqueeze(0), pos_attr[-1:]), dim=0)
            #update adjacency matrix
            rows_to_insert = torch.zeros((4, pos_adj.size(1)))
            pos_adj = torch.cat((pos_adj[: -1], rows_to_insert, pos_adj[-1:]), dim=0)
            cols_to_insert = torch.zeros((pos_adj.size(0), 4))
            pos_adj = torch.cat((pos_adj[:, : -1], cols_to_insert, pos_adj[:, -1:]), dim=1)
            children= conv.find_real_children(sel_node,adj,attr)
            pos_adj[sel_node][-5] = 1.0
            pos_adj[-5][children] = 1.0 #edge between two children of until
            pos_adj[-5][-4],pos_adj[-5][-3] = 1.0, 1.0 #OR
            pos_adj[-3][-2]= 1.0
            #var_nodes = torch.nonzero((attr[:, 9] == 1.0) | (attr[:, 8 + nvar] == 1.0), as_tuple=True)[0]
            var_nodes = torch.nonzero((attr[:, 9:(8 + nvar + 1)] == 1.0).any(dim=1), as_tuple=True)[0]
            if isinstance(var_nodes,int):
                var_nodes=[var_nodes] 
            var_chosen = random.choice(var_nodes)
            pos_adj[-4][var_chosen]=1.0 #first child
            pos_adj[-2][var_chosen]=1.0 #second child
            #print(pos_adj,pos_attr)
            pos_adj,pos_attr = conv.topological_ordering(pos_adj,pos_attr)
            return pos_adj,pos_attr


        elif len(eves)==0 or (len(globs)!= 0 and len(eves)!= 0 and sel_type==0):
            #replace globally[a,b](phi) with Not(eventually[a,b](Not(phi)))
            random_index = random.randint(0, len(globs) - 1)
            sel_node = globs[random_index]
            pos_attr[sel_node][6], pos_attr[sel_node][5] = 0.0, 1.0 #changed globally into eventually
            pos_adj, pos_attr = add_not(sel_node,adj,attr,pos_adj,pos_attr,nvar) 
            child = (pos_adj[sel_node] == 1.0).nonzero(as_tuple=True)[0].item()
            pos_adj, pos_attr = add_not(child,adj,attr,pos_adj,pos_attr,nvar) 
            pos_adj,pos_attr = conv.topological_ordering(pos_adj,pos_attr)
            return pos_adj,pos_attr





def neg_instance_matrices(adj,attr,nvar,type='random'):
    '''
    Given a labelled graph (adj,attr) generates a negative instance, possible instances in 
    ['change_var', 'add_not', 'or_and_exchange', 'glob_even_exchange','until_order']
    '''
    neg_adj = copy.deepcopy(adj)
    neg_attr = copy.deepcopy(attr)
    #print(neg_adj,neg_attr)
    if type == 'random':
        if nvar != 1:
            type = random.choice(['change_var', 'add_not', 'or_and_exchange', 'glob_even_exchange','until_order'])
        else:
            type = random.choice(['add_not', 'or_and_exchange', 'glob_even_exchange','until_order'])

    if type == 'change_var':
        list_sgn = [i for i, row in enumerate(attr) if row[8] == 1.0]
        var_nodes = torch.nonzero((attr[:, 9:(8 + nvar + 1)] == 1.0).any(dim=1), as_tuple=True)[0].tolist()
        #case of only one variable used
        sgn_sel =  random.choice(list_sgn)
        child_var = conv.find_real_children(sgn_sel,adj,attr)
        if isinstance(child_var,list):
            child_var=child_var[0]
        #print(child_var)
        #print(var_nodes)
        if len(var_nodes)!=1:
            neg_adj[sgn_sel][child_var]=0.0
            var_nodes.remove(child_var)
            if isinstance(var_nodes,list)==False:
                var_nodes=[var_nodes]
            
            #print(var_nodes)
            new_var = int(random.choice(var_nodes))
            #print(new_var)
            neg_adj[sgn_sel][new_var]=1.0
            #print(neg_adj,neg_attr)
            neg_adj,neg_attr = conv.topological_ordering(neg_adj,neg_attr)
            return neg_adj,neg_attr
        else:
            #print('here')
            y = torch.nonzero(attr[var_nodes[0], :] != 0, as_tuple=True)[0].tolist()
            y=int(y[0])-9
            x=y
            while y == x:
                x = random.randint(0, nvar-1)
            x=x+9
            neg_attr[var_nodes[0]][y], neg_attr[var_nodes[0]][x] = 0.0, 1.0
            neg_adj,neg_attr = conv.topological_ordering(neg_adj,neg_attr)
            return neg_adj,neg_attr


    if type == 'add_not':
        list_possible = [i for i in range(1, attr.size(0) - 1) if not torch.any(attr[i, 9:9 + nvar] == 1.0)]
        #print(list_possible)
        sel_point = random.choice(list_possible)
        #print(sel_point)
        neg_adj,neg_attr = add_not(sel_point,adj,attr,neg_adj,neg_attr,nvar)
        neg_adj,neg_attr = conv.topological_ordering(neg_adj,neg_attr)
        return neg_adj,neg_attr 

    
    if type == 'or_and_exchange':
        list_and = [i for i, row in enumerate(attr) if row[2] == 1.0]   #indices of AND rows
        list_or = [i for i, row in enumerate(attr) if row[3] == 1.0]    #indices of OR rows 
        sel_type = random.choice([0,1]) #select if we modify an OR (0) or a AND (1)

        #check whether one of the two lists is empty (or both)
        if len(list_and)==0 and len(list_or)==0:
            return neg_instance_matrices(adj,attr,nvar,type='random')
                
        elif len(list_and)==0 or (len(list_and)!= 0 and len(list_or)!= 0 and sel_type==1): #change OR-->AND
            random_index = random.randint(0, len(list_or) - 1)
            sel_node = list_or[random_index]
            neg_attr[sel_node][3]=0.0
            neg_attr[sel_node][2]=1.0

        elif len(list_or)==0 or (len(list_and)!= 0 and len(list_or)!= 0 and sel_type==0): #change AND-->OR
            random_index = random.randint(0, len(list_and) - 1)
            sel_node = list_and[random_index]
            neg_attr[sel_node][3]=1.0
            neg_attr[sel_node][2]=0.0
        neg_adj,neg_attr = conv.topological_ordering(neg_adj,neg_attr)
        return neg_adj,neg_attr

    if type == 'glob_even_exchange':
        list_global = [i for i, row in enumerate(attr) if row[6] == 1.0]   #indices of Global rows
        list_eventual = [i for i, row in enumerate(attr) if row[5] == 1.0]    #indices of Eventual rows 
        sel_type = random.choice([0,1]) #select if we modify an Global (0) or a Eventual (1)

        #check whether one of the two lists is empty (or both)
        if len(list_global)==0 and len(list_eventual)==0:
            return neg_instance_matrices(adj,attr,nvar,type='random')
                
        elif len(list_global)==0 or (len(list_global)!= 0 and len(list_eventual)!= 0 and sel_type==1): #change Eventual-->Global
            random_index = random.randint(0, len(list_eventual) - 1)
            sel_node = list_eventual[random_index]
            neg_attr[sel_node][5]=0.0
            neg_attr[sel_node][6]=1.0

        elif len(list_eventual)==0 or (len(list_global)!= 0 and len(list_eventual)!= 0 and sel_type==0): #change Global-->Eventual
            random_index = random.randint(0, len(list_global) - 1)
            sel_node = list_global[random_index]
            neg_attr[sel_node][6]=0.0
            neg_attr[sel_node][5]=1.0
        neg_adj,neg_attr = conv.topological_ordering(neg_adj,neg_attr)
        return neg_adj,neg_attr
    
    if type == 'until_order':
        list_until = [i for i in range(attr.size(0)) if attr[i][7] == 1.0]
        if len(list_until)==0: #no until in the formula
            return neg_instance_matrices(adj,attr,nvar,type='random')
        until_sel = random.choice(list_until)
        childs = (adj[until_sel] == 1.0).nonzero(as_tuple=True)[0].tolist()
        child_a,child_b = childs[0],childs[1]
        #case 1: child a is the first child
        if adj[child_a][child_b]== 1.0:
            neg_adj[child_a][child_b]=0.0
            neg_adj[child_b][child_a]=1.0
        #case 2: child b is the first child
        if adj[child_b][child_a]== 1.0:
            neg_adj[child_b][child_a]=0.0
            neg_adj[child_a][child_b]=1.0
        neg_adj,neg_attr = conv.topological_ordering(neg_adj,neg_attr)
        return neg_adj,neg_attr
    

    
def add_not(sel_point,adj,attr,new_adj,new_attr,nvar):
    '''Add a not at a given point in a graph (but reasoning with matrices)'''
    not_row = torch.zeros(12+nvar)
    not_row[4]=1.0
    new_attr = torch.cat((new_attr[:-1], not_row.unsqueeze(0), new_attr[-1:]))
    parent = find_parent(sel_point,adj,attr)
    #new adjacency matrix
    rows_to_insert = torch.zeros((1, new_adj.size(1)))
    new_adj = torch.cat((new_adj[:-1], rows_to_insert, new_adj[-1:]), dim=0)
    cols_to_insert = torch.zeros((new_adj.size(0), 1))
    new_adj = torch.cat((new_adj[:, : -1], cols_to_insert, new_adj[:, -1:]), dim=1)
    new_adj[parent][-2], new_adj[parent][sel_point]= 1.0, 0.0
    new_adj[-2][sel_point] = 1.0
    if new_attr[parent][7]==1.0: #father is until
        childs = (adj[parent] == 1.0).nonzero(as_tuple=True)[0].tolist()
        if len(childs) == 3: #case of until child of another until
            #print('until-until case')
            grandparent = find_parent(parent,adj,attr)
            uncle = ((adj[parent] == 1.0) & (adj[grandparent] == 1.0)).nonzero(as_tuple=True)[0].item()
            childs.remove(uncle)
        child_a,child_b = childs[0],childs[1]
        if adj[child_a][child_b]==1.0:
            child_1=child_a
            child_2=child_b
        elif adj[child_b][child_a]==1.0:
            child_1=child_b
            child_2=child_a
        else:
            print('Error',adj,attr)
            print(child_a,child_b)
        a=2
        if child_1 == sel_point:
            new_adj[child_1][child_2]=0.0
            new_adj[-2][child_2]=1.0
        elif child_2 == sel_point:
            new_adj[child_1][child_2]=0.0
            new_adj[child_1][-2]=1.0
    return new_adj,new_attr


def find_parent(sel_point,adj,attr):
    '''Find the parent of a given node by checking the graph matrices (can be not straightforward in the case of Until)'''
    parent = None
    parents=(adj[:, sel_point] == 1.0).nonzero(as_tuple=True)[0].tolist()
    if len(parents)==1:
        parent=parents[0]
    else:
        for j in parents:
            if attr[j][7]==1.0:
                parent=j
                break
    if parent is None:
        print('ERROR')
    return parent


def contrastive_generation(batches, n_pos, n_neg,nvar, device,space=5):
    '''Given the batches for the training of a NN it generates some positive and negative random instances
    n_pos and n_neg are the number of positive and negative instances for each anchor formula
    space is the number of epochs the generation of instances must cover
    '''
    positive_instances = []
    negative_instances = []

    for k in range(space):
        for batch in batches:
            for graph in batch:
                for i in range(n_pos):
                    p = pos_instance_matrices(graph[0], graph[1], nvar, 'random')
                    positive_instances.append(p)  # Collect positive instances

                for i in range(n_neg):
                    n = neg_instance_matrices(graph[0], graph[1], nvar, 'random')
                    negative_instances.append(n)  # Collect negative instances

    # Convert lists to tensors and move to the specified device
    return positive_instances, negative_instances






#VERSION THAT ACTS DIRECTLY ON FORMULAE


#positive instances
def positive_instance(formula,type,nvar):
    '''Create a positive instance of a formula
    type=[de_morgan, double_neg, insert_tautology,ev_glo_into_until]'''
    #create a copy of the formula
    positive = copy.deepcopy(formula)
    #create lists that contains the type of nodes in the formula
    list_var,list_all,list_or,list_and,list_globally,list_eventually,list_until = [],[],[],[],[],[],[]
    par_var,par_all,par_or,par_and,par_globally,par_eventually,par_until = [],[],[],[],[],[],[]
    #populates the lists by DFS
    build_dictionaries(positive,None,list_var,list_all,list_or,list_and,list_globally,list_eventually,list_until,
                       par_var,par_all,par_or,par_and,par_globally,par_eventually,par_until)
    if type == 'random':
        type = random.choice(['de_morgan', 'double_neg', 'insert_tautology', 'ev_glo_into_until'])
    if type == 'de_morgan':
        sel_type = random.choice([0,1]) #select if we modify an OR (0) or a AND (1)
        #check whether one of the two lists is empty (or both)
        if len(list_and)==0 and len(list_or)==0:
            return try_new_positive(formula,nvar)        
        elif len(list_and)==0 or (len(list_and)!= 0 and len(list_or)!= 0 and sel_type==1):
            random_index = random.randint(0, len(list_or) - 1)
            sel_node = list_or[random_index]
            par_node = par_or[random_index]
            new_node = stl.And(stl.Not(sel_node.left_child),stl.Not(sel_node.right_child))
        elif len(list_or)==0 or (len(list_and)!= 0 and len(list_or)!= 0 and sel_type==0):
            random_index = random.randint(0, len(list_and) - 1)
            sel_node = list_and[random_index]
            par_node = par_and[random_index]
            new_node = stl.Or(stl.Not(sel_node.left_child),stl.Not(sel_node.right_child))
       
        positive = replace(par_node,new_node,sel_node,par_all,list_all)
        return positive
    
    if type == 'double_neg':
        #sample a node and add NOT in one of its child
        sel_target = random.choice(par_all)
        if sel_target==None: #initial node selected
            positive=stl.Not(stl.Not(positive))
        else:
            if isinstance(sel_target, stl.And) or isinstance(sel_target, stl.Or) or isinstance(sel_target, stl.Until):
                #sample one between left and right child
                side = random.randint(0, 1)
                if side == 0:   #left child
                    sel_target.left_child = stl.Not(stl.Not(sel_target.left_child))
                else:           #right child
                    sel_target.right_child = stl.Not(stl.Not(sel_target.right_child))
            else:
                sel_target.child = stl.Not(stl.Not(sel_target.child)) 
        return positive
    
    if type == 'insert_tautology':
        #sample a node and add NOT in one of its child
        sel_target = random.choice(par_all)
        taut_1,taut_2 =  generate_tautology(nvar)
        if sel_target==None: #initial node selected
            positive = stl.And(positive,stl.Or(taut_1,taut_2))
        else:
            if isinstance(sel_target, stl.And) or isinstance(sel_target, stl.Or) or isinstance(sel_target, stl.Until):
                #sample one between left and right child
                side = random.randint(0, 1)
                if side == 0:   #left child
                    sel_target.left_child = stl.And(sel_target.left_child,stl.Or(taut_1,taut_2))
                else:           #right child
                    sel_target.right_child = stl.And(sel_target.right_child,stl.Or(taut_1,taut_2))
            else:
                sel_target.child = stl.Not(sel_target.child) 
        return positive
    
    if type == 'ev_glo_into_until':
        sel_type = random.choice([0,1]) #select if we modify a Globally (0) or an Eventually (1)
        #check whether one of the two lists is empty (or both)
        if len(list_globally)==0 and len(list_eventually)==0:
            return try_new_positive(formula,nvar)
        
        elif len(list_globally)==0 or (len(list_globally)!= 0 and len(list_eventually)!= 0 and sel_type==1):
            #replace eventually[a,b](phi) with True Until[a,b] phi
            random_index = random.randint(0, len(list_eventually) - 1)
            sel_node = list_eventually[random_index]
            par_node = par_eventually[random_index]
            taut_1,taut_2 = generate_tautology(nvar)
            new_node = stl.Until(stl.Or(taut_1,taut_2),sel_node.child,False,False,sel_node.left_time_bound,sel_node.right_time_bound-1)
        
        elif len(list_eventually)==0 or (len(list_globally)!= 0 and len(list_eventually)!= 0 and sel_type==0):
            #replace globally[a,b](phi) with Not(eventually[a,b](Not(phi)))
            random_index = random.randint(0, len(list_globally) - 1)
            sel_node = list_globally[random_index]
            par_node = par_globally[random_index]
            new_node = stl.Not(stl.Eventually(stl.Not(sel_node.child),False,False,sel_node.left_time_bound,sel_node.right_time_bound-1))  

        positive = replace(par_node,new_node,sel_node,par_all,list_all)
        return positive



#negative instances

def negative_instance(formula,type,nvar):
    '''Create a negative instance of a formula
    type=[change_var, add_not, or_and_exchange, glob_even_exchange]'''
    #create a copy of the formula
    negative = copy.deepcopy(formula)
    #create lists that contains the type of nodes in the formula
    list_var,list_all,list_or,list_and,list_globally,list_eventually,list_until = [],[],[],[],[],[],[]
    par_var,par_all,par_or,par_and,par_globally,par_eventually,par_until = [],[],[],[],[],[],[]
    #populates the lists by DFS
    build_dictionaries(negative,None,list_var,list_all,list_or,list_and,list_globally,list_eventually,list_until,
                       par_var,par_all,par_or,par_and,par_globally,par_eventually,par_until)
    if type == 'random':
        type = random.choice(['change_var', 'add_not', 'or_and_exchange', 'glob_even_exchange','unitil_order'])
    if type == 'change_var':
        #sample from the list of variables an entry and change the variable index
        sel_var = random.choice(list_var)
        current_var_ind = sel_var.var_index
        indices = [i for i in range(nvar) if i != current_var_ind]
        sel_var.var_index=random.choice(indices)
        return negative
    
    if type == 'add_not':
        #sample a node and add NOT in one of its child
        sel_target = random.choice(par_all)
        if sel_target==None: #initial node selected
            negative=stl.Not(negative)
        else:
            if isinstance(sel_target, stl.And) or isinstance(sel_target, stl.Or) or isinstance(sel_target, stl.Until):
                #sample one between left and right child
                side = random.randint(0, 1)
                if side == 0:   #left child
                    sel_target.left_child = stl.Not(sel_target.left_child)
                else:           #right child
                    sel_target.right_child = stl.Not(sel_target.right_child)
            else:
                sel_target.child = stl.Not(sel_target.child) 
        return negative
               
    if type == 'or_and_exchange':
        sel_type = random.choice([0,1]) #select if we modify an OR (0) or a AND (1)
        #check whether one of the two lists is empty (or both)
        if len(list_and)==0 and len(list_or)==0:
            return try_new_negative(formula,nvar)
        
        elif len(list_and)==0 or (len(list_and)!= 0 and len(list_or)!= 0 and sel_type==1):
            random_index = random.randint(0, len(list_or) - 1)
            sel_node = list_or[random_index]
            par_node = par_or[random_index]
            new_node = stl.And(sel_node.left_child,sel_node.right_child)
        elif len(list_or)==0 or (len(list_and)!= 0 and len(list_or)!= 0 and sel_type==0):
            random_index = random.randint(0, len(list_and) - 1)
            sel_node = list_and[random_index]
            par_node = par_and[random_index]
            new_node = stl.Or(sel_node.left_child,sel_node.right_child)
       
        negative = replace(par_node,new_node,sel_node,par_all,list_all)
        return negative
    
    if type == 'glob_even_exchange':
        sel_type = random.choice([0,1]) #select if we modify a Globally (0) or an Eventually (1)
        #check whether one of the two lists is empty (or both)
        if len(list_globally)==0 and len(list_eventually)==0:
            return try_new_negative(formula,nvar)
        
        elif len(list_globally)==0 or (len(list_globally)!= 0 and len(list_eventually)!= 0 and sel_type==1):
            random_index = random.randint(0, len(list_eventually) - 1)
            sel_node = list_eventually[random_index]
            par_node = par_eventually[random_index]
            new_node = stl.Globally(sel_node.child,False,False,sel_node.left_time_bound,sel_node.right_time_bound-1)
            
        elif len(list_eventually)==0 or (len(list_globally)!= 0 and len(list_eventually)!= 0 and sel_type==0):
            random_index = random.randint(0, len(list_globally) - 1)
            sel_node = list_globally[random_index]
            par_node = par_globally[random_index]
            new_node = stl.Eventually(sel_node.child,False,False,sel_node.left_time_bound,sel_node.right_time_bound-1)  

        negative = replace(par_node,new_node,sel_node,par_all,list_all)
        return negative
    
    if type == 'until_order':
        if len(list_until)==0: #no until in the formula
            return try_new_negative(formula,nvar)
        selected_until =  random.choice(list_until)
        left=selected_until.left_child
        right=selected_until.right_child
        selected_until.left_child=right
        selected_until.right_child=left    
    return negative



def build_dictionaries(formula,parent_node,list_var,list_all,list_or,list_and,list_globally,list_eventually,list_until,
                       par_var,par_all,par_or,par_and,par_globally,par_eventually,par_until):
    '''Given a formula it returns the lists that contains the points for a possible change'''
    current_node = formula
    list_all+=[current_node]
    par_all+=[parent_node]
    if type(current_node) is not stl.Atom:
        if (type(current_node) is stl.And) or (type(current_node) is stl.Or) or (type(current_node) is stl.Until):
            if type(current_node) is stl.And:
                par_and+=[parent_node]
                list_and+=[current_node]
            if type(current_node) is stl.Or:
                par_or+=[parent_node]
                list_or+=[current_node]
            if type(current_node) is stl.Until:
                par_until+=[parent_node]
                list_until+=[current_node]
            current_child=current_node.left_child
            if type(current_child) is stl.Atom:
                par_var+=[current_node]
                list_var+=[current_child]
                #par_all+=[current_child]           
            else:
                build_dictionaries(current_child,current_node,list_var,list_all,list_or,list_and,list_globally,list_eventually,list_until,
                                   par_var,par_all,par_or,par_and,par_globally,par_eventually,par_until)
            current_child=current_node.right_child
            if type(current_child) is stl.Atom:
                par_var+=[current_node]
                list_var+=[current_child] 
                #par_all+=[current_child]
            else:
                build_dictionaries(current_child,current_node,list_var,list_all,list_or,list_and,list_globally,list_eventually,list_until,
                                   par_var,par_all,par_or,par_and,par_globally,par_eventually,par_until)
        else:
            if type(current_node) is stl.Globally:
                par_globally+=[parent_node]
                list_globally+=[current_node]
            if type(current_node) is stl.Eventually:
                par_eventually+=[parent_node]
                list_eventually+=[current_node]           
            current_child=current_node.child
            if type(current_child) is stl.Atom:
                par_var+=[current_node]
                list_var+=[current_child] 
                #par_all+=[current_child]
            else:
                build_dictionaries(current_child,current_node,list_var,list_all,list_or,list_and,list_globally,list_eventually,list_until,
                                   par_var,par_all,par_or,par_and,par_globally,par_eventually,par_until)

def replace(parent,new_node,old_node,par_all,list_all):
    '''Given a parent node it exchanges a old node with a new one'''
    if parent == None:
        parent = None
        #print('no parent')
    while parent != None:
        old_parent = copy.deepcopy(parent)
        index = list_all.index(parent)
        grandparent = par_all[index]

        #two child case
        if isinstance(parent, stl.And) or isinstance(parent, stl.Or) or isinstance(parent, stl.Until):
            #determine if old node is left or right child
            if parent.left_child == old_node:
                parent.left_child = new_node
            elif parent.right_child == old_node:
                parent.right_child = new_node
        else: #single child case
            parent.child=new_node

        old_node = old_parent
        new_node = parent
        parent = grandparent  
    return new_node

def generate_tautology(nvar):
    var_index = random.randint(0,nvar-1)
    threshold = random.gauss(0,1)
    t1=stl.Atom(var_index,threshold,True)
    t2=stl.Not(stl.Atom(var_index,threshold,True))
    return t1,t2

def try_new_positive(formula,nvar):
    i_type = random.choice(['de_morgan', 'double_neg', 'insert_tautology','ev_glo_into_until'])
    return positive_instance(formula,i_type,nvar)

def try_new_negative(formula,nvar):
    i_type = random.choice(['change_var', 'add_not', 'or_and_exchange', 'glob_even_exchange'])
    return negative_instance(formula,i_type,nvar)            





