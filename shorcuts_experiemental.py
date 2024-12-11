import torch
import pickle
from phis_generator import StlGenerator
from traj_measure import BaseMeasure
from kernel import StlKernel


                ##################################################################################
                # Some simple functions to perform easily some common operations for experiments #
                ##################################################################################


def formulae_generation(n_vars,bag_size, pickle_dumping = None, leaf_prob = 0.38, inner_node_prob = None, threshold_mean= 0.0, threshold_sd = 1.0,
        unbound_prob= 0.1, right_unbound_prob = 0.2, time_bound_max_range = 20, adaptive_unbound_temporal_ops = True,
        max_timespan = 100, max_depth = 5):
    '''Returns a list of formulae with the possibility to store them. Parameters are the same as in the StlGenerator
    pickle_dumping is the name of the file to create'''
    sampler = StlGenerator(leaf_prob, inner_node_prob, threshold_mean, threshold_sd, unbound_prob, right_unbound_prob,
                            time_bound_max_range, adaptive_unbound_temporal_ops, max_timespan, max_depth)
    list_of_formulae = sampler.bag_sample(bag_size, n_vars, one_sign=True)
    
    if pickle_dumping:
        try:
            with open(pickle_dumping, 'wb') as file:
                pickle.dump(list_of_formulae, file)
        except Exception as e:
            print(f"Failed to pickle dump: {e}")

    return list_of_formulae


def generate_stoch_model(varn, num_traj, pickle_dumping=None, mu0=0.0, sigma0=1.0, mu1=0.0, sigma1=1.0, q=0.02, q0=0.5, 
        list_of_traj=False,period=False,norm_traj=False):
    '''Returns a tensor of trajectories with the possibility to store them (single stochastic model)
     pickle_dumping is the name of the file to create'''
    if list_of_traj==False:
        measure = BaseMeasure(mu0, sigma0, mu1, sigma1, q, q0,period=period)
        trajs = measure.sample(samples=num_traj,varn=varn)
    else: 
        #case of manually inserted trajectories
        trajs = list_of_traj
    if norm_traj == True:
        trajs=measure.normalize_trajectories(trajs)
    if pickle_dumping:
        with open(pickle_dumping, 'wb') as file:
            pickle.dump(trajs, file)
    return trajs

def avg_robustness_trajs_for_formulae(formulae, trajectories, varn, mean = True):
    '''
    Returns a matrix of the mean robustness of the formulae (rows), given the trajectories
    if mean is set to False it returns simply the robustness matrix formulae x trajectories
    '''
    measure = BaseMeasure()
    kernel = StlKernel(measure,varn=varn, signals=trajectories, samples=len(trajectories))
    rhos, self_kernel = kernel._compute_robustness_no_time(formulae)
    #average on the trajectories
    if mean == True:
        rhos = rhos.mean(dim=1, keepdim=True)
    return rhos

def robustness_stochastic_models(formulae, models, varn):
    '''Models is a list of stochastich models (i.e. a list of tensors of trajectories)'''
    column_vectors = []
    for i in range(0,len(models)):
        col_vector = avg_robustness_trajs_for_formulae(formulae, models[i], varn, mean = True)
        column_vectors.append(col_vector)
    rob_matrix = torch.cat(column_vectors, dim=1)    
    return rob_matrix

  