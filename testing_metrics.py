import torch
import conversions as conv
import parameters_fine_tuning as pft

def ker_sim(model, data, degree, relatives, topology, permutation, size=25, prior=False):
    '''
    Compute the kernel similarity between the original formulae and one randomly generated reconstruction per formulae'''
    #compute the subset of testing set for testing 
    cut_perm=permutation[:size]
    val_batches_sampled = [data[i] for i in cut_perm]
    v_batch_degree_sampled = [degree[i] for i in cut_perm]
    v_batch_relatives_sampled = [relatives[i] for i in cut_perm]
    val_topology_sampled = [topology[i] for i in cut_perm]

    val_rec=model.encode_decode(val_batches_sampled,v_batch_degree_sampled,v_batch_relatives_sampled, val_topology_sampled, prior=prior)
    kernel_distance_k=[]
    for j in range(0,size):
        ker_true_rec=model.kernel.compute_one_one(conv.matrices_to_formula(val_batches_sampled[j][0],val_batches_sampled[j][1],3),conv.matrices_to_formula(val_rec[j][0],val_rec[j][1],3))
        kernel_distance_k.append(ker_true_rec.item())
        del ker_true_rec    
                
    avg_kernel=sum(kernel_distance_k)/len(kernel_distance_k)
    #TODO with more computation power print also the highest kernel similarity for n generated reconstruction
    return avg_kernel



def parameters_refinement(model, g_in, g_degree, g_relatives, g_topology, permutation, size=10, n_gen=15, full_kernel=False,rnd_sample=False):
        '''
        generates n_gen formulae (from mu,sigma if rnd_sample=False, from N(0,1) otherwise)
        and perform optimization on their threshold and interval parameters directly on the formulae (no weights learned)
        '''
        #compute the subset of testing set for testing 
        cut_perm=permutation[:size]
        val_batches_sampled = [g_in[i] for i in cut_perm]
        v_batch_degree_sampled = [g_degree[i] for i in cut_perm]
        v_batch_relatives_sampled = [g_relatives[i] for i in cut_perm]
        val_topology_sampled = [g_topology[i] for i in cut_perm]


        avg_distances = []  #to store the average distances
        closest_distances = []  #to store the closest distances
        mu, sigma = model.encode(val_batches_sampled, v_batch_degree_sampled, v_batch_relatives_sampled, val_topology_sampled)
        y = torch.cat([g[2].reshape(1, -1) for g in val_batches_sampled], 0)

        for i in range(len(val_batches_sampled)):
            if rnd_sample==False:
                z = torch.stack([model.reparameterize(mu[i], sigma[i]) for _ in range(n_gen)], dim=0)
            else:
                z = torch.randn(n_gen, mu.shape[1], device=model.device)
            y_single = y[i].unsqueeze(0).repeat(n_gen, 1)  #shape (n_gen, y_dim)
            reconstructed = model.decode(z, y_single)
            rec_formulae=[conv.matrices_to_formula(g[0],g[1],3) for g in reconstructed]
            original_formula=conv.matrices_to_formula(val_batches_sampled[i][0],val_batches_sampled[i][1],3)
            

            #perform the actual optimization
            formulae_better=pft.optimize_formula(rec_formulae,original_formula)
            
            
            #STILL SUFFERS FROM MEMORY LEAKS PROBLEMS
            del rec_formulae
            if hasattr(model.kernel, "cache"):
                model.kernel.cache.clear()  
            #force garbage collection
            import gc
            gc.collect()
            kernel_vv=model.kernel.compute_one_bag(original_formula,formulae_better)
            del formulae_better, original_formula
            gc.collect()
            avg_distance = kernel_vv.mean().item()
            avg_distances.append(avg_distance)
            closest_distance = kernel_vv.max().item()
            
            closest_distances.append(closest_distance)
            del kernel_vv
            gc.collect()
        print('Perfected formulae average kernel: ') if rnd_sample==False else print('Perfected formulae average kernel from prior: ') 
        print(sum(closest_distances)/len(closest_distances))
        return avg_distances, closest_distances