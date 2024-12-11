import time
import torch
import pickle
from torch import optim
from torch.optim.lr_scheduler import CyclicLR
from torch.optim.lr_scheduler import StepLR,LambdaLR
from phis_generator import StlGenerator
from data_utils import StlFormulaeDataset,StlFormulaeLoader
from data_utils import get_structure_info_flattened
from stl_dataset import main_representation
from kernel_embedding import kernel_embedding, kernel_pca_embedding,transform_new
from utils import execution_time, save_fn
import LogicVAE as LVAE
import DAGNN_approx as gnn_app
import conversions as conv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import KernelPCA
import parameters_fine_tuning as pft
import sys
import gc
import testing_metrics as testing

sys.setrecursionlimit(10**9)


def train_STL_GVAE(arg, model, train_loader, optimizer, device, batch_size=32, lr=0.001, conditional=False, k_embeddings=False,  save_function=save_fn, checkpoint=None, end_epoch=1000):
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda')) 
    start_epoch = 0
    train_loss_list, validation_loss_list = [], []
    
    #initialize or load from checkpoint
    if checkpoint:
        print(f"Loading checkpoint from {checkpoint}...")
        checkpoint_data = torch.load(checkpoint, map_location=device,weights_only=False)
        model.load_state_dict(checkpoint_data['model_state_dict'])
        #optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        start_epoch = checkpoint_data['epoch']
        train_loss_list =checkpoint_data.get('train_loss_list', [])
        validation_loss_list = checkpoint_data.get('validation_loss_list', [])
        print(f"Resuming training from epoch {start_epoch}...")

    print("\nKernel contrastive pre-training")
    print("\n...Loading training data...")

    #load training data
    train_batches = train_loader.get_data(kind='train')
    data_size=len(train_batches)
    train_batches = [conv.topological_ordering(formula[0], formula[1]) for formula in train_batches]
    train_batches = [(batch[0].clone().detach().to(device), batch[1].clone().detach().to(device)) for batch in train_batches]
    t_batch_degree, t_batch_relatives = get_structure_info_flattened(train_batches, device)
    print('Size of the dataset: ', data_size)

    #load validation data
    val_batches = train_loader.get_data(kind='validation')
    val_batches = [conv.topological_ordering(formula[0], formula[1]) for formula in val_batches]
    val_batches = [(batch[0].clone().detach().to(device), batch[1].clone().detach().to(device)) for batch in val_batches]
    v_batch_degree, v_batch_relatives = get_structure_info_flattened(val_batches, device)

    #topology data
    train_topology = [conv.topological_info(formula) for formula in train_batches]
    train_topology = [torch.tensor(batch, device=device) for batch in train_topology]
    val_topology = [conv.topological_info(formula) for formula in val_batches]
    val_topology = [torch.tensor(batch, device=device) for batch in val_topology]


    #trees for teacher forcing
    cut_tree = [[tensor.to(device) for tensor in conv.dfs_ordered(conv.tree_pruning(g, remove_until=True, remove_final=True, remove_variables=True))]
    for g in train_batches]

    cut_val_tree = [[tensor.to(device) for tensor in conv.dfs_ordered(conv.tree_pruning(g, remove_until=True, remove_final=True, remove_variables=True))]
        for g in val_batches]

    print("...loaded")
    model.name_model()
    print('---------------------')

    #conditional mode
    if conditional:
        #add kernel embedding as the third component in describing graphs
        cut_tree = [tuple(list(node) + [embed]) for node, embed in zip(cut_tree, k_embeddings[:len(cut_tree)])]
        train_batches = [tuple(list(batch) + [embed]) for batch, embed in zip(train_batches, k_embeddings[:len(train_batches)])]
        cut_val_tree = [tuple(list(node) + [embed]) for node, embed in zip(cut_val_tree, k_embeddings[len(cut_tree):])]
        val_batches = [tuple(list(batch) + [embed]) for batch, embed in zip(val_batches, k_embeddings[len(train_batches):])]

        
    # scheduler
    n_batches = len(train_batches)//batch_size
    n_steps = 2000-n_batches//2
    cyclic_scheduler = CyclicLR(optimizer=optimizer, base_lr=lr, max_lr=3 * lr, mode='triangular', cycle_momentum=False, step_size_up=int(n_steps))
    decay_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.4 ** (epoch // 80))
    train_start = time.time()
    
    
    #TRAINING LOOP
    for epoch in range(start_epoch, end_epoch):
        epoch_start = time.time()
        model.train()
        #train losses initialisation
        epoch_loss, epoch_rec_loss, epoch_kld_loss,epoch_interval_loss, epoch_threshold_loss,epoch_variable_loss = [0 for _ in range(6)]
        epoch_acc=0

        #shuffle indices for batch processing
        indices = np.random.permutation(data_size)
        n_data = 0

        for i in range(n_batches):
            batch_indices = indices[i * batch_size:(i + 1) * batch_size]
            
            #check to ensure batch_indices are within the range
            if max(batch_indices) >= len(train_batches):
                print(f"Error: Batch indices exceed dataset size. Batch indices: {batch_indices}")
                continue
            
            #extract batch information
            g_batch_complete = [train_batches[j] for j in batch_indices]
            g_degree = [t_batch_degree[j] for j in batch_indices]
            g_relatives = [t_batch_relatives[j] for j in batch_indices]
            g_topology = [train_topology[j] for j in batch_indices] 
            g_cut = [cut_tree[j] for j in batch_indices]
          
            #training
            model.zero_grad(set_to_none=True)
            mu, sigma = model.encode(g_batch_complete,g_degree,g_relatives, g_topology)
            batch_loss, batch_rec_loss, batch_kld_loss, interval_loss, threshold_loss,batch_variable_loss, accuracy = model.loss(mu, sigma, g_cut,g_batch_complete,accuracy=True)
            
            #backward pass
            if scaler:
                scaler.scale(batch_loss).backward()
                scaler.unscale_(optimizer)  #unscales gradients for clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2.0)
                optimizer.step() 
            
            optimizer.zero_grad()

            #loss update
            epoch_loss += batch_loss
            epoch_rec_loss += batch_rec_loss
            epoch_kld_loss += batch_kld_loss
            epoch_interval_loss += interval_loss
            epoch_threshold_loss += threshold_loss
            epoch_variable_loss += batch_variable_loss
            epoch_acc+=accuracy

            batch_loss = batch_loss.detach()
            del batch_loss, batch_rec_loss, batch_kld_loss, interval_loss, threshold_loss,batch_variable_loss, accuracy

            n_data += len(g_batch_complete)
            decay_scheduler.step()
            cyclic_scheduler.step()

            #clear memory
            del g_batch_complete, g_degree, g_relatives, g_topology, g_cut
            

        epoch_end = time.time()
        epoch_h, epoch_m, epoch_s = execution_time(epoch_start, epoch_end)
        div = n_data
        print("Epoch: ", epoch, "Training Total/Reconstruction/KLD/Interval/Threshold/Variable Loss: {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.4f}".format(
            epoch_loss.item()/div, epoch_rec_loss.item()/div, epoch_kld_loss.item()/div, epoch_interval_loss.item()/div, epoch_threshold_loss.item()/div, epoch_variable_loss/div))
        print("Accuracy: {:.3f}".format(epoch_acc/n_batches) )
        print("Time for epoch: {:d} h {:d} m {:d} s ".format(epoch_h, epoch_m, epoch_s))
        print('----------------------------------------')

        #normalize epoch losses and log
        div = data_size
        epoch_loss_avg = epoch_loss.item() / div
        epoch_rec_loss_avg = epoch_rec_loss.item() / div
        epoch_kld_loss_avg = epoch_kld_loss.item() / div
        epoch_interval_loss_avg = epoch_interval_loss.item() / div
        epoch_threshold_loss_avg = epoch_threshold_loss.item() / div
        epoch_variable_loss_avg = epoch_variable_loss / div
        epoch_accuracy_avg = epoch_acc / n_batches

        #train_loss_list.append({'epoch': epoch,'total_loss': float(epoch_loss_avg),'rec_loss': float(epoch_rec_loss_avg), 'kld_loss': float(epoch_kld_loss_avg), 'interval_loss': float(epoch_interval_loss_avg),'threshold_loss': float(epoch_threshold_loss_avg), 'variable_loss': float(epoch_variable_loss_avg), 'accuracy': float(epoch_accuracy_avg)})
        #gc.collect() 
        del epoch_loss_avg, epoch_rec_loss_avg, epoch_kld_loss_avg, epoch_interval_loss_avg, epoch_threshold_loss_avg, epoch_variable_loss_avg, epoch_accuracy_avg
        del epoch_loss, epoch_kld_loss, epoch_interval_loss, epoch_threshold_loss, epoch_variable_loss, epoch_acc

        
        
        
        #tests
        if epoch %5==1:
            model.eval()
            with torch.no_grad():
                #validation losses
                mu, sigma = model.encode(val_batches,v_batch_degree,v_batch_relatives, val_topology)
                val_loss, val_rec_loss, val_kld_loss, val_interval_loss, val_threshold_loss,val_variable_loss,val_accuracy = model.loss(mu, sigma, cut_val_tree,val_batches,accuracy=True)

                val_loss_avg = val_loss.item() / len(val_batches)
                val_rec_loss_avg = val_rec_loss.item() / len(val_batches)
                val_kld_loss_avg = val_kld_loss.item() / len(val_batches)
                val_interval_loss_avg = val_interval_loss.item() / len(val_batches)
                val_threshold_loss_avg = val_threshold_loss.item() / len(val_batches)
                val_variable_loss_avg = val_variable_loss / len(val_batches)
                val_accuracy_avg = val_accuracy

                print("Validation Total/Reconstruction/KLD/Interval/Threshold/Variable Loss: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(
                val_loss_avg, val_rec_loss_avg, val_kld_loss_avg, val_interval_loss_avg,val_threshold_loss_avg, val_variable_loss_avg))
                print("Accuracy: {:.3f}".format(val_accuracy_avg))
                print('----------------------------------------')

                #validation_loss_list.append({'epoch': epoch, 'total_loss': float(val_loss_avg), 'rec_loss': float(val_rec_loss_avg), 'kld_loss': float(val_kld_loss_avg), 'interval_loss': float(val_interval_loss_avg),'threshold_loss': float(val_threshold_loss_avg), 'variable_loss': float(val_variable_loss_avg), 'accuracy': float(val_accuracy_avg)})
                del val_loss, val_rec_loss, val_kld_loss, val_interval_loss, val_threshold_loss,val_variable_loss,val_accuracy
                del val_loss_avg,val_rec_loss_avg, val_kld_loss_avg, val_interval_loss_avg, val_threshold_loss_avg, val_variable_loss_avg
             
            
        if epoch %40==10:
            model.eval()
            with torch.no_grad():
                if model.conditional:
                    random_indices_test = torch.randperm(len(val_batches)) #select a subset of indices for testing (whole test set if more computational power) 
                    
                    ks_t_np = testing.ker_sim(model,train_batches,t_batch_degree, t_batch_relatives, train_topology,random_indices_test, prior=False)
                    ks_v_np = testing.ker_sim(model,val_batches,v_batch_degree, v_batch_relatives, val_topology,random_indices_test, prior=False)
                    
                    ks_t_p = testing.ker_sim(model,train_batches,t_batch_degree, t_batch_relatives, train_topology, random_indices_test, prior=True)
                    ks_v_p = testing.ker_sim(model,val_batches,v_batch_degree, v_batch_relatives, val_topology,random_indices_test, prior=True)
                    print("Kernel similarity")
                    print("Train mu/sigma: {:.4f}, Test mu/sigma: {:.4f}, Train prior: {:.4f}, Test prior: {:.4f}".format(
                            ks_t_np, ks_v_np, ks_t_p, ks_v_p))
                    del ks_t_np, ks_v_np, ks_t_p, ks_v_p 
                    
                else:
                    random_indices_test = torch.randperm(len(val_batches))
                    ks_t_np = testing.ker_sim(model,train_batches,t_batch_degree, t_batch_relatives, train_topology,random_indices_test, prior=False)
                    ks_v_np = testing.ker_sim(model,val_batches,v_batch_degree, v_batch_relatives, val_topology,random_indices_test, prior=False)
                    print("Kernel similarity")
                    print("Train mu/sigma: {:.4f}, Test mu/sigma: {:.4f}".format(
                            ks_t_np, ks_v_np))                    
                    del ks_t_np, ks_v_np

            if model.conditional:
                #refinement based on kernel similarity
                testing.parameters_refinement(model, val_batches,v_batch_degree,v_batch_relatives, val_topology,random_indices_test, rnd_sample=False) 
                testing.parameters_refinement(model, val_batches,v_batch_degree,v_batch_relatives, val_topology,random_indices_test, rnd_sample=True)

            print('.....................................')
                
            gc.collect() 

        # Checkpoint saving
        if epoch % 5 == 0:
            checkpoint_path = f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                #'optimizer_state_dict': optimizer.state_dict(),
                'train_loss_list': train_loss_list,
                'validation_loss_list': validation_loss_list
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    #end of training
    train_end = time.time()
    train_h, train_m, train_s = execution_time(train_start, train_end)
    print("Training Time: [%d h, %d m, %d s]" % (train_h, train_m, train_s))



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True  
torch.backends.cuda.matmul.allow_tf32 = True 
print(f"Device: {device}")


if __name__ == "__main__":
    #load data (same code for generating it if the file is not already generated)
    dtst = StlFormulaeLoader(1500, 3, device, '1500_complex_aug', max_depth=5, leaf_prob=0.45, unbound_prob=0, right_unbound_prob=0)
    dataset = dtst.get_data('train', True, True, 0)

    #load kernel embeddings with PCA already performed (if not present see the functions in kernel_embedding.py )
    ker_transf = pickle.load(open('kern_emb_aug_1500_transf.pkl', 'rb')).to(device)

    #initialize model
    gat_enc = LVAE.DAG_GNN(3, 15, bidirectional=True, layers=3, conditional=True, semantic_encoding=True)
    gat_enc.to(device)
    optimizer = optim.Adam(gat_enc.parameters(), lr=0.001)

    train_STL_GVAE(
        [], gat_enc, dtst, optimizer, device,
        batch_size=128, lr=0.001, conditional=True, k_embeddings=ker_transf,
        end_epoch=600, checkpoint='checkpoint_epoch_445.pt'
    ) #name of the checkpoint