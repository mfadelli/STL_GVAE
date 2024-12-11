import torch
from traj_measure import BaseMeasure
import conversions as conv
import kernel as k



def kernel_embedding(kernel_formulae, train_formulae=False, var_n=3,samples=10000, anchor='both', PCA = True, PCA_size=100, X_mat=False, eigenvectors=False, k_original=False):
    '''
    Function that compute kernel embeddings of a set of formulae.
    With anchor=True it computes the PCA transformation given by the anchor-anchor embedding
    With anchor=False it computes the train-anchor embedding transformed by the anchor-anchor trasnformation
    With anchor='both' computes the two previous steps sequentially
    '''
    measure = BaseMeasure()
    ker = k.StlKernel(measure, varn=var_n, samples=samples)
    if anchor==True: #compute the kernel among the anchor set psi_i
        anchor_embeddings = ker.compute_bag_bag(kernel_formulae,kernel_formulae)
        if PCA==True: #perform PCA and save the transformation
            anchor_emb_transf,eig,X,K_centered=kernel_pca_embedding(anchor_embeddings,100)
            return anchor_emb_transf,eig,X,K_centered
        else:
            return anchor_embeddings
    if anchor==False: 
        embeddings = ker.compute_bag_bag(train_formulae,kernel_formulae) #compute the kernel between the train set and the anchor set
        if PCA==True: #perform PCA with the anchor-anchor transformation
            embeddings_transf = transform_new(embeddings,X_mat,eigenvectors,k_original)
            return embeddings_transf
        else:
            return embeddings
    if anchor=='both':
        anchor_emb_transf,eig,X,K_centered=kernel_embedding(kernel_formulae, train_formulae=False, var_n=3,samples=10000, anchor=True, PCA = True, PCA_size=100, X_mat=False, eigenvectors=False, k_original=False)
        embeddings= kernel_embedding(kernel_formulae, train_formulae=train_formulae, var_n=3,samples=10000, anchor=False, PCA = True, PCA_size=100, X_mat=X, eigenvectors=eig, k_original=K_centered)
        return embeddings
    


def kernel_pca_embedding(X, k, kernel='linear', gamma=None):
    '''
    Perform PCA and store the transformation obtained to replicate it
    '''
    n_samples, n_features = X.shape

    if gamma is None:
        gamma = 1.0 / n_features

    #Compute the kernel matrix
    if kernel == 'rbf':
        pairwise_sq_dists = torch.cdist(X, X, p=2).pow(2)
        K = torch.exp(-gamma * pairwise_sq_dists)
    elif kernel == 'linear':
        K = X @ X.T
    elif kernel == 'poly':
        K = (gamma * (X @ X.T) + 1).pow(3)
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")

    #center the kernel matrix
    one_n = torch.ones((n_samples, n_samples), device=X.device) / n_samples
    K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n

    #compute the eigenvalues and eigenvectors
    eigvals, eigvecs = torch.linalg.eigh(K_centered)
    #select the top-k eigenvalues and corresponding eigenvectors
    top_indices = torch.argsort(eigvals, descending=True)[:k]
    eigvals_top = eigvals[top_indices]
    eigvecs_top = eigvecs[:, top_indices]
    #normalize eigenvectors to have unit length
    eigvecs_top_normalized = eigvecs_top / torch.sqrt(eigvals_top + 1e-10)

    #compute the embedding
    embedding = K_centered @ eigvecs_top_normalized

    return embedding, eigvecs_top_normalized, X, K_centered


def transform_new(X_new, X_fit, eigvecs_top_normalized, K_fit_centered, kernel='linear', gamma=None):
    """
    Transforms new data points into the kernel PCA space using the fitted components.
    :param X_new: New data tensor of shape (m_samples, n_features)
    :param X_fit: Original data used to fit the Kernel PCA
    :param eigvecs_top_normalized: Eigenvectors obtained during Kernel PCA fitting
    :param K_fit_centered: Centered kernel matrix for the original data
    :param kernel: Type of kernel used during Kernel PCA fitting
    :param gamma: Kernel coefficient for 'rbf', 'poly', and 'sigmoid'
    :return: k-dimensional embedding of the new data points
    """
    #ensure input dimensions match
    n_samples_fit, n_features = X_fit.shape
    m_samples, new_features = X_new.shape

    if new_features != n_features:
        raise ValueError(f"Feature mismatch: X_new has {new_features} features, but X_fit has {n_features}.")

    if gamma is None:
        gamma = 1.0 / n_features

    #compute the kernel matrix between the new data and the original data
    if kernel == 'rbf':
        pairwise_sq_dists = torch.cdist(X_new, X_fit, p=2).pow(2)
        K_new = torch.exp(-gamma * pairwise_sq_dists)
    elif kernel == 'linear':
        K_new = X_new @ X_fit.T
    elif kernel == 'poly':
        K_new = (gamma * (X_new @ X_fit.T) + 1).pow(3)
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")

    #center the new kernel matrix
    one_n_fit = torch.ones((n_samples_fit, n_samples_fit), device=X_fit.device) / n_samples_fit
    one_n_new = torch.ones((m_samples, n_samples_fit), device=X_fit.device) / n_samples_fit
    K_new_centered = K_new - one_n_new @ K_fit_centered - K_new @ one_n_fit + one_n_new @ K_fit_centered @ one_n_fit

    #project the new kernel matrix onto the top eigenvectors
    embedding_new = K_new_centered @ eigvecs_top_normalized

    return embedding_new


