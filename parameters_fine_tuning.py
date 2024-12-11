import torch
import stl
import torch.optim as optim
import kernel as kernel
from traj_measure import BaseMeasure
import gc
import pickle
from kernel_embedding import kernel_embedding,transform_new
from stl_dataset import main_representation
from data_utils import get_structure_info_flattened
from torch.autograd import Variable


#WORK IN PROGRESS
#perform refinement of the formulae (intervals and thresholds)
#huge memory leaks




#MODIFICATION FO THE ORIGINAL NODE CLASSES IN ORDER TO HAVE TRAINABLE PARAMETERS

class TrainableAtom(stl.Atom):
    def __init__(self, var_index, threshold, lte):
        super().__init__(var_index, threshold, lte)
        self.threshold = torch.tensor(threshold, requires_grad=True)

class TrainableGlobally(stl.Globally):
    def __init__(self, child, unbound, right_unbound, left_time_bound, right_time_bound, adaptive):
        super().__init__(child, unbound, right_unbound, left_time_bound, right_time_bound, adaptive)
        self.left_time_bound = torch.tensor(float(left_time_bound), requires_grad=True)
        self.right_time_bound = torch.tensor(float(right_time_bound), requires_grad=True)

class TrainableEventually(stl.Eventually):
    def __init__(self, child, unbound, right_unbound, left_time_bound, right_time_bound, adaptive):
        super().__init__(child, unbound, right_unbound, left_time_bound, right_time_bound, adaptive)
        self.left_time_bound = torch.tensor(float(left_time_bound), requires_grad=True)
        self.right_time_bound = torch.tensor(float(right_time_bound), requires_grad=True)

class TrainableUntil(stl.Until):
    def __init__(self, left_child, right_child, unbound,  left_time_bound, right_time_bound):
        super().__init__(left_child, right_child, unbound, left_time_bound, right_time_bound)
        self.left_time_bound = torch.tensor(float(left_time_bound), requires_grad=True)
        self.right_time_bound = torch.tensor(float(right_time_bound), requires_grad=True)



class TrainableAnd(stl.And):
    """
    Wrapper for And without trainable parameters.
    Retains the same behavior as stl.And.
    """
    def __init__(self, left_child, right_child):
        super().__init__(left_child, right_child)

class TrainableOr(stl.Or):
    """
    Wrapper for Or without trainable parameters.
    Retains the same behavior as stl.Or.
    """
    def __init__(self, left_child, right_child):
        super().__init__(left_child, right_child)

class TrainableNot(stl.Not):
    """
    Wrapper for Not without trainable parameters.
    Retains the same behavior as stl.Not.
    """
    def __init__(self, child):
        super().__init__(child)


def convert_to_trainable(formula):
    """
    Converts the nodes of a formula into trainable versions.

    Parameters:
    ----------
    formula : Node
        An STL formula.

    Returns:
    -------
    trainable_formula : Node
        An STL formula where nodes have been replaced with trainable counterparts.
    """

    if isinstance(formula, stl.Atom):
        # Convert Atom to TrainableAtom
        return TrainableAtom(formula.var_index, formula.threshold, formula.lte)
    
    elif isinstance(formula, stl.Globally):
        # Convert Globally to TrainableGlobally
        child = convert_to_trainable(formula.child)
        return TrainableGlobally(child, formula.unbound, formula.right_unbound, 
                                 formula.left_time_bound, formula.right_time_bound, 
                                 formula.adapt_unbound)
    
    elif isinstance(formula, stl.Eventually):
        # Convert Eventually to TrainableEventually
        child = convert_to_trainable(formula.child)
        return TrainableEventually(child, formula.unbound, formula.right_unbound, 
                                   formula.left_time_bound, formula.right_time_bound, 
                                   formula.adapt_unbound)
    
    elif isinstance(formula, stl.Until):
        # Convert Until to TrainableUntil
        left_child = convert_to_trainable(formula.left_child)
        right_child = convert_to_trainable(formula.right_child)
        return TrainableUntil(left_child, right_child, formula.unbound, 
                              formula.left_time_bound, formula.right_time_bound)
    
    elif isinstance(formula, stl.And):
        # Convert And to TrainableAnd
        left_child = convert_to_trainable(formula.left_child)
        right_child = convert_to_trainable(formula.right_child)
        return TrainableAnd(left_child, right_child)
    
    elif isinstance(formula, stl.Or):
        # Convert Or to TrainableOr
        left_child = convert_to_trainable(formula.left_child)
        right_child = convert_to_trainable(formula.right_child)
        return TrainableOr(left_child, right_child)
    
    elif isinstance(formula, stl.Not):
        # Convert Not to TrainableNot
        child = convert_to_trainable(formula.child)
        return TrainableNot(child)

    else:
        raise ValueError(f"Unknown node type: {type(formula)}")


def optimize_formula(formulae, original_formula, lr=0.1, epochs=12):
    """
    Optimize thresholds and interval bounds in a set of STL formulae to reduce kernel-based loss.

    Parameters:
    ----------
    formulae : list of Node
        List of STL formulae with trainable parameters (thresholds, interval bounds).
    original : Node
        The target STL formula against which the kernel is computed.
    kernel : Kernel
        An object with a method `compute_one_bag` to compute the kernel similarity.
    lr : float
        Learning rate for optimization.
    epochs : int
        Number of training epochs.

    Returns:
    -------
    formulae : list of Node
        The optimized STL formulae.
    loss_history : list
        The history of the loss during training.
    """
    measure=BaseMeasure()
    kernel_d = kernel.StlKernel(measure,samples=200)

    #collect trainable parameters from the formulae
    trainable_params = []

    def collect_trainable_parameters(node):
        if isinstance(node, TrainableAtom):
            trainable_params.append(node.threshold)
        elif isinstance(node, TrainableGlobally) or isinstance(node, TrainableEventually) or isinstance(node, TrainableUntil):
            trainable_params.extend([node.left_time_bound, node.right_time_bound])
        if hasattr(node, 'left_child'):
            collect_trainable_parameters(node.left_child)
        if hasattr(node, 'right_child'):
            collect_trainable_parameters(node.right_child)
        if hasattr(node, 'child'):
            collect_trainable_parameters(node.child)

    for i, formula in enumerate(formulae):
        formulae[i] = convert_to_trainable(formula)  
        collect_trainable_parameters(formulae[i])  #now working on the updated formula
    
    optimizer = optim.Adam(trainable_params, lr=lr)
    for param in trainable_params:
        param.retain_grad()
        

    #def loss_function(formulae,original_formula): #in this implementation proportional to the kernel between original and reconstructed formulae
    #    kernel_v = kernel.compute_one_bag(original_formula, formulae)
    #    k_v = kernel_v.max()  
    #    loss = 1 - k_v 
    #    return loss

    #training loop
    loss_history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        try:
            kernel_v = kernel_d.compute_one_bag(original_formula, formulae)
        except Exception as e:
            print(f"An error occurred: {e}")
            print(f"Formulae: {formulae}")
        k_v = kernel_v.max()
        loss = 1- k_v
        loss.backward()
        loss_history.append(loss.item())

        optimizer.step()
        
        #if epoch==epochs-1:
        #    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

        #memory deletion
        if hasattr(kernel, "cache"):
                kernel.cache.clear()
        if epoch%10==1:   
            gc.collect()  
        del loss
    formulae = detach_formulae(formulae)
    return formulae


def detach_trainable_formula(formula):
    """
    Detaches all trainable parameters in a formula to remove gradient information.

    Parameters:
    ----------
    formula : Node
        A trainable STL formula.

    Returns:
    -------
    formula : Node
        The same formula with all parameters detached.
    """

    if isinstance(formula, TrainableAtom):
        formula.threshold = formula.threshold.detach()
    
    elif isinstance(formula, TrainableGlobally):
        formula.left_time_bound = formula.left_time_bound.detach()
        formula.right_time_bound = formula.right_time_bound.detach()
        detach_trainable_formula(formula.child)
    
    elif isinstance(formula, TrainableEventually):
        formula.left_time_bound = formula.left_time_bound.detach()
        formula.right_time_bound = formula.right_time_bound.detach()
        detach_trainable_formula(formula.child)
    
    elif isinstance(formula, TrainableUntil):
        formula.left_time_bound = formula.left_time_bound.detach()
        formula.right_time_bound = formula.right_time_bound.detach()
        detach_trainable_formula(formula.left_child)
        detach_trainable_formula(formula.right_child)
    
    elif isinstance(formula, TrainableAnd) or isinstance(formula, TrainableOr):
        detach_trainable_formula(formula.left_child)
        detach_trainable_formula(formula.right_child)
    
    elif isinstance(formula, TrainableNot):
        detach_trainable_formula(formula.child)

    # For non-trainable formulae, do nothing.
    return formula

def detach_formulae(formulae):
    """
    Detach all trainable parameters in a list of formulas.

    Parameters:
    ----------
    formulae : list of Node
        List of trainable STL formulae.

    Returns:
    -------
    formulae : list of Node
        List of formulae with detached parameters.
    """
    return [detach_trainable_formula(formula) for formula in formulae]

