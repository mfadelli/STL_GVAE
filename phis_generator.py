import stl
import numpy.random as rnd
from typing import Union
from stl import Node


# TODO: time densities vs thresholds
# TODO: uniform generator should be a special case of this (add option in var threshold setting)
# TODO: also adding params to init! (at the moment uniform has threshold bounds as specific parameters)
# TODO: maybe we can generalize, since both require two parameters


class StlGenerator:
    def __init__(
        self,
        leaf_prob: float = 0.3,
        inner_node_prob: list = None,
        threshold_mean: float = 0.0,
        threshold_sd: float = 1.0,
        unbound_prob: float = 0.1,
        right_unbound_prob: float = 0.2,
        time_bound_max_range: float = 20,
        adaptive_unbound_temporal_ops: bool = True,
        max_timespan: int = 100,
        max_depth = 6
    ):
        """
        leaf_prob
            probability of generating a leaf (always zero for root)
        node_types = ["not", "and", "or", "always", "eventually", "until"]
            Inner node types
        inner_node_prob
            probability vector for the different types of internal nodes
        threshold_mean
        threshold_sd
            mean and std for the normal distribution of the thresholds of atoms
        unbound_prob
            probability of a temporal operator to have a time bound of the type [0,infty]
        time_bound_max_range
            maximum value of time span of a temporal operator (i.e. max value of t in [0,t])
        adaptive_unbound_temporal_ops
            if true, unbounded temporal operators are computed from current point to the end of the signal, otherwise
            they are evaluated only at time zero.
        max_timespan
            maximum time depth of a formula.
        """

        # Address the mutability of default arguments
        if inner_node_prob is None:
            inner_node_prob = [0.166, 0.166, 0.166, 0.17, 0.166, 0.166]

        self.leaf_prob = leaf_prob
        self.inner_node_prob = inner_node_prob
        self.threshold_mean = threshold_mean
        self.threshold_sd = threshold_sd
        self.unbound_prob = unbound_prob
        self.right_unbound_prob = right_unbound_prob
        self.time_bound_max_range = time_bound_max_range
        self.adaptive_unbound_temporal_ops = adaptive_unbound_temporal_ops
        self.node_types = ["not", "and", "or", "always", "eventually", "until"]
        self.max_timespan = max_timespan
        self.max_depth = max_depth

    def sample(self, nvars, one_sign = True):
        """
        Samples a random formula with distribution defined in class instance parameters

        Parameters
        ----------
        nvars : number of variables of input signals
            how many variables the formula is expected to consider.

        one_sign : if True the only type of atom possible is <=
            (>= are converted into NOT <=)

        Returns
        -------
        TYPE
            A random formula.

        """
        formula = self._sample_internal_node(nvars,0)
        if one_sign == True:
            self.single_sign_conversion(formula)
        return formula

    def bag_sample(self, bag_size, nvars, one_sign=True):
        """
        Samples a bag of bag_size formulae

        Parameters
        ----------
        bag_size : INT
            number of formulae.
        nvars : INT
            number of vars in formulae.
        one_sign : if True the only type of atom possible is <=
            (>= are converted into NOT <=)

        Returns
        -------
        a list of formulae.

        """
        formulae = []
        for _ in range(bag_size):
            phi = self.sample(nvars,one_sign)
            formulae.append(phi)
        return formulae

    def _sample_internal_node(self, nvars,depth):
        # Declare & dummy-assign "idiom"
        node: Union[None, Node]
        node = None
        # choose node type
        nodetype = rnd.choice(self.node_types, p=self.inner_node_prob)
        while True:
            if nodetype == "not":
                n = self._sample_node(nvars,depth)
                node = stl.Not(n) #n is the children
            elif nodetype == "and":
                n1 = self._sample_node(nvars,depth)
                n2 = self._sample_node(nvars,depth)
                node = stl.And(n1, n2)
            elif nodetype == "or":
                n1 = self._sample_node(nvars,depth)
                n2 = self._sample_node(nvars,depth)
                node = stl.Or(n1, n2)
            elif nodetype == "always":
                n = self._sample_node(nvars,depth)
                unbound, right_unbound, left_time_bound, right_time_bound = self._get_temporal_parameters()
                node = stl.Globally(
                    n, unbound, right_unbound, left_time_bound, right_time_bound, self.adaptive_unbound_temporal_ops
                )
            elif nodetype == "eventually":
                n = self._sample_node(nvars,depth)
                unbound, right_unbound, left_time_bound, right_time_bound = self._get_temporal_parameters()
                node = stl.Eventually(
                    n, unbound, right_unbound, left_time_bound, right_time_bound, self.adaptive_unbound_temporal_ops
                )
            elif nodetype == "until":
                n1 = self._sample_node(nvars,depth)
                n2 = self._sample_node(nvars,depth)
                unbound, right_unbound, left_time_bound, right_time_bound = self._get_temporal_parameters()
                node = stl.Until(
                    n1, n2, unbound, right_unbound, left_time_bound, right_time_bound
                )

            if (node is not None) and (node.time_depth() < self.max_timespan):
                return node

    def _sample_node(self, nvars,depth):
        if rnd.rand() < self.leaf_prob or depth>=self.max_depth-1:
            # sample a leaf
            var, thr, lte = self._get_atom(nvars)
            return stl.Atom(var, thr, lte)
        else:
            depth += 1
            return self._sample_internal_node(nvars,depth)

    def _get_temporal_parameters(self):
        if rnd.rand() < self.unbound_prob:
            return True, False, 0, 0
        elif rnd.rand() < self.right_unbound_prob:
            return False, True, rnd.randint(self.time_bound_max_range), 1
        else:
            left_bound = rnd.randint(self.time_bound_max_range)
            return False, False, left_bound, rnd.randint(left_bound, self.time_bound_max_range) + 1

    def _get_atom(self, nvars):
        variable = rnd.randint(nvars)
        lte = rnd.rand() > 0.5
        threshold = rnd.normal(self.threshold_mean, self.threshold_sd)
        return variable, threshold, lte

    # def _get_atom(self, nvars):
    #    variable = rnd.randint(nvars)
    #    lte = rnd.rand() > 0.5
    #    threshold = rnd.uniform(self.threshold_bounds[0], self.threshold_bounds[1])
    #    return variable, threshold, lte

    def replace_not_form(self, atom_node):
        '''Changes an atom >= into NOT <='''
        new_atom = atom_node
        new_atom.lte=1
        new_not=stl.Not(new_atom)
        return new_not

    def single_sign_conversion(self, formula):
        '''
        This function take an STL formula and converts it such as all the atoms of the type x_i>=t
        are converted in NOT x_i<=t
        '''
        current_node = formula
        if type(current_node) is not stl.Atom:
            if (type(current_node) is stl.And) or (type(current_node) is stl.Or) or (type(current_node) is stl.Until):
                current_child=current_node.left_child
                if type(current_child) is stl.Atom and current_child.lte==0:
                    current_node.left_child=self.replace_not_form(current_child)                    
                else:
                    self.single_sign_conversion(current_child)

                current_child=current_node.right_child
                if type(current_child) is stl.Atom and current_child.lte==0:
                    current_node.right_child=self.replace_not_form(current_child)
                else:
                    self.single_sign_conversion(current_child)
            else:
                current_child=current_node.child
                if type(current_child) is stl.Atom  and current_child.lte==0:
                    current_node.child=self.replace_not_form(current_child)
                else:
                    self.single_sign_conversion(current_child)


