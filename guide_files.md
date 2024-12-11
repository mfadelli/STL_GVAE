 # Help
## stl.py
``Node`` is an abstract class for nodes. Each actual class has *boolean* and *quantitative* methods for the truth value of the subtree starting from the node.
- ``Atom`` is the class for atom nodes. Initialised with (var_index, threshold, lte).
- ``Not`` has no parameters expect for the child (one child). 
- ``And`` has two children (right and left child).
- ``Or`` has two children (right and left child).
- ``Globally`` has one child and the following parameters for the time interval (unbound, right_unbound, left_time_bound, right_time_bound, adapt_unbound).
- ``Eventually`` has one child and the same parameters of Globally for the time interval.
- ``Until`` has two children (left and right children) plus the same parameters of the other temporal operators.
### Example
To construct the formula $$G_{[0,5]}\big[\lnot(x_1\geq 1) \land F_{[2,3]}(x_2 \leq -1)\big]$$
``a,b = Atom(1,1,False), Atom(2,-1,True) ``\
``nt = Not(a)``\
``f = Eventually(b,False,False,2,2)``\
``nd = And(nt,f)``\
``g = Globally(nd,False,False,0,4)``

## traj_measure.py
``Measure`` abstract class to be overwritten.\
``BaseMeasure`` is the actual class used with default parameters (mu0=0.0, sigma0=1.0, mu1=0.0, sigma1=1.0, q=0.02, q0=0.5, device="cpu", density=1, period=False). It has the following methods.
- ``BaseMeasure.sample`` with parameters (samples, varn, points=100) where samples is the number of trajectories to be sampled. It outputs a pytorch tensor of shape $$samples \times varn \times points$$ 

### Example
``m=BaseMeasure()``\
``t=m.sample(samples=1000,varn=3, points=100)``\
then t is a tensor such that ``torch.Size([1000, 3, 100]) ``.

## phis_generator.py
``StlGenerator`` is a class with the following parameters:\
(leaf_prob = 0.3, inner_node_prob (list), threshold_mean = 0.0, threshold_sd = 1.0, unbound_prob = 0.1, right_unbound_prob = 0.2,time_bound_max_range = 20, adaptive_unbound_temporal_ops = True, max_timespan = 100, max_depth = 6). It contains the following methods:
- ``StlGenerator.sample`` has parameters (nvars, one_sign = True); samples a formula.
- ``StlGenerator.bag_sample`` has parameters (bag_size, nvars, one_sign=True); it samples many formulae.
- ``StlGenerator.replace_not_form`` and ``StlGenerator.single_sign_conversion`` convert a formula in the form with only the $\leq$ sign, eventually by adding a not.

### Example
``gen=StlGenerator(leaf_prob=0.7,max_depth=5)``\
``phis=gen.bag_sample(50,3)``\
phis is a list of objects (trees).\
If ``x`` is a tensor of trajectories the line
``phis[0].quantitative(x)``\
returns a tensor of the trajectories evaluated for the first formula.

## kernel.py
``StlKernel`` is a class with parameters \
(measure, normalize=True, exp_kernel=True, sigma2=0.2, integrate_time=False, samples=10000, varn=2, points=100,boolean=False, signals=None).\
When initalized it generates a tensor of trajectories (or takes one as an input).
It has the following methods:
- ``StlKernel._compute_robustness_no_time`` takes as input a list of formulae $\{\varphi_i\}_i$ and compute the robustness matrix between these formulae and the trajectories generated. 
- ``Stlkernel.compute_bag_bag`` given two sets of formulae $\{\varphi_i\}_i$ and $\{\psi_i\}_i$ it computes the kernel between elements of the two sets (on the set of the trajectories generated).
- ``Stlkernel.compute_one_bag`` returns the tensor of the kernel of a formula $\varphi$ w.r.t. a set of formulae $\{\psi_i\}_i$ (i.e. $[k(\varphi,\psi_1),\dots,k(\varphi,\psi_n)]$, the kernel embedding for the formula $\varphi$).