"""
.. module:: mo_adaptive_sampling
   :synopsis: Ways of finding the next point to evaluate in the adaptive phase of MoPySOT

.. moduleauthor:: Taimoor Akhtar
                David Eriksson <dme65@cornell.edu>,
                David Bindel <bindel@cornell.edu>
"""

import math
from pySOT.utils import *
import scipy.spatial as scp
from pySOT.heuristic_methods import GeneticAlgorithm as GA
from scipy.optimize import minimize
import scipy.stats as stats
import types
from mo_utils import *
import random
from hv import HyperVolume
import numpy as np
import time


def __fix_docs(cls):
    """Help function for stealing docs from the parent"""
    for name, func in vars(cls).items():
        if isinstance(func, types.FunctionType) and not func.__doc__:
            for parent in cls.__bases__:
                parfunc = getattr(parent, name, None)
                if parfunc and getattr(parfunc, '__doc__', None):
                    func.__doc__ = parfunc.__doc__
                    break
    return cls

class MultiSampling(object):
    """Maintains a list of adaptive sampling methods

    A collection of adaptive sampling methods and weights so that the user
    can use multiple adaptive sampling methods for the same optimization
    problem. This object keeps an internal list of proposed points
    in order to be able to compute the minimum distance from a point
    to all proposed evaluations. This list has to be reset each time
    the optimization algorithm restarts

    :param strategy_list: List of adaptive sampling methods to use
    :type strategy_list: list
    :param cycle: List of integers that specifies the sampling order, e.g., [0, 0, 1] uses
        method1, method1, method2, method1, method1, method2, ...
    :type cycle: list
    :raise ValueError: If cycle is incorrect

    :ivar sampling_strategies: List of adaptive sampling methods to use
    :ivar cycle: List that specifies the sampling order
    :ivar nstrats: Number of adaptive sampling strategies
    :ivar current_strat: The next adaptive sampling strategy to be used
    :ivar proposed_points: List of points proposed to the optimization algorithm
    :ivar data: Optimization problem object
    :ivar fhat: Response surface object
    :ivar budget: Remaining evaluation budget

    .. note:: This object needs to be initialized with the init method. This is done when the
        initial phase has finished.

    .. todo:: Get rid of the proposed_points object and replace it by something that is
        controlled by the strategy.
    """

    def __init__(self, strategy_list, cycle):
        if cycle is None:
            cycle = range(len(strategy_list))
        if (not all(isinstance(i, int) for i in cycle)) or \
                np.min(cycle) < 0 or \
                np.max(cycle) > len(strategy_list)-1:
            raise ValueError("Incorrect cycle!!")
        self.sampling_strategies = strategy_list
        self.nstrats = len(strategy_list)
        self.cycle = cycle
        self.current_strat= 0
        self.proposed_points = None
        self.data = strategy_list[0].data
        self.fhat = None
        self.budget = None
        self.n0 = None

    def init(self, start_sample, fhat, budget):
        """Initialize the sampling method after the initial phase

        This initializes the list of sampling methods after the initial phase
        has finished and the experimental design has been evaluated. The user
        provides the points in the experimental design, the surrogate model,
        and the remaining evaluation budget.

        :param start_sample: Points in the experimental design
        :type start_sample: numpy.array
        :param fhat: Surrogate model
        :type fhat: Object
        :param budget: Evaluation budget
        :type budget: int
        """

        self.proposed_points = start_sample
        self.fhat = fhat
        self.n0 = start_sample.shape[0]
        for i in range(self.nstrats):
            self.sampling_strategies[i].init(self.proposed_points, fhat, budget)

    def remove_point(self, x):
        """Remove x from proposed_points

        This removes x from the list of proposed points in the case where the optimization
        strategy decides to not evaluate x.

        :param x: Point to be removed
        :type x: numpy.array
        :return: True if points was removed, False otherwise
        :type: bool
        """

        idx = np.sum(np.abs(self.proposed_points - x), axis=1).argmin()
        if np.sum(np.abs(self.proposed_points[idx, :] - x)) < 1e-10:
            self.proposed_points = np.delete(self.proposed_points, idx, axis=0)
            for i in range(self.nstrats):
                self.sampling_strategies[i].remove_point(x)
            return True
        return False

    def make_points(self, npts, xbest, sigma, front, subset=None, proj_fun=None):
        """Proposes npts new points to evaluate

        :param npts: Number of points to select
        :type npts: int
        :param xbest: Best solution found so far
        :type xbest: numpy.array
        :param sigma: Current sampling radius w.r.t the unit box
        :type sigma: float
        :param subset: Coordinates to perturb
        :type subset: numpy.array
        :param proj_fun: Routine for projecting infeasible points onto the feasible region
        :type proj_fun: Object
        :param merit: Merit function for selecting candidate points
        :type merit: Object

        :return: Points selected for evaluation, of size npts x dim
        :rtype: numpy.array

        .. todo:: Change the merit function from being hard-coded
        """

        new_points = np.zeros((npts, self.data.dim))

        # Figure out what we need to generate
        npoints = np.zeros((self.nstrats,), dtype=int)
        for i in range(npts):
            npoints[self.cycle[self.current_strat]] += 1
            self.current_strat = (self.current_strat + 1) % len(self.cycle)

        # Now generate the points from one strategy at the time
        count = 0
        for i in range(self.nstrats):
            if npoints[i] > 0:
                new_points[count:count+npoints[i], :] = \
                    self.sampling_strategies[i].make_points(npts=npoints[i], xbest=xbest,
                                                            sigma=sigma, front=front, subset=subset,
                                                            proj_fun=proj_fun)

                count += npoints[i]
                # Update list of proposed points
                for j in range(self.nstrats):
                    if j != i:
                        self.sampling_strategies[j].proposed_points = \
                            self.sampling_strategies[i].proposed_points

        return new_points

class CandidateSRBF(object):
    """An implementation of Stochastic RBF

    This is an implementation of the candidate points method that is
    proposed in the first SRBF paper. Candidate points are generated
    by making normally distributed perturbations with standard
    deviation sigma around the best solution. The candidate point that
    minimizes a specified merit function is selected as the next
    point to evaluate.

    :param data: Optimization problem object
    :type data: Object
    :param numcand: Number of candidate points to be used. Default is min([5000, 100*data.dim])
    :type numcand: int
    :param weights: Weights used for the merit function, to balance exploration vs exploitation
    :type weights: list of numpy.array

    :raise ValueError: If number of candidate points is
        incorrect or if the weights aren't a list in [0, 1]

    :ivar data: Optimization problem object
    :ivar fhat: Response surface object
    :ivar xrange: Variable ranges, xup - xlow
    :ivar dtol: Smallest allowed distance between evaluated points 1e-3 * sqrt(dim)
    :ivar weights: Weights used for the merit function
    :ivar proposed_points: List of points proposed to the optimization algorithm
    :ivar dmerit: Minimum distance between the points and the proposed points
    :ivar xcand: Candidate points
    :ivar fhvals: Predicted values by the surrogate model
    :ivar next_weight: Index of the next weight to be used
    :ivar numcand: Number of candidate points
    :ivar budget: Remaining evaluation budget

    .. note:: This object needs to be initialized with the init method. This is done when the
        initial phase has finished.

    .. todo:: Get rid of the proposed_points object and replace it by something that is
        controlled by the strategy.
    """

    def __init__(self, data, numcand=None, weights=None):
        self.data = data
        self.fhat = None
        self.xrange = self.data.xup - self.data.xlow
        self.dtol = 1e-3 * math.sqrt(data.dim)
        self.weights = weights
        if self.weights is None:
            self.weights = [0.3, 0.5, 0.8, 0.95]
        self.proposed_points = None
        self.dmerit = None
        self.xcand = None
        self.fhvals = None
        self.next_weight = 0
        self.numcand = numcand
        if self.numcand is None:
            self.numcand = min([5000, 100*data.dim])
        self.budget = None

        # Check that the inputs make sense
        if not(isinstance(self.numcand, int) and self.numcand > 0):
            raise ValueError("The number of candidate points has to be a positive integer")
        if not((isinstance(self.weights, np.ndarray) or isinstance(self.weights, list))
               and max(self.weights) <= 1 and min(self.weights) >= 0):
            raise ValueError("Incorrect weights")

    def init(self, start_sample, fhat, budget):
        """Initialize the sampling method after the initial phase

        This initializes the list of sampling methods after the initial phase
        has finished and the experimental design has been evaluated. The user
        provides the points in the experimental design, the surrogate model,
        and the remaining evaluation budget.

        :param start_sample: Points in the experimental design
        :type start_sample: numpy.array
        :param fhat: Surrogate model
        :type fhat: Object
        :param budget: Evaluation budget
        :type budget: int
        """

        self.proposed_points = start_sample
        self.budget = budget
        self.fhat = fhat

    def remove_point(self, x):
        """Remove x from proposed_points

        This removes x from the list of proposed points in the case where the optimization
        strategy decides to not evaluate x.

        :param x: Point to be removed
        :type x: numpy.array
        :return: True if points was removed, False otherwise
        :type: bool
        """

        idx = np.sum(np.abs(self.proposed_points - x), axis=1).argmin()
        if np.sum(np.abs(self.proposed_points[idx, :] - x)) < 1e-10:
            self.proposed_points = np.delete(self.proposed_points, idx, axis=0)
            return True
        return False

    def __generate_cand__(self, scalefactors, xbest, subset):
        self.xcand = np.ones((self.numcand,  self.data.dim)) * xbest
        if np.random.rand() <= 0.65:
            for i in subset:
                lower, upper = self.data.xlow[i], self.data.xup[i]
                ssigma = scalefactors[i]
                self.xcand[:, i] = stats.truncnorm.rvs(
                    (lower - xbest[i]) / ssigma, (upper - xbest[i]) / ssigma,
                    loc=xbest[i], scale=ssigma, size=self.numcand)
        else:
            for i in subset:
                lower, upper = self.data.xlow[i], self.data.xup[i]
                ssigma = scalefactors[i]
                self.xcand[:, i] = stats.norm.rvs(
                    loc=xbest[i], scale=ssigma, size=self.numcand)
                self.xcand[:,i] = np.minimum(upper, np.maximum(lower, self.xcand[:,i]))

    def make_points(self, npts, xbest, sigma, front, subset=None, proj_fun=None):
        """Proposes npts new points to evaluate

        :param npts: Number of points to select
        :type npts: int
        :param xbest: Best solution found so far
        :type xbest: numpy.array
        :param sigma: Current sampling radius w.r.t the unit box
        :type sigma: float
        :param subset: Coordinates to perturb, the others are fixed
        :type subset: numpy.array
        :param proj_fun: Routine for projecting infeasible points onto the feasible region
        :type proj_fun: Object
        :param merit: Merit function for selecting candidate points
        :type merit: Object

        :return: Points selected for evaluation, of size npts x dim
        :rtype: numpy.array

        .. todo:: Change the merit function from being hard-coded
        """

        if subset is None:
            subset = np.arange(0, self.data.dim)
        scalefactors = sigma * self.xrange
        # Make sure that the scale factors are correct for
        # the integer variables (at least 1)
        ind = np.intersect1d(self.data.integer, subset)
        if len(ind) > 0:
            scalefactors[ind] = np.maximum(scalefactors[ind], 1.0)

        # Generate candidate points
        start = time.clock()
        self.__generate_cand__(scalefactors, xbest, subset)
        if proj_fun is not None:
            self.xcand = proj_fun(self.xcand)

        # Compute surrogate approximations for each objective for each candidate
        fhvals = np.zeros((self.numcand, self.data.nobj))
        i = 0
        for fhat in self.fhat:
            fvals = fhat.evals(self.xcand)
            fvals = fvals.flatten()
            fhvals[:,i] = fvals
            i = i+1

        end = time.clock()
        totalTime = end - start
        #print('CAndidate Generation Time: ' + str(totalTime))


        start = time.clock()
        # Find non-dominated candidate points
        (ndf_index, df_index) = ND_Front(np.transpose(fhvals))
        self.xcand_nd = self.xcand[ndf_index,:]
        self.fhvals_nd = fhvals[ndf_index,:]
        end = time.clock()
        totalTime = end - start
        #print('Candidate ND Time: ' + str(totalTime))

        # Use merit function to propose new point
        start = time.clock()
        if random.uniform(0,1) <= 0.35:
            dists = scp.distance.cdist(self.xcand_nd, self.proposed_points)
            self.dmerit = np.amin(np.asmatrix(dists), axis=1)
            index = np.argmax(self.dmerit)
            xnew = self.xcand_nd[index,:]

        else:
            # Use a random solution from ND candidates
            (M, l) = self.xcand_nd.shape
            # index = random.randint(0,M-1)
            # xnew = self.xcand_nd[index,:]

            # Use hypervolume contribution to select the next best
            # Step 1 - Normalize Objectives
            temp_all = np.vstack((self.fhvals_nd, front))
            minpt = np.zeros(self.data.nobj)
            maxpt = np.zeros(self.data.nobj)
            for i in range(self.data.nobj):
                minpt[i] = np.min(temp_all[:,i])
                maxpt[i] = np.max(temp_all[:,i])
            normalized_front = np.asarray(normalize_objectives(front, minpt, maxpt))
            (N, l) = normalized_front.shape
            normalized_cand_fh = np.asarray(normalize_objectives(self.fhvals_nd.tolist(), minpt, maxpt))
            # Step 2 - Compute Hypervolume Contribution
            hv = HyperVolume(1.1*np.ones(self.data.nobj))
            base_hv = hv.compute(normalized_front)
            hv_vals = np.zeros(M)
            for i in range(M):
                nd = range(N)
                dominated = []
                fvals = np.vstack((normalized_front, normalized_cand_fh[i,:]))
                (nd, dominated) = ND_Add(np.transpose(fvals), dominated, nd)
                if dominated and dominated[0] == N: # Record is dominated
                    hv_vals[i] = 0
                else:
                    new_hv = hv.compute(fvals)
                    hv_vals[i] = new_hv - base_hv
            index = np.argmax(hv_vals)
            xnew = self.xcand_nd[index,:]

        # update list of proposed points
        self.proposed_points = np.vstack((self.proposed_points,
                                          np.asmatrix(xnew)))

        end = time.clock()
        totalTime = end - start
        #print('Candidate Selection Time: ' + str(totalTime))
        return xnew


class CandidateDYCORS(CandidateSRBF):
    """An implementation of the DYCORS method

    The DYCORS method only perturbs a subset of the dimensions when
    perturbing the best solution. The probability for a dimension
    to be perturbed decreases after each evaluation and is capped
    in order to guarantee global convergence.

    :param data: Optimization problem object
    :type data: Object
    :param numcand: Number of candidate points to be used. Default is min([5000, 100*data.dim])
    :type numcand: int
    :param weights: Weights used for the merit function, to balance exploration vs exploitation
    :type weights: list of numpy.array

    :raise ValueError: If number of candidate points is
        incorrect or if the weights aren't a list in [0, 1]

    :ivar data: Optimization problem object
    :ivar fhat: Response surface object
    :ivar xrange: Variable ranges, xup - xlow
    :ivar dtol: Smallest allowed distance between evaluated points 1e-3 * sqrt(dim)
    :ivar weights: Weights used for the merit function
    :ivar proposed_points: List of points proposed to the optimization algorithm
    :ivar dmerit: Minimum distance between the points and the proposed points
    :ivar xcand: Candidate points
    :ivar fhvals: Predicted values by the surrogate model
    :ivar next_weight: Index of the next weight to be used
    :ivar numcand: Number of candidate points
    :ivar budget: Remaining evaluation budget
    :ivar minprob: Smallest allowed perturbation probability
    :ivar n0: Evaluations spent when the initial phase ended
    :ivar probfun: Function that computes the perturbation probability of a given iteration

    .. note:: This object needs to be initialized with the init method. This is done when the
        initial phase has finished.

    .. todo:: Get rid of the proposed_points object and replace it by something that is
        controlled by the strategy.
    """

    def __init__(self, data, numcand=None, weights=None):
        CandidateSRBF.__init__(self, data, numcand=numcand, weights=weights)
        self.minprob = np.min([1.0, 1.0/self.data.dim])
        self.n0 = None

        if data.dim <= 1:
            raise ValueError("You can't use DYCORS on a 1d problem")

        def probfun(numevals, budget):
            if budget < 2:
                return 0
            return min([20.0/data.dim, 1.0]) * (1.0 - (np.log(numevals + 1.0) / np.log(budget)))
        self.probfun = probfun

    def init(self, start_sample, fhat, budget):
        CandidateSRBF.init(self, start_sample, fhat, budget)
        self.n0 = start_sample.shape[0]

    def remove_point(self, x):
        return CandidateSRBF.remove_point(self, x)

    def make_points(self, npts, xbest, sigma, front, subset=None, proj_fun=None):
        return CandidateSRBF.make_points(self, npts, xbest, sigma, front, subset, proj_fun)

    def __generate_cand__(self, scalefactors, xbest, subset):
        ddsprob = self.probfun(self.proposed_points.shape[0] - self.n0, self.budget - self.n0)
        ddsprob = np.max([self.minprob, ddsprob])

        nlen = len(subset)

        # Fix when nlen is 1
        # Todo: Use SRBF instead
        if nlen == 1:
            ar = np.ones((self.numcand, 1))
        else:
            ar = (np.random.rand(self.numcand, nlen) < ddsprob)
            ind = np.where(np.sum(ar, axis=1) == 0)[0]
            ar[ind, np.random.randint(0, nlen - 1, size=len(ind))] = 1

        self.xcand = np.ones((self.numcand, self.data.dim)) * xbest

        if np.random.rand() <= 0.65:
            for i in range(nlen):
                lower, upper = self.data.xlow[i], self.data.xup[i]
                ssigma = scalefactors[subset[i]]
                ind = np.where(ar[:, i] == 1)[0]
                self.xcand[ind, subset[i]] = stats.truncnorm.rvs(
                    (lower - xbest[subset[i]]) / ssigma, (upper - xbest[subset[i]]) / ssigma,
                    loc=xbest[subset[i]], scale=ssigma, size=len(ind))
        else:
            for i in range(nlen):
                lower, upper = self.data.xlow[i], self.data.xup[i]
                ssigma = scalefactors[subset[i]]
                ind = np.where(ar[:, i] == 1)[0]
                self.xcand[ind, subset[i]] = stats.norm.rvs(
                    loc=xbest[subset[i]], scale=ssigma, size=len(ind))
                self.xcand[:,i] = np.minimum(upper, np.maximum(lower, self.xcand[:,i]))

class CandidateELIPSE(CandidateSRBF):
    """An implementation of the ELLIPTICAL method

    The DYCORS method only perturbs a subset of the dimensions when
    perturbing the best solution. The probability for a dimension
    to be perturbed decreases after each evaluation and is capped
    in order to guarantee global convergence.

    :param data: Optimization problem object
    :type data: Object
    :param numcand: Number of candidate points to be used. Default is min([5000, 100*data.dim])
    :type numcand: int
    :param weights: Weights used for the merit function, to balance exploration vs exploitation
    :type weights: list of numpy.array

    :raise ValueError: If number of candidate points is
        incorrect or if the weights aren't a list in [0, 1]

    :ivar data: Optimization problem object
    :ivar fhat: Response surface object
    :ivar xrange: Variable ranges, xup - xlow
    :ivar dtol: Smallest allowed distance between evaluated points 1e-3 * sqrt(dim)
    :ivar weights: Weights used for the merit function
    :ivar proposed_points: List of points proposed to the optimization algorithm
    :ivar dmerit: Minimum distance between the points and the proposed points
    :ivar xcand: Candidate points
    :ivar fhvals: Predicted values by the surrogate model
    :ivar next_weight: Index of the next weight to be used
    :ivar numcand: Number of candidate points
    :ivar budget: Remaining evaluation budget
    :ivar minprob: Smallest allowed perturbation probability
    :ivar n0: Evaluations spent when the initial phase ended
    :ivar probfun: Function that computes the perturbation probability of a given iteration

    .. note:: This object needs to be initialized with the init method. This is done when the
        initial phase has finished.

    .. todo:: Get rid of the proposed_points object and replace it by something that is
        controlled by the strategy.
    """

    def __init__(self, data, numcand=None, weights=None):
        CandidateSRBF.__init__(self, data, numcand=numcand, weights=weights)

    def init(self, start_sample, fhat, budget):
        CandidateSRBF.init(self, start_sample, fhat, budget)

    def remove_point(self, x):
        return CandidateSRBF.remove_point(self, x)

    def make_points(self, npts, xbest, sigma, front, subset=None, proj_fun=None):
        return CandidateSRBF.make_points(self, npts, xbest, sigma, front, subset, proj_fun)

    def __generate_cand__(self, scalefactors, xbest, subset):
        self.xcand = np.ones((self.numcand,  self.data.dim)) * xbest
        if np.random.rand() <= 0.65:
            for i in subset:
                lower, upper = self.data.xlow[i], self.data.xup[i]
                ssigma = scalefactors[i]
                ssigma = np.abs(np.random.normal(ssigma, ssigma / 2))
                self.xcand[:, i] = stats.truncnorm.rvs(
                    (lower - xbest[i]) / ssigma, (upper - xbest[i]) / ssigma,
                    loc=xbest[i], scale=ssigma, size=self.numcand)
        else:
            for i in subset:
                lower, upper = self.data.xlow[i], self.data.xup[i]
                ssigma = scalefactors[i]
                ssigma = np.abs(np.random.normal(ssigma, ssigma / 2))
                self.xcand[:, i] = stats.norm.rvs(
                    loc=xbest[i], scale=ssigma, size=self.numcand)
                self.xcand[:,i] = np.minimum(upper, np.maximum(lower, self.xcand[:,i]))


class CandidateDYUNIF(CandidateDYCORS):
    """An implementation of the DDS candidate points method

    Only a few candidate points are generated
    and the candidate point with the lowest value predicted
    by the surrogate model is selected. The DDS method only
    perturbs a subset of the dimensions when perturbing the
    best solution. The probability for a dimension to be
    perturbed decreases after each evaluation and is capped
    in order to guarantee global convergence.

    :param data: Optimization problem object
    :type data: Object
    :param numcand: Number of candidate points to be used. Default is min([5000, 100*data.dim])
    :type numcand: int
    :param weights: Weights used for the merit function, to balance exploration vs exploitation
    :type weights: list of numpy.array

    :raise ValueError: If number of candidate points is
        incorrect or if the weights aren't a list in [0, 1]

    :ivar data: Optimization problem object
    :ivar fhat: Response surface object
    :ivar xrange: Variable ranges, xup - xlow
    :ivar dtol: Smallest allowed distance between evaluated points 1e-3 * sqrt(dim)
    :ivar weights: Weights used for the merit function
    :ivar proposed_points: List of points proposed to the optimization algorithm
    :ivar dmerit: Minimum distance between the points and the proposed points
    :ivar xcand: Candidate points
    :ivar fhvals: Predicted values by the surrogate model
    :ivar next_weight: Index of the next weight to be used
    :ivar numcand: Number of candidate points
    :ivar budget: Remaining evaluation budget
    :ivar probfun: Function that computes the perturbation probability of a given iteration

    .. note:: This object needs to be initialized with the init method. This is done when the
        initial phase has finished.

    .. todo:: Get rid of the proposed_points object and replace it by something that is
        controlled by the strategy.
    """

    def __init__(self, data, numcand=None, weights=None):
        CandidateDYCORS.__init__(self, data, numcand=numcand, weights=weights)

    def init(self, start_sample, fhat, budget):
        CandidateDYCORS.init(self, start_sample, fhat, budget)
        self.n0 = start_sample.shape[0]

    def remove_point(self, x):
        return CandidateDYCORS.remove_point(self, x)

    def make_points(self, npts, xbest, sigma, front, subset=None, proj_fun=None):
        return CandidateDYCORS.make_points(self, npts, xbest, sigma, front, subset, proj_fun)

    def __generate_cand__(self, scalefactors, xbest, subset):
        ddsprob = self.probfun(self.proposed_points.shape[0] - self.n0, self.budget - self.n0)
        ddsprob = np.max([self.minprob, ddsprob])

        nlen = len(subset)

        # Fix when nlen is 1
        # Todo: Use SRBF instead
        if nlen == 1:
            ar = np.ones((self.numcand, 1))
        else:
            ar = (np.random.rand(self.numcand, nlen) < ddsprob)
            ind = np.where(np.sum(ar, axis=1) == 0)[0]
            ar[ind, np.random.randint(0, nlen - 1, size=len(ind))] = 1

        self.xcand = np.ones((self.numcand, self.data.dim)) * xbest
        for i in range(nlen):
            ind = np.where(ar[:, i] == 1)[0]
            self.xcand[ind, subset[i]] = np.random.uniform(
            self.data.xlow[subset[i]], self.data.xup[subset[i]], size=len(ind))


