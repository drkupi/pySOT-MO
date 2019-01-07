"""
.. module:: msot_sync_strategies
   :synopsis: Parallel synchronous MO optimization strategy

.. moduleauthor:: David Bindel <bindel@cornell.edu>,
                David Eriksson <dme65@cornell.edu>,
                Taimoor Akhtar <erita@nus.edu.sg>

"""

from __future__ import print_function
import numpy as np
import math
import logging
from pySOT.experimental_design import SymmetricLatinHypercube, LatinHypercube
from poap.strategy import BaseStrategy, RetryStrategy
from pySOT.rbf import *
from pySOT.utils import *
from pySOT.rs_wrappers import *
import time

from mo_adaptive_sampling import CandidateSRBF
from copy import deepcopy
from mo_utils import *
from hv import HyperVolume
from matplotlib import pyplot as plt

# Get module-level logger
logger = logging.getLogger(__name__)
POSITIVE_INFINITY = float("inf")

class MemoryRecord():
    "Record that Represents Memory of Optimization Progress Attained Around this Center Point"

    def __init__(self, x, fx, sigma, nfail=0, ntabu=0, rank=POSITIVE_INFINITY, fitness=POSITIVE_INFINITY):
        """Initialize the record.

        Args:
            params: Evaluation point for the function
        Kwargs:
            status: Status of the evaluation (default 'pending')
        """
        self.x = x
        self.fx = fx
        self.nfail = nfail
        self.ntabu = ntabu
        self.rank = rank
        self.fitness = fitness
        self.sigma_init = sigma
        self.sigma = sigma
        self.noffsprings = 1
        self.offsprings = []
        self.fhat_pts = []

    def reset(self):
        self.ntabu = 0
        self.nfail = 0
        self.sigma = self.sigma_init


class MemoryArchive():

    def __init__(self, size_max):
        """Initialize the record.

        Args:
            params: Evaluation point for the function
        Kwargs:
            status: Status of the evaluation (default 'pending')
        """
        self.contents = []
        self.size_max = size_max
        self.num_records = 0

    def add(self, record, cur_rank=None):
        if cur_rank == None:
            cur_rank = 1
        if self.contents: # if Archive is not Empty
            ranked = False
            while cur_rank <= len(self.contents): # Traverse through all front to find front where record is to be inserted
                front = self.contents[cur_rank-1]
                dominated_records = []
                fvals = [rec.fx for rec in front]
                num_front = len(fvals)
                nd = range(num_front)
                dominated = []
                fvals.append(record.fx)
                fvals = np.asarray(fvals)
                (nd, dominated) = ND_Add(np.transpose(fvals), dominated, nd)
                if dominated == []: # Record is in front and all other records are also non-dominated
                    ranked = True
                    # 1 - Update Add Record to Current Front in Memory Archive
                    record.rank = cur_rank
                    front.append(record)
                    for item in front: # INDICATE THAT FITNESS needs to be re-evaluated
                        item.fitness = POSITIVE_INFINITY
                    self.num_records+=1
                    break
                if dominated[0] == num_front: # Record is Not this front
                    fvals = None
                else: # this is the front and it dominates other points already on the front
                    ranked = True
                    # 1 - Update Add Record to Current Front in Memory Archive
                    record.rank = cur_rank
                    front.append(record)
                    self.num_records+=1
                    # 2 - Remove dominated solutions from current front and add them later
                    dominated = sorted(dominated, reverse=True)
                    for i in dominated:
                        dominated_record = deepcopy(front[i])
                        front.remove(front[i])
                        self.num_records-=1
                        self.add(dominated_record,cur_rank)
                    for item in front: # INDICATE THAT FITNESS needs to be re-evaluated
                        item.fitness = POSITIVE_INFINITY
                    break
                cur_rank+=1

            if ranked == False:
                record.rank = len(self.contents) + 1
                record.fitness = POSITIVE_INFINITY
                self.contents.append([record])
                self.num_records+=1

        else:
            self.contents.append([record])
            self.num_records+=1
            record.rank = 1
            record.fitness = POSITIVE_INFINITY

        # Make Sure that number of records in archive is less than size_max
        if self.num_records > self.size_max:
            self.contents[-1].remove(self.contents[-1][-1])
            if self.contents[-1] == []:
                self.contents.remove(self.contents[-1])
            self.num_records -=1

    def compute_hv_fitness(self, cur_rank):
        # Step 0 - Obtain fevals of front
        front = deepcopy(self.contents[cur_rank-1])
        nrec = len(front)
        if nrec == 1:
            self.contents[cur_rank-1][0].fitness = 1
        else:
            fvals = [rec.fx for rec in front]
            # Step 1 - Normalize Objectives
            nobj = len(front[0].fx)
            normalized_fvals = normalize_objectives(fvals)
            # Step 2 - Compute Hypervolume Contribution
            hv = HyperVolume(1.1*np.ones(nobj))
            base_hv = hv.compute(np.asarray(normalized_fvals))
            for i in range(nrec):
                fval_without = deepcopy(normalized_fvals)
                fval_without.remove(fval_without[i])
                new_hv = hv.compute(np.asarray(fval_without))
                hv_contrib = base_hv - new_hv
                self.contents[cur_rank-1][i].fitness = hv_contrib

    def select_center_population(self, npts, d_thresh=1.0):
        center_pts = []
        count = 1
        nfronts = len(self.contents)
        cur_rank = 1
        flag_tabu = False  # Only true if all points in archive are tabu
        while count <= npts: # Traverse through Memory Archive to Select Center Population
            front = self.contents[cur_rank-1] # Iterate through fronts
            if front[0].fitness == POSITIVE_INFINITY:
                self.compute_hv_fitness(cur_rank)
            front.sort(key=lambda x: x.fitness, reverse=True)
            for rec in front: # Traverse through sorted front (by fitness)
                if flag_tabu == True: # If we cycled through all fronts and did not get enough pts
                    rec.reset()
                    center_pts.append(rec)
                    count +=1
                    if count > npts:
                        break
                elif rec.ntabu == 0:
                    # Radius Rule Check goes first
                    flag_radius = radius_rule(rec, center_pts, d_thresh)
                    if flag_radius == True:
                        center_pts.append(rec)
                        count +=1
                        if count > npts:
                            break
            cur_rank = int((cur_rank % nfronts) + 1)
            if cur_rank == 1:
                flag_tabu = True
        return center_pts


class MoSyncStrategyNoConstraints(BaseStrategy):
    """Parallel Multi-Objective synchronous optimization strategy without non-bound constraints.

    This class implements the parallel synchronous MOPLS strategy
    described by Akhtar and Shoemaker.  After the initial experimental
    design (which is embarrassingly parallel), the optimization
    proceeds in phases.  During each phase, we allow nsamples
    simultaneous function evaluations.  We insist that these
    evaluations run to completion -- if one fails for whatever reason,
    we will resubmit it.  Samples are drawn randomly from around the
    "best points" as per i) Non-domination rank and ii) epsilon
    contribution, and are sorted according to a merit function.
    After several successive significant improvements, we increase
    the sampling radius; after several failures to improve the function
    value, we decrease the sampling radius.  We restart once the
    sampling radius decreases below a threshold.

    :param worker_id: Start ID in a multi-start setting
    :type worker_id: int
    :param data: Problem parameter data structure
    :type data: Object
    :param response_surface: Surrogate model object
    :type response_surface: Object
    :param maxeval: Stopping criterion. If positive, this is an
                    evaluation budget. If negative, this is a time
                    budget in seconds.
    :type maxeval: int
    :param nsamples: Number of simultaneous fevals allowed
    :type nsamples: int
    :param exp_design: Experimental design
    :type exp_design: Object
    :param sampling_method: Sampling method for finding
        points to evaluate
    :type sampling_method: Object
    :param extra: Points to be added to the experimental design
    :type extra: numpy.array
    :param extra_vals: Values of the points in extra (if known). Use nan for values that are not known.
    :type extra_vals: numpy.array
    """

    def __init__(self, worker_id, data, response_surface, maxeval, nsamples,
                 exp_design=None, sampling_method=None, extra=None, extra_vals=None):

        # Check stopping criterion
        self.start_time = time.time()
        if maxeval < 0:  # Time budget
            self.maxeval = np.inf
            self.time_budget = np.abs(maxeval)
        else:
            self.maxeval = maxeval
            self.time_budget = np.inf

        # Import problem information
        self.worker_id = worker_id
        self.data = data
        self.fhat = []
        if response_surface is None:
            for i in range(self.data.nobj):
                self.fhat.append(RBFInterpolant(kernel=CubicKernel, tail=LinearTail, maxp=maxeval)) #MOPLS ONLY
        else:
            for i in range(self.data.nobj):
                response_surface.reset()  # Just to be sure!
                self.fhat.append(deepcopy(response_surface)) #MOPLS ONLY

        self.ncenters = nsamples
        self.nsamples = 1
        self.numinit = None
        self.extra = extra
        self.extra_vals = extra_vals

        # Default to generate sampling points using Symmetric Latin Hypercube
        self.design = exp_design
        if self.design is None:
            if self.data.dim > 50:
                self.design = LatinHypercube(data.dim, data.dim+1)
            else:
                self.design = SymmetricLatinHypercube(data.dim, 2*(data.dim+1))

        self.xrange = np.asarray(data.xup - data.xlow)

        # algorithm parameters
        self.sigma_min = 0.005
        self.sigma_max = 0.2
        self.sigma_init = 0.2

        self.failtol = max(5, data.dim)
        self.succtol = 3
        self.numeval = 0
        self.status = 0
        self.sigma = 0
        self.resubmitter = RetryStrategy()
        self.xbest = None
        self.fbest = np.inf
        self.fbest_old = None

        # population of centers and long-term archive
        self.centers = []
        self.nd_archives = []
        self.new_pop = []
        self.memory_archive = MemoryArchive(200)
        self.tabu_list = []
        self.evals = []
        self.maxfit = min(200,20*self.data.dim)
        self.d_thresh = 1.0

        # Set up search procedures and initialize
        self.sampling = sampling_method
        if self.sampling is None:
            self.sampling = CandidateSRBF(data)

        self.check_input()

        # Start with first experimental design
        self.sample_initial()

    def check_input(self):
        """Checks that the inputs are correct"""

        self.check_common()
        if hasattr(self.data, "eval_ineq_constraints"):
            raise ValueError("Optimization problem has constraints,\n"
                             "SyncStrategyNoConstraints can't handle constraints")
        if hasattr(self.data, "eval_eq_constraints"):
            raise ValueError("Optimization problem has constraints,\n"
                             "SyncStrategyNoConstraints can't handle constraints")

    def check_common(self):
        """Checks that the inputs are correct"""

        # Check evaluation budget
        if self.extra is None:
            if self.maxeval < self.design.npts:
                raise ValueError("Experimental design is larger than the evaluation budget")
        else:
            # Check the number of unknown extra points
            if self.extra_vals is None:  # All extra point are unknown
                nextra = self.extra.shape[0]
            else:  # We know the values at some extra points so count how many we don't know
                nextra = np.sum(np.isinf(self.extra_vals[0])) + np.sum(np.isnan(self.extra_vals[0]))

            if self.maxeval < self.design.npts + nextra:
                raise ValueError("Experimental design + extra points "
                                 "exceeds the evaluation budget")

        # Check dimensionality
        if self.design.dim != self.data.dim:
            raise ValueError("Experimental design and optimization "
                             "problem have different dimensions")
        if self.extra is not None:
            if self.data.dim != self.extra.shape[1]:
                raise ValueError("Extra point and optimization problem "
                                 "have different dimensions")
            if self.extra_vals is not None:
                if self.extra.shape[0] != len(self.extra_vals):
                    raise ValueError("Extra point values has the wrong length")

        # Check that the optimization problem makes sense
        check_opt_prob(self.data)

    def proj_fun(self, x):
        """Projects a set of points onto the feasible region

        :param x: Points, of size npts x dim
        :type x: numpy.array
        :return: Projected points
        :rtype: numpy.array
        """

        x = np.atleast_2d(x)
        return round_vars(self.data, x)

    def log_completion(self, record):
        """Record a completed evaluation to the log.

        :param record: Record of the function evaluation
        :type record: Object
        """

        xstr = np.array_str(record.params[0], max_line_width=np.inf,
                            precision=5, suppress_small=True)
        fstr = np.array_str(record.value, max_line_width=np.inf,
                            precision=5, suppress_small=True)
        if record.feasible:
            logger.info("{} {} @ {}".format("True", fstr, xstr))
        else:
            logger.info("{} {} @ {}".format("False", fstr, xstr))


    def sample_initial(self):
        """Generate and queue an initial experimental design."""

        if self.numeval == 0:
            logger.info("=== Start ===")
        else:
            logger.info("=== Restart ===")
        for fhat in self.fhat:
            fhat.reset() #MOPLS Only
        self.sigma = self.sigma_init
        self.status = 0
        self.xbest = None
        self.fbest_old = None
        self.fbest = np.inf
        for fhat in self.fhat:
            fhat.reset() #MOPLS Only

        start_sample = self.design.generate_points()
        assert start_sample.shape[1] == self.data.dim, \
            "Dimension mismatch between problem and experimental design"
        start_sample = from_unit_box(start_sample, self.data)

        if self.extra is not None:
            # We know the values if this is a restart, so add the points to the surrogate
            if self.numeval > 0:
                for i in range(len(self.extra_vals)):
                    xx = self.proj_fun(np.copy(self.extra[i, :]))
                    for j in range(self.data.nobj):
                        self.fhat[j].add_point(np.ravel(xx), self.extra_vals[i][j])
            else:  # Check if we know the values of the points
                if self.extra_vals is None:
                    self.extra_vals = np.nan * np.ones((self.extra.shape[0], self.data.nobj))

                for i in range(len(self.extra_vals)):
                    xx = self.proj_fun(np.copy(self.extra[i, :]))
                    if np.isnan(self.extra_vals[i][0]) or np.isinf(self.extra_vals[i][0]):  # We don't know this value
                        proposal = self.propose_eval(np.ravel(xx))
                        proposal.extra_point_id = i  # Decorate the proposal
                        self.resubmitter.rput(proposal)
                    else:  # We know this value
                        for j in range(self.data.nobj):
                            self.fhat[j].add_point(np.ravel(xx), self.extra_vals[i][j])

        # Evaluate the experimental design
        for j in range(min(start_sample.shape[0], self.maxeval - self.numeval)):
            start_sample[j, :] = self.proj_fun(start_sample[j, :])  # Project onto feasible region
            proposal = self.propose_eval(np.copy(start_sample[j, :]))
            self.resubmitter.rput(proposal)

        if self.extra is not None:
            self.sampling.init(np.vstack((start_sample, self.extra)), self.fhat, self.maxeval - self.numeval)
        else:
            self.sampling.init(start_sample, self.fhat, self.maxeval - self.numeval)

        if self.numinit is None:
            self.numinit = start_sample.shape[0]

        print('Initialization completed successfully')

    def update_archives(self):
        """Update the Tabu list, Tabu Tenure, memory archive and non-dominated front.
        """
        # Step 1 - Update Tabu tenure of points in Tabu list and remove from list if tenure goes to zero
        for rec in self.tabu_list:
            rec.ntabu -= 1
            if rec.ntabu == 0:
                self.tabu_list.remove(rec)

        # Step 2 -  Add center to Tabu list if failure count exceeds threshold
        for center in self.centers:
            if center.nfail > 3:
                center.ntabu = 5
                center.nfail = 0
                center.sigma = self.sigma_init
                self.tabu_list.append(center)

        # Step 3 - Add newly Evaluated Points to Memory Archive and update ND_Archives list
        for rec in self.new_pop:
            self.memory_archive.add(rec)
        self.new_pop = []

        # Step 4 - Store the Front in a separate archive
        front = self.memory_archive.contents[0]
        fvals = [rec.fx for rec in front]
        fvals = np.asarray(fvals)
        self.nd_archives.append(fvals)
        self.d_thresh = 1 - float(self.numeval - self.numinit) / float(self.maxeval - self.numinit)

    def sample_adapt(self):
        """Generate and queue samples from the search strategy"""

        # # Step 1 - Add Newly Evaluated Points to Memory Archive
        # start = time.clock()
        self.update_archives()

        front = self.memory_archive.contents[0]
        fvals = [rec.fx for rec in front]
        fvals = np.asarray(fvals)

        # Step 2 - Select The New Center Points
        maxgen = (self.maxeval - self.numinit)/(self.nsamples*self.ncenters)
        curgen = (self.numeval - self.numinit)/(self.nsamples*self.ncenters) + 1

        print('GENERATION NUMBER: ' + str(curgen) + ' OF ' + str(maxgen))
        nsamples = min(self.nsamples*self.ncenters, self.maxeval - self.numeval)
        self.centers = self.memory_archive.select_center_population(self.ncenters, self.d_thresh)

        # end = time.clock()
        # totalTime = end - start
        # print('CENTER SELECTION TIME: ' + str(totalTime))

        self.interactive_plotting(fvals)

        start = time.clock()
        j = 0
        new_points = np.zeros((nsamples,self.data.dim))
        ## Choose points around each center using the sampling scheme identified

        for rec in self.centers:
            xcenter = np.copy(rec.x)
            xsigma = rec.sigma
            if self.fhat[0].nump >= self.maxfit:
                self.fit_local_surrogate(xcenter)
            new_points[j:j+self.nsamples,:] = self.sampling.make_points(npts=1, xbest=xcenter, sigma=xsigma, front=fvals,
                                               proj_fun=self.proj_fun)
            rec.offsprings.append(new_points[j:j+self.nsamples,:])
            j = j + self.nsamples
            if j >= nsamples:
                break

        end = time.clock()
        totalTime = end - start
        print('CANDIDATE SELECTION TIME: ' + str(totalTime))

        for i in range(nsamples):
            proposal = self.propose_eval(np.copy(np.ravel(new_points[i, :])))
            self.resubmitter.rput(proposal)

    def start_batch(self):
        """Generate and queue a new batch of points"""
        # NOTE: There is no re-start in the basic MOPLS strategy
        self.sample_adapt()

    def propose_action(self):
        """Propose an action
        """
        if self.numeval == self.maxeval:
            # Save results to Array and Terminate
            X = np.zeros((self.maxeval, self.data.dim + self.data.nobj))
            all_xvals = [rec.x for rec in self.evals]
            all_xvals = np.asarray(all_xvals)
            X[:,0:self.data.dim] = all_xvals
            all_fvals = [rec.fx for rec in self.evals]
            all_fvals = np.asarray(all_fvals)
            X[:,self.data.dim:self.data.dim + self.data.nobj] = all_fvals
            np.savetxt('final.txt', X)
            return self.propose_terminate()
        elif self.resubmitter.num_eval_outstanding == 0:
            # UPDATE MEMORY ARCHIVE
            self.start_batch()
        return self.resubmitter.get()

    def update_memory(self, X_new, Fval_new):
        """Update the memory archive of centers that have been selected in the past.
        """
        # 1 - Check if new record improves the non-dominated archive
        F = np.vstack((self.nd_archives[-1], np.asarray(Fval_new)))
        (l, M) = F.shape
        nd = range(l-1)
        dominated = []
        (nd, dominated) = ND_Add(np.transpose(F), dominated, nd)
        #self.nd_archives.append(F[nd,:])
        if dominated and dominated[0] == l-1: # Only if new record is dominated
            # 2 - Find out the center to which the new point belongs and update the radius and failure count
            for center in self.centers:
                for offspring_batch in center.offsprings:
                    nsamples = offspring_batch.shape[0]
                for j in range(nsamples):
                    if np.array_equal(np.copy(X_new),offspring_batch[j,:]):
                        center.nfail+=1
                        center.sigma = center.sigma/2
                        break

    def fit_local_surrogate(self, xbest):
        all_fvals = [rec.fx for rec in self.evals]
        all_fvals = np.asarray(all_fvals)
        all_xvals = [rec.x for rec in self.evals]
        all_xvals = np.asarray(all_xvals)
        dists = scp.distance.cdist(np.atleast_2d(xbest), all_xvals)
        index = np.ravel(np.argsort(dists))
        j = 0
        for fhat in self.fhat:
            fhat.reset()
            for i in range(self.maxfit):
                x = np.ravel(all_xvals[index[i],0:self.data.dim])
                fx = all_fvals[index[i],j]
                fhat.add_point(x, fx)
            j=j+1

    def on_complete(self, record):
        """Handle completed function evaluation.

        When a function evaluation is completed we need to ask the constraint
        handler if the function value should be modified which is the case for
        say a penalty method. We also need to print the information to the
        logfile, update the best value found so far and notify the GUI that
        an evaluation has completed.

        :param record: Evaluation record
        """
        self.numeval += 1
        record.worker_id = self.worker_id
        record.worker_numeval = self.numeval
        record.feasible = True
        self.log_completion(record)
        # 1 - Update Response Surface Model
        i = 0
        for fhat in self.fhat:
            fhat.add_point(np.copy(record.params[0]), record.value[i])
            i +=1

        # 2 - Generate a Memory Record of the New Evaluation
        srec = MemoryRecord(np.copy(record.params[0]),record.value,self.sigma_init)
        self.new_pop.append(srec)
        self.evals.append(srec)
         # 3 - Update radius and failure count of center if new point does not improve non-dominated set
        if self.centers:
            self.update_memory(np.copy(record.params[0]), record.value)

    def interactive_plotting(self, fvals):
        """"If interactive plotting is on,
        """
        maxgen = (self.maxeval - self.numinit)/(self.nsamples*self.ncenters)
        curgen = (self.numeval - self.numinit)/(self.nsamples*self.ncenters) + 1
        cent_fvals = [rec.fx for rec in self.centers]
        cent_fvals = np.asarray(cent_fvals)

        plt.show()
        plt.plot(self.data.pf[:,0], self.data.pf[:,1], 'g')
        all_fvals = [rec.fx for rec in self.evals]
        all_fvals = np.asarray(all_fvals)
        plt.plot(all_fvals[:,0], all_fvals[:,1], 'k+')
        plt.plot(cent_fvals[:,0], cent_fvals[:,1], 'ro', markersize = 10)
        plt.plot(fvals[:,0], fvals[:,1], 'b*')
        if self.tabu_list:
            tabu_fvals = [rec.fx for rec in self.tabu_list]
            tabu_fvals = np.asarray(tabu_fvals)
            plt.plot(tabu_fvals[:,0], tabu_fvals[:,1], 'ys')
        plt.draw()
        if curgen < maxgen:
            plt.pause(0.001)
        else:
            plt.show()

    def save_plot(self, i):
        """"If interactive plotting is on,
        """
        plt.figure(i)
        front = self.memory_archive.contents[0]
        fvals = [rec.fx for rec in front]
        fvals = np.asarray(fvals)
        maxgen = (self.maxeval - self.numinit)/(self.nsamples*self.ncenters)
        curgen = (self.numeval - self.numinit)/(self.nsamples*self.ncenters) + 1
        cent_fvals = [rec.fx for rec in self.centers]
        cent_fvals = np.asarray(cent_fvals)
        plt.plot(self.data.pf[:,0], self.data.pf[:,1], 'g')
        all_fvals = [rec.fx for rec in self.evals]
        all_fvals = np.asarray(all_fvals)
        plt.plot(all_fvals[:,0], all_fvals[:,1], 'k+')
        plt.plot(cent_fvals[:,0], cent_fvals[:,1], 'ro', markersize = 10)
        plt.plot(fvals[:,0], fvals[:,1], 'b*')
        if self.tabu_list:
            tabu_fvals = [rec.fx for rec in self.tabu_list]
            tabu_fvals = np.asarray(tabu_fvals)
            plt.plot(tabu_fvals[:,0], tabu_fvals[:,1], 'ys')
        plt.draw()
        plt.savefig('Final')
        plt.clf()


