
from msot_sync_strategies import MoSyncStrategyNoConstraints
from mo_adaptive_sampling import *
from ZDT1 import *
from pySOT import SymmetricLatinHypercube, RBFInterpolant, CubicKernel, \
    LinearTail
from poap.controller import SerialController, ThreadController, BasicWorkerThread
import numpy as np
import os.path
import logging


def main():
    if not os.path.exists("./logfiles"):
        os.makedirs("logfiles")
    if os.path.exists("./logfiles/test_simple.log"):
        os.remove("./logfiles/test_simple.log")
    logging.basicConfig(filename="./logfiles/test_simple.log",
                        level=logging.INFO)

    nthreads = 4
    maxeval = 400
    nsamples = nthreads

    print("\nNumber of threads: " + str(nthreads))
    print("Maximum number of evaluations: " + str(maxeval))
    print("Sampling method: Mixed")
    print("Experimental design: Symmetric Latin Hypercube")
    print("Surrogate: Cubic RBF")

    data = LZF2()
    num = 1

    # Create a strategy and a controller
    controller = ThreadController()
    if data.dim < 10:
        numcand = 150*data.dim
    elif data.dim < 20:
        numcand = 50*data.dim
    else:
        numcand = 25*data.dim
    sampling_method = [CandidateDYCORS(data=data, numcand=numcand),
                       CandidateDYUNIF(data=data ,numcand=numcand)]
    #sampling_method=MultiSampling(sampling_method, [0, 1, 2, 3])
    controller.strategy = \
        MoSyncStrategyNoConstraints(
            worker_id=0, data=data,
            maxeval=maxeval, nsamples=nsamples,
            exp_design=SymmetricLatinHypercube(dim=data.dim, npts=2*(data.dim+1)),
            response_surface=RBFInterpolant(kernel=CubicKernel, tail=LinearTail,
                                            maxp=maxeval),
            sampling_method=MultiSampling(sampling_method, [1, 0]))

    # Launch the threads and give them access to the objective function
    for _ in range(nthreads):
        worker = BasicWorkerThread(controller, data.objfunction)
        controller.launch_worker(worker)

    # Run the optimization strategy
    def merit(r):
        return r.value[0]
    result = controller.run(merit=merit)

    controller.strategy.save_plot(num)


if __name__ == '__main__':
    main()
