import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import pyopencl.clrandom as clrand
import pyopencl.tools as cltools
from pyopencl.scan import GenericScanKernel
import matplotlib.pyplot as plt
import time
import pandas as pd
from scipy.optimize import minimize

def avg_first_negative(matrix):
    neg_tracker = np.array([])
    row_count = matrix.shape[0]
    col_count = matrix.shape[1]

    for i in range(row_count):
        for j in range(col_count):
          if matrix[i, j] <= 0:
              neg_tracker = np.append(neg_tracker, j)
              break
    
    return neg_tracker.mean()

def sim_lifetime(rho):
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    t0 = time.time()

    S = 1000
    T = int(4160)

    rand_gen = clrand.PhiloxGenerator(ctx, seed=25)
    eps_mat = rand_gen.normal(queue, (S*T), np.float32, mu=0, sigma=1)

    z_row = np.array(([3] + [0] * (T-1)), dtype=np.float32)
    z_mat = np.tile(z_row, int(S))
    z_mat = cl_array.to_device(queue, z_mat)

    seg_boundaries = [1] + [0]*(T-1)
    seg_boundaries = np.array(seg_boundaries, dtype=np.uint8)
    seg_boundary_flags = np.tile(seg_boundaries, int(S))
    seg_boundary_flags = cl_array.to_device(queue, seg_boundary_flags)
    
    prefix_sum = GenericScanKernel(ctx, np.float32,
                arguments="__global float *ary, __global char *segflags, "
                    "__global float *eps, __global float *out, __global float r",
                input_expr="ary[i] + eps[i] + 3*(1-r)",
                scan_expr="across_seg_boundary ? b : (r*a+b)", neutral="0",
                is_segment_start_expr="segflags[i]",
                output_statement="out[i] = item",
                options=[])

    dev_result = cl_array.empty_like(eps_mat)

    prefix_sum(z_mat, seg_boundary_flags, eps_mat, dev_result, rho)

    simulation_all = (dev_result.get()
                         .reshape(S, T).transpose()
                         )
    
    avg_first_neg = avg_first_negative(simulation_all)
   
    return -avg_first_neg # turned negative for minimization

def optimize():
    x0 = [0.1]
    res = minimize(sim_lifetime, x0, method='Nelder-Mead')

    print('Optimized rho: {}'.format(res.x[0]))
    print('Max Value: {}'.format(res.fun))

def main():
    t0 = time.time()
    optimize()
    time_elapsed = time.time() - t0
    print('Time taken: {}'.format(time_elapsed))

if __name__ == '__main__':
    main()

