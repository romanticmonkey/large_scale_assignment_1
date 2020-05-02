import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import pyopencl.clrandom as clrand
import pyopencl.tools as cltools
from pyopencl.scan import GenericScanKernel
import matplotlib.pyplot as plt
import time

def sim_lifetime(S, T):
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    t0 = time.time()

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
                    "__global float *eps, __global float *out",
                input_expr="ary[i]",
                scan_expr="across_seg_boundary ? b : (a+b)", neutral="0",
                is_segment_start_expr="segflags[i]",
                output_statement="out[i] = 0.5*item + eps[i] + 1.5",
                options=[])

    dev_result = cl_array.empty_like(eps_mat)

    prefix_sum(z_mat, seg_boundary_flags, eps_mat, dev_result)

    simulation_all = (dev_result.get()
                         .reshape(S, T)
                         .transpose()
                         )

    average_finish = np.mean(simulation_all[-1])
    std_finish = np.std(simulation_all[-1])
    final_time = time.time()
    time_elapsed = final_time - t0

    print("Simulated %d lifetimes in: %f seconds"
                % (S, time_elapsed))
    print("Average final health score: %f, Standard Deviation: %f"
                % (average_finish, std_finish))
   
    return

def main():
    sim_lifetime(1000, int(4160))


if __name__ == '__main__':
    main()
