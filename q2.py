import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import pyopencl.clrandom as clrand
import pyopencl.tools as cltools
from pyopencl.scan import GenericScanKernel, GenericDebugScanKernel
import matplotlib.pyplot as plt
import time
import scipy.stats as sts

def simulation(S, T):
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    mem_pool = cltools.MemoryPool(cltools.ImmediateAllocator(queue))

    t0 = time.time()

    rho = 0.5
    mu = 3.0
    sigma = 1.0
  
    rand_gen = clrand.PhiloxGenerator(ctx, seed=25)
    eps_mat = rand_gen.normal(queue, (T*S), np.float32, mu=0, sigma=sigma, 
                            allocator=mem_pool)
  
    z_row = np.array(([mu] + [0] * (T-1)), dtype=np.uint8)
    z_mat = np.tile(z_row, int(S))
    z_mat = cl_array.to_device(queue, z_mat, allocator=mem_pool)

    seg_boundaries = [1] + [0]*(T-1)
    seg_boundaries = np.array(seg_boundaries, dtype=np.uint8)
    seg_boundary_flags = np.tile(seg_boundaries, int(S))
    seg_boundary_flags = cl_array.to_device(queue, seg_boundary_flags, 
                                          allocator=mem_pool)

    prefix_sum = GenericDebugScanKernel(ctx, np.float32,
                arguments="__global float *ary, __global char *segflags, "
                    "__global float *eps, __global float *out",
                input_expr="ary[i]",
                scan_expr="across_seg_boundary ? b :(0.5*a + b + 1.5 + eps[i])", 
                neutral="0",
                is_segment_start_expr="segflags[i]",
                output_statement="out[i] = item",
                options=[])
            
    dev_result = cl_array.empty_like(z_mat, allocator=mem_pool)

    prefix_sum(z_mat, seg_boundary_flags, eps_mat, dev_result)

    simulation_all = (dev_result.get().reshape(S, T).transpose())

    average_finish = np.mean(simulation_all[-1])
    std_finish = np.std(simulation_all[-1])
    final_time = time.time()
    time_elapsed = final_time - t0
  
    print("Simulated %d health shocks in: %f seconds"
                % (S, time_elapsed))
    print("Average final position: %f, Standard Deviation: %f"
                % (average_finish, std_finish))

  
def main():
  simulation(1000, int(4160))


if __name__ == '__main__':
    main()


