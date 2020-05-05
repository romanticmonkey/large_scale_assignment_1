from mpi4py import MPI 
import numpy as np
import time
import scipy.stats as sts
from scipy.optimize import minimize

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def simulation(rho):
    mu = 3.0 
    sigma = 1.0 
    z_0 = mu

    # Set simulation parameters, draw all idiosyncratic random shocks, 
    # # and create empty containers 
    S = int(1000/size) # Set the number of lives to simulate 
    T = int(4160) # Set the number of periods for each simulation 
    np.random.seed(25)
    eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, S)) 
    z_mat = np.zeros((T, S)) 
    z_mat[0, :] = z_0

    tracker = np.array([])
    for s_ind in range(S): 
        z_tm1 = z_0 
        for t_ind in range(T): 
            e_t = eps_mat[t_ind, s_ind] 
            z_t = rho * z_tm1 + (1 - rho) * mu + e_t 
            if z_t <= 0:
                tracker = np.append(tracker, t_ind)
                break
            else:
                z_tm1 = z_t
    
    tracker_all = None
    if rank == 0:
        tracker_all = np.empty([S*size], dtype='float')
    comm.Gather(sendbuf = tracker, recvbuf = tracker_all, root=0)

    return -tracker_all.mean()

def main():

    t0 = time.time()
    x0 = [0.1]
    res = minimize(simulation, x0, method='Nelder-Mead')
    time_elapsed = time.time() - t0
    print('Optimized rho: {}'.format(res.x[0]))
    print('Max Value: {}'.format(res.fun))
    print('Time taken: {}'.format(time_elapsed))

if __name__ == '__main__':
    main()

