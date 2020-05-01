from mpi4py import MPI 
import numpy as np
import time
import scipy.stats as sts
import csv

def simulation():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    t0 = time.time()

    # Set model parameters 
    rho = 0.5 
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

    for s_ind in range(S): 
        z_tm1 = z_0 
        for t_ind in range(T): 
            e_t = eps_mat[t_ind, s_ind] 
            z_t = rho * z_tm1 + (1 - rho) * mu + e_t 
            z_mat[t_ind, s_ind] = z_t 
            z_tm1 = z_t

    z_mat_all = None
    if rank == 0:
        z_mat_all = np.empty([T, S*size], dtype='float')
    comm.Gather(sendbuf = z_mat, recvbuff = z_mat_all, root=0)

    if rank == 0: 
        time_elapsed = time.time() - t0

        with open('q1_result.csv', 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([size, time_elapsed])


def main():
    simulation()

if __name__ == '__main__':
    main()






