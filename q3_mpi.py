from mpi4py import MPI 
import numpy as np
import time
import scipy.stats as sts
import csv

def grid_search():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    t0 = time.time()

    mu = 3.0 
    sigma = 1.0 
    z_0 = mu

    S = 1000
    T = int(4160) 
    np.random.seed(25)
    eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, S))
    
    search_size = int(200/size)

    if rank == 0:
        rhos = np.linspace(-0.95, 0.95, 200)
        recvbuf_scatter = np.empty(search_size, dtype='d')
    else:
        rhos = None
    comm.Scatter(rhos, recvbuf_scatter, root=0)
    
    for rho in rhos:
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
        
        with open('q3_result.csv', 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([rho, tracker.mean()])
    
    if rank == 0: 
        time_elapsed = time.time() - t0
        print('Time taken to run: {}'.format(time_elapsed))


def main():
    grid_search()

if __name__ == '__main__':
    main()

