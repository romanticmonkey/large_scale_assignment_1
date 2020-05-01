import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np


def plot_csv(csv_filename):
    df = pd.read_csv(csv_filename, names=['cores', 'run_time'])
    df.sort_values(by='cores', inplace=True)
    df.plot(x='cores', y='run_time')
    plt.ylabel('time (s)')
    plt.title('Program Running Time (s) From 1 to 20 Cores')
    plt.xticks(np.arange(1,21))
    plt.savefig(f'{csv_filename[:-4]}_plot.png')


def main():
    plot_csv(sys.argv[-1])


if __name__ == '__main__':
    main()
