import matplotlib.pyplot as plt
import pandas as pd
import sys


def plot_csv(csv_filename):
    df = pd.read_csv(csv_filename, columns=['cores', 'time'])

    df.plot(x='cores', y='time')
    plt.title('Program Running Time (s) From 1 to 100 Cores')

    plt.savefig(f'{csv_filename[:-4]}_plot.png')


def main():
    plot_csv(sys.argv[-1])


if __name__ == '__main__':
    main()
