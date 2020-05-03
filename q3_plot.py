import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np


def best_and_plot(csv_filename):
    df = pd.read_csv(csv_filename, names=['rho', 'period'])
    max_entry = df.sort_values(by='period', ascending=False).iloc[0]
    print(max_entry)
    df.sort_values(by='rho', inplace=True)
    df.plot(x='rho', y='period')
    plt.ylabel('Time period before 1st negative')
    plt.title('Grid Search for Best Rho')
    plt.savefig(f'{csv_filename[:-4]}_plot.png')

def main():
    best_and_plot(sys.argv[-1])


if __name__ == '__main__':
    main()
