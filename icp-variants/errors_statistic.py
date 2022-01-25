from __future__ import print_function
import pandas as pd
import sys

if __name__ == "__main__":
    if len(sys.argv ) < 2:
        print("Usage: python errors_statistic.py {error_file} {optional: stat_name}")
        print("Package requirement: pandas.")
        sys.exit(1)
    filename = sys.argv[1]
    stat_name = 'rmse'
    if len(sys.argv) > 2:
        stat_name = sys.argv[2]
    df = pd.read_csv(filename, index_col=False, names=[stat_name])
    print(df.describe())