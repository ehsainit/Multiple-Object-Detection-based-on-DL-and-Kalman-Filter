import sys

import pandas as pd


def readh5(f):
    data = pd.read_hdf(f)
    print(data)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        f = sys.argv[1]
        readh5(f)
    else:
        print('Usage: python3 readh5.py [filename]')
