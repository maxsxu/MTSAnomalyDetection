# coding=utf-8

# Created by max on 17-12-19

import os
import sys
import time


def main(args):
    pass


if __name__ == "__main__":
    start = time.time()
    print("Start: " + str(start))

    main(sys.argv[1:])

    elapsed = (time.time() - start)
    print("Used {0:0.3f} seconds".format(elapsed))