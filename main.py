# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import morton


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Strg+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # meta parameters
    dimension = 3
    resolution = 32 # in bites
    x = 50
    y = 23
    z = 235


    m = morton.Morton(dimensions=dimension, bits=resolution)
    code = m.pack(x, y, z)  # pack two values

    values = m.unpack(code)  # should get back 13 and 42

    print(values)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
