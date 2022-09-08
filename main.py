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

    resolution = 4  # anzahl der bits, die nötig sind um die werte im originalen array abzubilden (z.B. 4 für werte zwischen 0-15)

    # meta parameters
    np_array = np.array([[15, 15], [3, 4]])
    morton_codes = []

    value_cnt, dimension = np_array.shape

    m = morton.Morton(dimensions=dimension, bits=resolution)

    for i in range(0, value_cnt):
        morton_codes.append(m.pack(np_array[i, 0], np_array[i, 1])) #hier ggf. noch die Dimension anpassen

    for code in morton_codes:
        print(code)
        print(m.unpack(code))


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
