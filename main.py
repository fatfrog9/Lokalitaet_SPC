# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import morton


def mortonFromArray(resolution, np_array):

    # meta parameters
    morton_codes = []

    value_cnt, dimension = np_array.shape

    m = morton.Morton(dimensions=dimension, bits=resolution)

    for i in range(0, value_cnt):
        morton_codes.append(m.pack(np_array[i, 0], np_array[i, 1]))  # hier ggf. noch die Dimension anpassen

    return morton_codes, m



def generateArray(resolution):
    max_value = (2 ** resolution) - 1  # maximum value depending on resolution

    np_array = np.zeros((max_value ** 2, 2), dtype=int)

    # set inital values to populate array
    cnt_left = 0
    cnt_right = 0

    for i in range(0, max_value ** 2):
        # write values to array
        np_array[i][0] = cnt_left
        np_array[i][1] = cnt_right
        # reset values
        cnt_right += 1
        if (cnt_right >= max_value):
            cnt_left += 1
            cnt_right = 0

    return np_array

def plotZline(morton_codes, m):
    # werte nach latentspace sortieren
    morton_codes.sort()

    sorted_array = []
    for code in morton_codes:
        x, y = m.unpack(code)
        sorted_array.append((x, y))

    zip(*sorted_array)
    plt.plot(*zip(*sorted_array), 'o-')
    plt.show()




# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    resolution = 4  # anzahl der bits, die nötig sind um die werte im originalen array abzubilden (z.B. 4 für werte zwischen 0-15)
    # np_array = np.array([[15, 15], [3, 4]])

    np_array = generateArray(resolution)
    #print(np_array)

    morton_codes, m = mortonFromArray(resolution, np_array)

    plotZline(morton_codes, m)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
