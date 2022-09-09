# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import morton

dim3 = False



def mortonFromArray(resolution, np_array):

    # meta parameters
    morton_codes = []

    value_cnt, dimension = np_array.shape

    m = morton.Morton(dimensions=dimension, bits=resolution)

    for i in range(0, value_cnt):
        if(dim3 == False):
            morton_codes.append(m.pack(np_array[i, 0], np_array[i, 1]))
        else:
            morton_codes.append(m.pack(np_array[i, 0], np_array[i, 1], np_array[i, 2]))

    return morton_codes, m



def generateArray(resolution):
    max_value = (2 ** resolution) - 1  # maximum value depending on resolution

    dim = 2 if dim3 == False else 3

    np_array = (np.zeros((max_value ** dim, dim), dtype=int))

    # set inital values to populate array
    cnt_one = 0
    cnt_two = 0
    if dim3 == True: cnt_three = 0

    for i in range(0, max_value ** dim):
        # write values to array
        np_array[i][0] = cnt_one
        np_array[i][1] = cnt_two
        if dim3 == True: np_array[i][2] = cnt_three
        # reset values
        if(dim3 == False):
            cnt_two += 1
            if (cnt_two >= max_value):
                cnt_one += 1
                cnt_two = 0
        else:
            cnt_three += 1
            if(cnt_three >= max_value):
                cnt_two += 1
                cnt_three = 0
                if(cnt_two >= max_value):
                    cnt_one += 1
                    cnt_two = 0
                    cnt_three = 0

    return np_array



def plotZline(morton_codes, m):
    # werte nach latentspace sortieren
    morton_codes.sort()

    array_unpack = []
    for code in morton_codes:
        if(dim3 == False):
            x, y = m.unpack(code)
            array_unpack.append((x, y))
        else:
            x, y, z = m.unpack(code)
            array_unpack.append((x, y, z))

    zip(*array_unpack)

    if dim3 == True:
        ax = plt.axes(projection='3d')
        ax.plot3D(*zip(*array_unpack))
    else:
        plt.plot(*zip(*array_unpack))

<<<<<<< Updated upstream
=======


def plotScatterAndLines(np_array, morton_codes, m):
    morton_code_array = np.array(morton_codes)
    np_array = np.column_stack((np_array, morton_codes))

    plt.plot(np_array[:, 0], np_array[:, 1], "-o")

    for i, point in enumerate(np_array):
        plt.scatter(point[0], point[1], marker='o')
        plt.annotate(point[2], (np_array[i, 0] + 0.02, np_array[i, 1] + 0.02))

    plotZline(morton_codes, m)
>>>>>>> Stashed changes
    plt.show()


def plotLatentSpace(morten_codes):
    print(morton_codes)

    plt.plot(morton_codes)
    plt.show()




# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    resolution = 4  # anzahl der bits, die nötig sind um die werte im originalen array abzubilden (z.B. 4 für werte zwischen 0-15)

    np_array = generateArray(resolution)

    morton_codes, m = mortonFromArray(resolution, np_array)
<<<<<<< Updated upstream
=======

    plotScatterAndLines(np_array, morton_codes, m)




    #print(morton_codes)
>>>>>>> Stashed changes

    plotLatentSpace(morton_codes)

    plotZline(morton_codes, m)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
