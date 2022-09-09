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
    max_value = (2 ** resolution)  # maximum value depending on resolution

    dim = 2 if dim3 == False else 3


    np_array = (np.zeros((max_value**dim, dim), dtype=int))

    # set inital values to populate array
    cnt_one = 0
    cnt_two = 0
    if dim3 == True: cnt_three = 0

    for i in range(0, max_value**dim):
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

    morton_value = []
    array_unpack = []
    for code in morton_codes:
        if(dim3 == False):
            morton_value.append(code)
            x, y = m.unpack(code)
            array_unpack.append((x, y))
        else:
            x, y, z = m.unpack(code)
            array_unpack.append((x, y, z))

    zip(*array_unpack)

    if dim3 == True:
        ax = plt.axes(projection='3d')
        ax.plot3D(*zip(*array_unpack), 'o-')
    else:
        plt.plot(*zip(*array_unpack), 'o-')

    print(morton_codes)
    plt.show()


def calcDistanceToNeighbors (np_array, position, distance_circle):

    zeilen, spalten = np_array.shape

    np_array = np.append(np_array, np.empty((zeilen, 1)), axis=1)

    point = np_array[position]

    cnt = 0
    for ref_point in np_array:
        np_array[cnt][2] = math.dist(point, ref_point)
        cnt+=1

    df = pd.DataFrame(np_array)
    sorted_df = df.sort_values(by=2, ascending=True)

    # filter mit minimaler distance anwenden
    sorted_df = sorted_df[sorted_df[2] <= distance_circle]
    # spalten der punkte auswählen und in array zurückwandeln
    sorted_df = sorted_df[[0, 1]]
    sorted_df = sorted_df.astype(int)

    array_sorted = np.array(sorted_df)
    print(array_sorted)

    return array_sorted #TODO: das hier funktioniert nicht zuverlässig....



def getMaxSpredInMorton (np_array, distance_circle):
    zeilen, spalten = np_array.shape

    maxDistance = 0
    point_max_distance_from = []
    point_max_distance_to = []

    for i in range(0, zeilen):
        temp_array = calcDistanceToNeighbors(np_array, i, distance_circle)
        morton_codes, m = mortonFromArray(resolution, temp_array)
        print("morten code", morton_codes)

        morton_codes.sort()
        point_from = morton_codes.pop(0)
        point_max = morton_codes.pop()
        distance_latent_cur = math.dist([point_from], [point_max])
        print("first:", point_from, "last", point_max)
        print("Distance: ", distance_latent_cur)
        #print("currentDistance", distance_latent_cur)
        if distance_latent_cur > maxDistance:
            maxDistance = distance_latent_cur
            point_max_distance_from = np_array[i]
            point_max_distance_to = m.unpack(point_max)

        distance_latent_cur = 0

    return maxDistance, point_max_distance_from, point_max_distance_to




# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    resolution = 2  # anzahl der bits, die nötig sind um die werte im originalen array abzubilden (z.B. 4 für werte zwischen 0-15)

    np_array = generateArray(resolution)
    print(np_array)
    # Plot Verlauf des Arrays
    morton_codes, m = mortonFromArray(resolution, np_array)

    morton_code_array = np.array(morton_codes)
    np_array = np.column_stack((np_array, morton_codes))

    print(np_array)

    plt.plot(np_array[:, 0], np_array[:, 1], "-o")

    for i, point in enumerate(np_array):
        plt.scatter(point[0], point[1], marker='o')
        plt.annotate(point[2], (np_array[i, 0]+0.02, np_array[i, 1]+0.02))
    plt.show()




    #print(morton_codes)

    #dist, point_from, poin_to = getMaxSpredInMorton(np_array, 1)
    #print("Maximale Distanz", dist, "on between: ", point_from, "and", poin_to)

    # plotZline(morton_codes, m)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
