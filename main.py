# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import morton
from tqdm import tqdm
import sys
from decimal import *


dim3 = False



def mortonFromArray(resolution, np_array):

    # meta parameters
    morton_codes = []

    value_cnt, dimension = np_array.shape

    m = morton.Morton(dimensions=dimension, bits=resolution)

    print("Calculate Morton-Codes." )
    for i in tqdm(range(0, value_cnt)):
        if(dim3 == False):
            morton_codes.append(m.pack(np_array[i, 0], np_array[i, 1]))
        else:
            morton_codes.append(m.pack(np_array[i, 0], np_array[i, 1], np_array[i, 2]))

    return morton_codes, m


def generateArray_df(resolution, dimension):
    max_value = (2 ** resolution)
    dim = 2 if dim3 == False else 3
    m = morton.Morton(dimensions=dimension, bits=resolution)

    df_array = pd.DataFrame(columns=['x', 'y', 'morton'])

    for cnt_one in tqdm(range (0, max_value)):
        for cnt_two in range(0, max_value):
            temp_df = pd.DataFrame([{'x': cnt_one, 'y': cnt_two, 'morton': m.pack(cnt_one, cnt_two)}])
            df_array = pd.concat([df_array, temp_df], axis=0, ignore_index=True)

    return df_array, m


def generateArray(resolution):
    max_value = (2 ** resolution)  # maximum value depending on resolution

    dim = 2 if dim3 == False else 3


    np_array = (np.zeros((max_value**dim, dim), dtype=int))

    # set inital values to populate array
    cnt_one = 0
    cnt_two = 0
    if dim3 == True: cnt_three = 0

    print("Generate datapoint array (", max_value**dim, ";", dim, ").")
    for i in tqdm(range(0, max_value**dim)):
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


def plotScatterAnnotationLatentSpace_df(df_array, m):

    df_array.plot(x = 'x', y = 'y', marker="o")
    for idx, row in df_array.iterrows():
        plt.annotate(row['morton'], (row['x']+0.02, row['y']+0.02))

    df_array = df_array.sort_values(by='morton')
    # print(df_array)
    df_array.reset_index()

    df_array.plot(x='x', y='y', marker="o")
    for idx, row in df_array.iterrows():
        plt.annotate(row['morton'], (row['x'] + 0.02, row['y'] + 0.02))
    plt.show()


    # for i, point in enumerate(np_array_morton):
    #     plt.scatter(point[0], point[1], marker='o')
    #     plt.annotate(point[2], (np_array_morton[i, 0] + 0.02, np_array_morton[i, 1] + 0.02))
    #
    # plotZline(morton_codes, m)
    #
    # morton_codes.sort()
    #
    # morton_value = []
    # array_unpack = []
    # for code in morton_codes:
    #     if (dim3 == False):
    #         morton_value.append(code)
    #         x, y = m.unpack(code)
    #         array_unpack.append((x, y))
    #     else:
    #         x, y, z = m.unpack(code)
    #         array_unpack.append((x, y, z))
    #
    # zip(*array_unpack)
    #
    # if dim3 == True:
    #     ax = plt.axes(projection='3d')
    #     ax.plot3D(*zip(*array_unpack), 'o-')
    # else:
    #     plt.plot(*zip(*array_unpack), 'o-')


def plotScatterAnnotationLatentSpace(np_array_morton, morton_codes, m):

    plt.plot(np_array_morton[:, 0], np_array_morton[:, 1], "-o")

    for i, point in enumerate(np_array_morton):
        plt.scatter(point[0], point[1], marker='o')
        plt.annotate(point[2], (np_array_morton[i, 0] + 0.02, np_array_morton[i, 1] + 0.02))

    plotZline(morton_codes, m)



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

def calcMaximumDistanceBetweenPoints(np_array_morton):
    df = pd.DataFrame(np_array_morton, columns=['x', 'y', 'morton'])
    df = df.sort_values(by='morton', ascending=True)
    df = df.reset_index()

    sys.stdout.write("")
    print("Determine maximum distance.")

    df_length = len(df)
    for i in tqdm(range(0, df_length)):
        if i == 0:
            df['dist_to_prev'] = 0
        else:
            df.at[i, 'dist_to_prev'] = math.dist([df['x'][i], df['y'][i]], [df['x'][i - 1], df['y'][i - 1]])

    idx_max_distance = df['dist_to_prev'].idxmax()
    max_dist_A = (df['x'][idx_max_distance - 1], df['y'][idx_max_distance - 1]) if idx_max_distance > 0 else (0, 0)
    max_dist_B = (df['x'][idx_max_distance], df['y'][idx_max_distance])

    print("Maximum euclidean distance of ", df['dist_to_prev'][idx_max_distance], " between A=", max_dist_A, " and B=",
          max_dist_B)


def calculateSampleRate(df_array, rangeThreshold):
    # df = pd.DataFrame(np_array_morton, columns=['x', 'y', 'morton'])
    # df = df.sort_values(by='morton', ascending=True)

    #Dataframe erweitern und die Distanzen zu jeweiligen Punkten berechnen
    # for j in tqdm(range(0,len(df))): #,len(df)
    #     refPoint = (df['x'][j], df['y'][j])
    #     df.insert(loc=j + 3, column="dist_to_point_" + str(j), value = 0)
    #
    #     for i in range(0, len(df)):
    #         curPoint = (df['x'][i], df['y'][i])
    #         df.at[i, 'dist_to_point_'+str(j)] = math.dist(curPoint, refPoint)

    print("Determine Sample Rate.")
    maxMortonDist = 0
    refMaxPoint = (0,0)
    distMaxPoint = (0,0)

    for j in tqdm(range(0, len(df_array))):
        distInRange_df = pd.DataFrame(columns=['x', 'y', 'morton', 'dist'])
        refPoint = (df_array['x'][j], df_array['y'][j])

        for i in range(0, len(df_array)):
            curPoint = (df_array['x'][i], df_array['y'][i])
            dist = math.dist(curPoint, refPoint)

            dist_temp = int(dist*100)
            rangeThreshold_temp = int(rangeThreshold*100)

            if dist_temp < rangeThreshold_temp:
                temp_df = pd.DataFrame([{'x' : df_array['x'][i] , 'y' : df_array['y'][i], 'morton' : df_array['morton'][i], 'dist' : dist}])
                distInRange_df = pd.concat([distInRange_df, temp_df], axis=0, ignore_index=True)

        #print(distInRange_df)
        #print("max:", distInRange_df['morton'].max(), "min:", distInRange_df['morton'].min())

        mortonRefPoint = df_array['morton'][j]
        mortonMin = distInRange_df['morton'].min()
        mortonMax = distInRange_df['morton'].max()

        mortonMin = mortonRefPoint - mortonMin if mortonMin < mortonRefPoint else mortonMin - mortonRefPoint
        mortonMax = mortonMax - mortonRefPoint if mortonMax > mortonRefPoint else mortonRefPoint - mortonMax

        curMaxMortonDist = mortonMax if mortonMax > mortonMin else mortonMin

        # print("current Distance:",curMaxMortonDist)

        if curMaxMortonDist > maxMortonDist:
            maxMortonDist = curMaxMortonDist
            refMaxPoint = (df_array['x'][j], df_array['y'][j])

            if mortonMax > mortonMin:
                distMaxPoint = (distInRange_df['x'][distInRange_df['morton'].astype(float).idxmax()],
                        distInRange_df['y'][distInRange_df['morton'].astype(float).idxmax()])
            else:
                distMaxPoint = (distInRange_df['x'][distInRange_df['morton'].astype(float).idxmin()],
                        distInRange_df['y'][distInRange_df['morton'].astype(float).idxmin()])


        # print("Point: ", curPoint, "Dist: ", curDist)
        #print("MaxMortonDist_current:", curMaxMortonDist)
    print("The maximum Morton Distance of", maxMortonDist, "is between P_ref", refMaxPoint, "and P_min=", distMaxPoint)
    return maxMortonDist, refMaxPoint, distMaxPoint



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    print("Hello...")

    resolution = 2  # anzahl der bits, die nötig sind um die werte im originalen array abzubilden (z.B. 4 für werte zwischen 0-15)# we need like 30 bits
    rangeThreshold = 1.1

    print("Let's determine the (half) Sample Rate in latent space;"
          "maximum distance between points that have an euclidean distance of max: ", rangeThreshold)
    print("The resolution is set to", resolution, "Bits.")

    print("Generate array.")
    df_array, m = generateArray_df(resolution=resolution, dimension=2)

    #np_array = generateArray(resolution)
    #morton_codes, m = mortonFromArray(resolution, np_array)
    #np_array_morton = np.column_stack((np_array, morton_codes))

    calculateSampleRate(df_array, rangeThreshold)

    # print("Determine maximum distance of datapoint with a resolution of", resolution, "Bits.")
    # calcMaximumDistanceBetweenPoints(np_array_morton)

    plotScatterAnnotationLatentSpace_df(df_array, m)
    plt.show()




    #print(morton_codes)

    #dist, point_from, poin_to = getMaxSpredInMorton(np_array, 1)
    #print("Maximale Distanz", dist, "on between: ", point_from, "and", poin_to)

    # plotZline(morton_codes, m)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
