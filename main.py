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

    calculateSampleRate(df_array, rangeThreshold)

    # print("Determine maximum distance of datapoint with a resolution of", resolution, "Bits.")
    # calcMaximumDistanceBetweenPoints(np_array_morton)

    # plotScatterAnnotationLatentSpace_df(df_array, m)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
