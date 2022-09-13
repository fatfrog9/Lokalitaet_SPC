# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math
import multiprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import morton
from tqdm import tqdm
import sys
from scipy.spatial import distance
from hilbertcurve.hilbertcurve import HilbertCurve
from decimal import *


dim3 = False


def generateArray_df_morton(resolution, dimension):
    max_value = (2 ** resolution)
    dim = 2 if dim3 == False else 3
    m = morton.Morton(dimensions=dimension, bits=resolution)
    hilbert_curve = HilbertCurve(resolution, dimension, n_procs=0)

    df_array = pd.DataFrame(columns=['x', 'y', 'morton', 'hilbert'])

    for cnt_one in tqdm(range (0, max_value)):
        for cnt_two in range(0, max_value):
            temp_df = pd.DataFrame([{'x': cnt_one, 'y': cnt_two, 'morton': m.pack(cnt_one, cnt_two), 'hilbert' : hilbert_curve.distance_from_point((cnt_one, cnt_two))}])
            # print(temp_df)
            df_array = pd.concat([df_array, temp_df], axis=0, ignore_index=True)

    return df_array, m, hilbert_curve


def plotScatterAnnotationLatentSpace_df(df_array, curve):

    df_array.plot(x = 'x', y = 'y', marker="o")
    for idx, row in df_array.iterrows():
        plt.annotate(row[curve], (row['x']+0.02, row['y']+0.02))

    df_array = df_array.sort_values(by=curve)
    # print(df_array)
    df_array.reset_index()

    df_array.plot(x='x', y='y', marker="o")
    for idx, row in df_array.iterrows():
        plt.annotate(row[curve], (row['x'] + 0.02, row['y'] + 0.02))


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
            df.at[i, 'dist_to_prev'] = distance.euclidean([df['x'][i], df['y'][i]], [df['x'][i - 1], df['y'][i - 1]])

    idx_max_distance = df['dist_to_prev'].idxmax()
    max_dist_A = (df['x'][idx_max_distance - 1], df['y'][idx_max_distance - 1]) if idx_max_distance > 0 else (0, 0)
    max_dist_B = (df['x'][idx_max_distance], df['y'][idx_max_distance])

    print("Maximum euclidean distance of ", df['dist_to_prev'][idx_max_distance], " between A=", max_dist_A, " and B=",
          max_dist_B)


def determineSampleRateExperimental(df_array, rangeThreshold, curve):

    print("Determine Sample Rate.")
    maxMortonDist = 0
    refMaxPoint = (0,0)
    distMaxPoint = (0,0)

    for j in tqdm(range(0, len(df_array))):
        distInRange_df = pd.DataFrame(columns=['x', 'y', curve, 'dist'])
        refPoint = (df_array['x'][j], df_array['y'][j])

        for i in range(0, len(df_array)):
            curPoint = (df_array['x'][i], df_array['y'][i])
            dist = distance.euclidean(curPoint, refPoint) #math.dist

            dist_temp = int(dist*100)
            rangeThreshold_temp = int(rangeThreshold*100)

            if dist_temp < rangeThreshold_temp:
                temp_df = pd.DataFrame([{'x' : df_array['x'][i] , 'y' : df_array['y'][i], curve : df_array[curve][i], 'dist' : dist}])
                distInRange_df = pd.concat([distInRange_df, temp_df], axis=0, ignore_index=True)

        #print(distInRange_df)
        #print("max:", distInRange_df['morton'].max(), "min:", distInRange_df['morton'].min())

        mortonRefPoint = df_array[curve][j]
        mortonMin = distInRange_df[curve].min()
        mortonMax = distInRange_df[curve].max()

        mortonMin = mortonRefPoint - mortonMin if mortonMin < mortonRefPoint else mortonMin - mortonRefPoint
        mortonMax = mortonMax - mortonRefPoint if mortonMax > mortonRefPoint else mortonRefPoint - mortonMax

        curMaxMortonDist = mortonMax if mortonMax > mortonMin else mortonMin

        # print("current Distance:",curMaxMortonDist)

        if curMaxMortonDist > maxMortonDist:
            maxMortonDist = curMaxMortonDist
            refMaxPoint = (df_array['x'][j], df_array['y'][j])

            if mortonMax > mortonMin:
                distMaxPoint = (distInRange_df['x'][distInRange_df[curve].astype(float).idxmax()],
                        distInRange_df['y'][distInRange_df[curve].astype(float).idxmax()])
            else:
                distMaxPoint = (distInRange_df['x'][distInRange_df[curve].astype(float).idxmin()],
                        distInRange_df['y'][distInRange_df[curve].astype(float).idxmin()])


        # print("Point: ", curPoint, "Dist: ", curDist)
        #print("MaxMortonDist_current:", curMaxMortonDist)
    print("The maximum", curve, " Distance is experimentally determined as ", maxMortonDist, " between P_ref", refMaxPoint, "and P_min=", distMaxPoint)
    return maxMortonDist, refMaxPoint, distMaxPoint


def calculateSampleRate(resolution, m):

    max_value = 2**resolution-1

    max_A = (0, int((max_value / 2) - 0.5))
    max_B = (0, int((max_value / 2) + 0.5))

    sampleRate = (m.pack(0, int((max_value / 2) + 0.5)) - m.pack(0, int((max_value / 2) - 0.5))) # *2 <- für die eigentliche Rate, hier berechnen wir erstmal nur die Distanz
    print("The maximum Morton Distance ist caluclatet as ", sampleRate, " between P_ref", max_A, "and P_min=", max_B)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # if len(sys.argv) != 2:
    #     print('Invalid Numbers of Arguments. Script will be terminated.')
    #
    # else:
    #     try:
    #         n = int(sys.argv[1])
    #     except IndexError:
    #         print('missing argument')
    #     except ValueError:
    #         print('argument must be an integer')
    #     else:
    #         if n <= 0:
    #             print('argument must be non-negative ang greater than 0')
    #         else:

            print("Hello...")

            resolution = 4  # anzahl der bits, die nötig sind um die werte im originalen array abzubilden (z.B. 4 für werte zwischen 0-15)# we need like 30 bits
            rangeThreshold = 1.1

            print("Let's determine the (half) Sample Rate in latent space;"
                    "maximum distance between points that have an euclidean distance of max: ", rangeThreshold)

            #for resolution in range(2,3):

            print("The resolution is set to", resolution, "Bits.")

            print("Generate array.")
            df_array, m, hilbert_curve = generateArray_df_morton(resolution=resolution, dimension=2)

            print(df_array)

            #determineSampleRateExperimental(df_array, rangeThreshold, 'morton')
            #determineSampleRateExperimental(df_array, rangeThreshold, 'hilbert')
            #calculateSampleRate(resolution, m)

            print()
            # print("Determine maximum distance of datapoint with a resolution of", resolution, "Bits.")
            # calcMaximumDistanceBetweenPoints(np_array_morton)

            #plotScatterAnnotationLatentSpace_df(df_array, 'morton')

            # these values are hilbert curve specific

            plotScatterAnnotationLatentSpace_df(df_array, 'morton')

            geofence = [[0,6], [4,9]]

            search()


            # plotScatterAnnotationLatentSpace_df(df_array, m)

            plt.show()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
