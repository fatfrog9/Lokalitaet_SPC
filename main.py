# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math
import multiprocessing

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import morton
from tqdm import tqdm
import sys
from scipy.spatial import distance
from hilbertcurve.hilbertcurve import HilbertCurve
import random
from decimal import *


dim3 = False


def generateArray_df_morton(resolution, dimension):
    print("Generate array.")
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


def plotScatterAnnotationLatentSpace_df(df_array, curve, ax):

    #df_array.plot(x = 'x', y = 'y', marker="o", ax = ax)
    #for idx, row in df_array.iterrows():
    #    ax.annotate(row[curve], (row['x']+0.02, row['y']+0.02))

    df_array = df_array.sort_values(by=curve)
    # print(df_array)
    df_array.reset_index()

    df_array.plot(x='x', y='y', marker="o", ax=ax, label="SFC")
    for idx, row in df_array.iterrows():
        ax.annotate(row[curve], (row['x'] + 0.07, row['y'] + 0.02))

    ax.set_ylabel('y', rotation=0)


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

def calc_Performance(geofence, search_mask, m):

    flag = True
    # check, whether each point is considered:
    for x in range(geofence[0][0], geofence[1][0] + 1):
        for y in range(geofence[0][1], geofence[1][1] + 1):
            morton = m.pack(x, y)
            if not (search_mask.isin([morton]).any().any()) == True:
                flag = False

    #print("All areas covered:", flag)

    geofence_area = (geofence[1][0] - geofence[0][0] + 1) * (geofence[1][1] - geofence[0][1] + 1)
    search_area = len(search_mask.axes[0])
    precision = geofence_area / (geofence_area + (search_area - geofence_area))

    #print("Precision:", precision)

    return flag, precision


# def search(geofence, df_array, curve, m, ax):
#     offset = 10
#     faktor_multiply = 10000
#
#     A = [0, 0]  # geofence[0]
#     A[0] = int((geofence[0][0] + offset) * faktor_multiply)
#     A[1] = int((geofence[0][1] + offset) * faktor_multiply)
#     C = [0, 0]  # geofence[1]
#     C[0] = int((geofence[1][0] + offset) * faktor_multiply)
#     C[1] = int((geofence[1][1] + offset) * faktor_multiply)
#     B = [A[0], C[1]]
#     D = [A[1], C[0]]
#
#     search_space = [m.pack(A[0], A[1]), m.pack(C[0], C[1])]
#
#     ax.add_patch(Rectangle((A[0]-0.25, A[1]-0.25), C[0]-A[0]+0.5, C[1]-A[1]+0.5, fill=False, color='red', lw = 2))
#
#     #df_array[(df_array.morton >= search_space[0]) & (df_array.morton <= search_space[1])].sort_values(by='morton').reset_index().plot(x='x', y='y', marker="o", ax=ax, label="SearchSpace")
#
#     search_mask = df_array[(df_array.morton >= search_space[0]) & (df_array.morton <= search_space[1])]
#
#     min = 0
#     max = (2**resolution)-1
#     search_mask = identifyNonRelvantAreas(m, geofence, search_mask, min, min, max, max)
#     search_mask.sort_values(by='morton').reset_index().plot(x='x', y='y', marker="o", ax=ax, label="SearchSpace")
#
#     geofence_area = (C[0] - A[0] + 1) * (C[1] - A[1] + 1)
#     search_area = len(search_mask.axes[0])
#     precision = geofence_area / (geofence_area + (search_area - geofence_area))
#
#     print("Search space for geofence:", geofence, "requires search between", search_space[0], "and", search_space[1], "requires", search_area, "queries to search ", geofence_area, "entries." )
#     print("Precision of", round(precision, 3))
#
#     return search_mask
#
# def identifyNonRelvantAreas(m, geofence, search_mask, min_value_x, min_value_y, max_value_x, max_value_y, offset, faktor_multiply):
#
#     if (m.pack(max_value_x, max_value_y) - m.pack(min_value_x, min_value_y)) <=3:
#         return search_mask
#
#     A = [0, 0]  # geofence[0]
#     A[0] = int((geofence[0][0] + offset) * faktor_multiply)
#     A[1] = int((geofence[0][1] + offset) * faktor_multiply)
#     C = [0, 0]  # geofence[1]
#     C[0] = int((geofence[1][0] + offset) * faktor_multiply)
#     C[1] = int((geofence[1][1] + offset) * faktor_multiply)
#
#     half_value_x = int(((max_value_x - min_value_x) / 2) + 0.5 + min_value_x)
#     half_value_y = int(((max_value_y - min_value_y) / 2) + 0.5 + min_value_y)
#
#     # search_mask = df_array[['x', 'y', 'morton']]
#
#     Q1 = False
#     Q2 = False
#     Q3 = False
#     Q4 = False
#
#     if (A[0] < half_value_x) & (A[1] < half_value_y) & (C[0] >= half_value_x) & (C[1] >= half_value_y):
#         # alle
#         Q1 = True
#         Q2 = True
#         Q3 = True
#         Q4 = True
#     elif (A[0] < half_value_x) & (A[1] >= half_value_y) & (C[0] >= half_value_x) & (C[1] >= half_value_y):
#         # oben beide
#         Q3 = True
#         Q4 = True
#     elif (A[0] < half_value_x) & (A[1] < half_value_y) & (C[0] >= half_value_x) & (C[1] < half_value_y):
#         # unten beide
#         Q1 = True
#         Q2 = True
#     elif (A[0] < half_value_x) & (A[1] < half_value_y) & (C[0] < half_value_x) & (C[1] >= half_value_y):
#         # links beide
#         Q1 = True
#         Q3 = True
#     elif (A[0] >= half_value_x) & (A[1] < half_value_y) & (C[0] >= half_value_x) & (C[1] >= half_value_y):
#         # rechts beide
#         Q2 = True
#         Q4 = True
#     elif (A[0] < half_value_x) & (A[1] >= half_value_y) & (C[0] < half_value_x) & (C[1] >= half_value_y):
#         # oben links
#         Q3 = True
#     elif (A[0] < half_value_x) & (A[1] < half_value_y) & (C[0] < half_value_x) & (C[1] < half_value_y):
#         # unten links
#         Q1 = True
#     elif (A[0] >= half_value_x) & (A[1] < half_value_y) & (C[0] >= half_value_x) & (C[1] < half_value_y):
#         # unten rechts
#         Q2 = True
#     elif (A[0] >= half_value_x) & (A[1] >= half_value_y) & (C[0] >= half_value_x) & (C[1] >= half_value_y):
#         # oben rechts
#         Q4 = True
#     else:
#         #irgendwas stimmt mit der eingabe nicht
#         sys.exit("Geofence is incorrect; please check!")
#
#     Q1_range = (m.pack(min_value_x, min_value_y), m.pack((half_value_x-1), (half_value_y-1)))
#     Q2_range = (m.pack(half_value_x, min_value_y), m.pack(max_value_x, (half_value_y - 1)))
#     Q3_range = (m.pack(min_value_x, half_value_y), m.pack((half_value_x - 1), max_value_y))
#     Q4_range = (m.pack(half_value_x, half_value_y), m.pack(max_value_x, max_value_y))
#
#     # print("Q1", Q1_range, "Q2", Q2_range, "Q3", Q3_range, "Q4", Q4_range)
#
#     #Q1_range = (0, 63)
#     #Q2_range = (64, 127)
#     #Q3_range = (128, 191)
#     #Q4_range = (192, 255)
#
#     if Q1 == False:
#         search_mask = search_mask.drop(search_mask[(search_mask.morton < Q1_range[1]+1) & (search_mask.morton > Q1_range[0])].index)
#     else:
#         search_mask = identifyNonRelvantAreas(m, geofence, search_mask, min_value_x=min_value_x, min_value_y=min_value_y,
#                                             max_value_x=half_value_x-1, max_value_y=half_value_y-1, offset=offset, faktor_multiply=faktor_multiply)
#     if Q2 == False:
#         search_mask = search_mask.drop(search_mask[(search_mask.morton < Q2_range[1]+1) & (search_mask.morton > Q2_range[0])].index)
#     else:
#         search_mask = identifyNonRelvantAreas(m, geofence, search_mask, min_value_x=half_value_x, min_value_y=min_value_y,
#                                             max_value_x=max_value_x, max_value_y=half_value_y - 1, offset=offset, faktor_multiply=faktor_multiply)
#     if Q3 == False:
#         search_mask = search_mask.drop(search_mask[(search_mask.morton < Q3_range[1]+1) & (search_mask.morton > Q3_range[0])].index)
#     else:
#         search_mask = identifyNonRelvantAreas(m, geofence, search_mask, min_value_x=min_value_x, min_value_y=half_value_y,
#                                             max_value_x=half_value_x - 1, max_value_y=max_value_y, offset=offset, faktor_multiply=faktor_multiply)
#     if Q4 == False:
#         search_mask = search_mask.drop(search_mask[(search_mask.morton < Q4_range[1]+1) & (search_mask.morton > Q4_range[0])].index)
#     else:
#         search_mask = identifyNonRelvantAreas(m, geofence, search_mask, min_value_x=half_value_x, min_value_y=half_value_y,
#                                             max_value_x=max_value_x, max_value_y=max_value_y, offset=offset, faktor_multiply=faktor_multiply)
#
#     return search_mask

################################################################

def transfer_Geofence_to_Morton(geofence, m, resolution, resolution_search_space):
    offset = 0

    faktor_multiply = 1

    A = [0,0] #geofence[0]
    A[0] = int((geofence[0][0] + offset) * faktor_multiply)
    A[1] = int((geofence[0][1] + offset) * faktor_multiply)
    C = [0,0] #geofence[1]
    C[0] = int((geofence[1][0] + offset) * faktor_multiply)
    C[1] = int((geofence[1][1] + offset) * faktor_multiply)
    B = [A[0], C[1]]
    D = [A[1], C[0]]

    search_space = [m.pack(A[0], A[1]), m.pack(C[0], C[1])] # geofence in morton bereich; erste Wertebereich ermittlung

    #ax.add_patch(Rectangle((A[0]-0.25, A[1]-0.25), C[0]-A[0]+0.5, C[1]-A[1]+0.5, fill=False, color='red', lw = 2))

    #df_array[(df_array.morton >= search_space[0]) & (df_array.morton <= search_space[1])].sort_values(by='morton').reset_index().plot(x='x', y='y', marker="o", ax=ax, label="SearchSpace")

    np_ar = np.arange(search_space[0], (search_space[1]+1), resolution_search_space)
    search_mask = pd.DataFrame(np_ar, columns = ['morton'])
    #print(len(search_mask))

    #search_mask = df_array[(df_array.morton >= search_space[0]) & (df_array.morton <= search_space[1])]

    min = 0
    max = (2**resolution)-1



    search_mask = identifyNonRelvantAreas(m, geofence, search_mask, min, min, max, max, offset, faktor_multiply)

    #search_mask.sort_values(by='morton').reset_index().plot(x='x', y='y', marker="o", ax=ax, label="SearchSpace")

    return search_mask

################################################################

def identifyNonRelvantAreas(m, geofence, search_mask, min_value_x, min_value_y, max_value_x, max_value_y, offset, faktor_multiply):

    # print("identification of non relevant areas", min_value_x, min_value_y, ";", max_value_x, max_value_y, "search_mask has", len(search_mask.index), "lines.")

    if (m.pack(max_value_x, max_value_y) - m.pack(min_value_x, min_value_y)) <=3:
        return search_mask

    A = [0, 0]  # geofence[0]
    A[0] = int((geofence[0][0] + offset) * faktor_multiply)
    A[1] = int((geofence[0][1] + offset) * faktor_multiply)
    C = [0, 0]  # geofence[1]
    C[0] = int((geofence[1][0] + offset) * faktor_multiply)
    C[1] = int((geofence[1][1] + offset) * faktor_multiply)

    #A = geofence[0]
    #C = geofence[1]

    half_value_x = int(((max_value_x - min_value_x) / 2) + 0.5 + min_value_x)
    half_value_y = int(((max_value_y - min_value_y) / 2) + 0.5 + min_value_y)

    # search_mask = df_array[['x', 'y', 'morton']]

    Q1 = False
    Q2 = False
    Q3 = False
    Q4 = False

    if (A[0] < half_value_x) & (A[1] < half_value_y) & (C[0] >= half_value_x) & (C[1] >= half_value_y):
        # alle
        Q1 = True
        Q2 = True
        Q3 = True
        Q4 = True
    elif (A[0] < half_value_x) & (A[1] >= half_value_y) & (C[0] >= half_value_x) & (C[1] >= half_value_y):
        # oben beide
        Q3 = True
        Q4 = True
    elif (A[0] < half_value_x) & (A[1] < half_value_y) & (C[0] >= half_value_x) & (C[1] < half_value_y):
        # unten beide
        Q1 = True
        Q2 = True
    elif (A[0] < half_value_x) & (A[1] < half_value_y) & (C[0] < half_value_x) & (C[1] >= half_value_y):
        # links beide
        Q1 = True
        Q3 = True
    elif (A[0] >= half_value_x) & (A[1] < half_value_y) & (C[0] >= half_value_x) & (C[1] >= half_value_y):
        # rechts beide
        Q2 = True
        Q4 = True
    elif (A[0] < half_value_x) & (A[1] >= half_value_y) & (C[0] < half_value_x) & (C[1] >= half_value_y):
        # oben links
        Q3 = True
    elif (A[0] < half_value_x) & (A[1] < half_value_y) & (C[0] < half_value_x) & (C[1] < half_value_y):
        # unten links
        Q1 = True
    elif (A[0] >= half_value_x) & (A[1] < half_value_y) & (C[0] >= half_value_x) & (C[1] < half_value_y):
        # unten rechts
        Q2 = True
    elif (A[0] >= half_value_x) & (A[1] >= half_value_y) & (C[0] >= half_value_x) & (C[1] >= half_value_y):
        # oben rechts
        Q4 = True
    else:
        #irgendwas stimmt mit der eingabe nicht
        sys.exit("Geofence is incorrect; please check!")

    Q1_range = (m.pack(min_value_x, min_value_y), m.pack((half_value_x-1), (half_value_y-1)))
    Q2_range = (m.pack(half_value_x, min_value_y), m.pack(max_value_x, (half_value_y - 1)))
    Q3_range = (m.pack(min_value_x, half_value_y), m.pack((half_value_x - 1), max_value_y))
    Q4_range = (m.pack(half_value_x, half_value_y), m.pack(max_value_x, max_value_y))


    if Q1 == False:
        search_mask = search_mask.drop(search_mask[(search_mask.morton < (Q1_range[1]+1)) & (search_mask.morton >= Q1_range[0])].index)
    else:
        search_mask = identifyNonRelvantAreas(m, geofence, search_mask, min_value_x=min_value_x, min_value_y=min_value_y, max_value_x=half_value_x-1, max_value_y=half_value_y-1, offset=offset, faktor_multiply=faktor_multiply)
    if Q2 == False:
        search_mask = search_mask.drop(search_mask[(search_mask.morton < (Q2_range[1] + 1)) & (search_mask.morton >= Q2_range[0])].index)
    else:
        search_mask = identifyNonRelvantAreas(m, geofence, search_mask, min_value_x=half_value_x, min_value_y=min_value_y,
                                            max_value_x=max_value_x, max_value_y=half_value_y - 1, offset=offset, faktor_multiply=faktor_multiply)
    if Q3 == False:
        search_mask = search_mask.drop(search_mask[(search_mask.morton < (Q3_range[1] + 1)) & (search_mask.morton >= Q3_range[0])].index)
    else:
        search_mask = identifyNonRelvantAreas(m, geofence, search_mask, min_value_x=min_value_x, min_value_y=half_value_y,
                                            max_value_x=half_value_x - 1, max_value_y=max_value_y, offset=offset, faktor_multiply=faktor_multiply)
    if Q4 == False:
        search_mask = search_mask.drop(search_mask[(search_mask.morton < (Q4_range[1] + 1)) & (search_mask.morton >= Q4_range[0])].index)
    else:
        search_mask = identifyNonRelvantAreas(m, geofence, search_mask, min_value_x=half_value_x, min_value_y=half_value_y,
                                            max_value_x=max_value_x, max_value_y=max_value_y, offset=offset, faktor_multiply=faktor_multiply)

    return search_mask




################################################################


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    print("Hello...")

    results = pd.DataFrame(
        columns=['resolution', 'geofence_x_min', 'geofence_y_min', 'geofence_x_max', 'geofence_y_max',
                 'duration_binary', 'covered', 'precision'])

    #for resolution in range(2,18):

    for resolution in range(3,12):
        #resolution = 4  # anzahl der bits, die nötig sind um die werte im originalen array abzubilden (z.B. 4 für werte zwischen 0-15)# we need like 30 bits
        rangeThreshold = 1.1

        #print("Let's determine the (half) Sample Rate in latent space;"
        #        "maximum distance between points that have an euclidean distance of max: ", rangeThreshold)

        #for resolution in range(2,3):

        print("The resolution is set to", resolution, "Bits.")


        df_array, m, hilbert_curve = generateArray_df_morton(resolution=resolution, dimension=2)

        fig, ax = plt.subplots(2, gridspec_kw={'height_ratios':[5,1]})
        fig.canvas.set_window_title('Search Space with Morton Codes')

        plotScatterAnnotationLatentSpace_df(df_array, 'morton', ax=ax[0])

        for i in tqdm(range(0,1)):

            x_min = random.randint(0, 2**resolution-1)
            y_min = random.randint(0, 2**resolution-1)
            x_max = random.randint(x_min, 2**resolution-1)
            y_max = random.randint(y_min, 2**resolution-1)


            #geofence = [[x_min, y_min], [x_max, y_max]]
            #geofence = [[2,1],[6,5]]
            geofence = [[1, 4], [5, 6]]

            #print("Geofence: ", geofence)

            search_start = time.time()
            # search_mask = search(geofence, df_array, 'morton', m, ax[0])
            search_mask = transfer_Geofence_to_Morton(geofence, m, resolution, 1)
            search_end = time.time()
            #print("Binary search needs", round(search_end-search_start,5), "secounds.")

            covered, precision = calc_Performance(geofence, search_mask, m)

            temp = pd.DataFrame({'resolution': resolution,
                                      'geofence_x_min': geofence[0][0],
                                      'geofence_y_min': geofence[0][1],
                                      'geofence_x_max': geofence[1][0],
                                      'geofence_y_max': geofence[1][1],
                                      'duration_binary': round(search_end-search_start,5),
                                      'covered': covered,
                                      'precision': round(precision,7)}, index=[0])

            results = pd.concat([results, temp], ignore_index=True, sort=False)


            ax[0].title.set_text('data space')
            ax[0].add_patch(
                Rectangle((geofence[0][0] - 0.4, geofence[0][1] - 0.4), geofence[1][0] - geofence[0][0] + 0.8, geofence[1][1] - geofence[0][1] + 0.8, fill=False, color='red',
                          lw=2))
            # print(search_mask)

            filter = df_array["morton"].isin(search_mask['morton'])
            df_relevant_values = df_array[filter]

            df_relevant_values.sort_values(by='morton').reset_index().plot(x='x', y='y', marker="o", ax=ax[0],
                                                                    label="area of interest")


            min = df_array['morton'].min()
            max = df_array['morton'].max()

            ax[1].hist(search_mask['morton'], bins=range(min, max + 1), color='orange')
            ax[1].set_xlim(min, max)
            ax[1].title.set_text('area of interest in one-dimensional space')

            fig.tight_layout()
            plt.show()

            fig.savefig('lokalitaet_dataspace.pdf')

            plotScatterAnnotationLatentSpace_df(df_array, m, ax)




    #print(results)
        #results.to_csv('lokalitaet_results' + resolution + '.csv', index=False)
    #print("We are done!")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
