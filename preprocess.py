#!/usr/bin/env python
# coding: utf-8

import os
import ast
import cv2
import glob
import math
import json
import pickle
import random

import numpy as np
import pandas as pd
import seaborn as sns

from PIL import Image
from tqdm import tqdm

from collections import deque

import matplotlib.pyplot as plt
import matplotlib.animation as ani
import matplotlib.patches as patches

from scipy import signal

from scipy.fft import fft
from scipy.stats import zscore, entropy, pearsonr
from scipy.spatial.distance import euclidean

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from scipy.signal import savgol_filter

polyorder     = 3   # Polynomial order
window_length = 11  # Must be odd and <= size of data


# In[9]:


plt.ioff()


# In[ ]:


### CRC VERSION ###
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--idx', type=int, required=True)
args = parser.parse_args()
arg_idx = args.idx
### ### 


QRADIAN = math.radians(90)


lms_all = (
    'nose',
    'left_eye', 'right_eye',
    'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle'
)


# In[15]:


def rotate_point(x, y, angle):
    """
    Rotate a point (x, y) counterclockwise by a given angle in radians.
    """
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    x_rot = x * cos_angle - y * sin_angle
    y_rot = x * sin_angle + y * cos_angle
    return (x_rot, y_rot)


# ### # Functions


def get_df_scaled_from_df_raw( df_snippet_raw ):

    df_snippet_smooth = df_snippet_raw.copy()

    for col in df_snippet_raw.columns:

        s_raw = df_snippet_raw[col].bfill()
                
        df_snippet_smooth[col] = savgol_filter(s_raw, window_length=window_length, polyorder=polyorder)

    zs = df_snippet_smooth[ [col for col in df_snippet_smooth.columns if '_z' in col] ]
    zs_min  = zs.min().min()
    zs_max  = zs.max().max()
    z_range = zs_max - zs_min
    
    df_snippet_centered = df_snippet_smooth.copy()

    for col in df_snippet_smooth.columns:

        s_smooth = df_snippet_smooth[col]

        if '_x' in col:
            s_centered = s_smooth - df_snippet_smooth['center_hip_x']
        elif '_y' in col:
            s_centered = s_smooth - df_snippet_smooth['center_hip_y']
        else:
#             s_centered = s_smooth - df_snippet_smooth['center_hip_z']
#             s_centered = s_smooth - s_smooth.mean() / (s_smooth.max() - s_smooth.min())
            s_centered = ( zs_max - s_smooth ) / z_range

        df_snippet_centered[col] = s_centered

    rows_rotated    = []
    rotation_angles = []
    torso_lengths   = []
    for _, row in df_snippet_centered.iterrows():
    
        center_shoulder_x, center_shoulder_y = row[['center_shoulder_x','center_shoulder_y']].values
        angle = np.arctan2( center_shoulder_y, center_shoulder_x )
    
        if angle > QRADIAN:
            angle = -1.0*(angle - QRADIAN)
        else:
            angle = QRADIAN - angle
    
        rotation_angles.append(angle)
    
        torso_lengths.append( np.linalg.norm( (center_shoulder_x, center_shoulder_y) ) )
        
        row_rotated = []
        for lm in lms_all:
    
            x = row['{}_x'.format(lm)]
            y = row['{}_y'.format(lm)]
            z = row['{}_z'.format(lm)]
            
            x_rot, y_rot = rotate_point(x, y, angle)
    
            row_rotated.append( x_rot )
            row_rotated.append( y_rot )
            row_rotated.append( z )
    
        rows_rotated.append(row_rotated)

    df_snippet_rotated = pd.DataFrame(rows_rotated, columns=df_snippet_centered.columns[:-6])

    df_snippet_scaled = df_snippet_rotated.copy()

    for col in df_snippet_rotated.columns:
        
        s_rotated = df_snippet_rotated[col]
        
        if '_z' not in col:
            s_scaled  = s_rotated / torso_lengths
            df_snippet_scaled[col] = s_scaled
        
        else:
        	df_snippet_scaled[col] = s_rotated

    return df_snippet_scaled
    
    
##### #####


df_meta = pd.read_csv('./sensor_fusion/df_meta_withJSON_withTimes_goodDetections_min2700.csv', index_col=0)

# os.makedirs('./dfs_snippets_scaled/', exist_ok=True)

for _, row in df_meta.iloc[ (arg_idx - 1) * 8 : (arg_idx ) * 8 ].iterrows():

    # print(row['study_id'])

    num_frames_json = row['num_frames_json']

    fname_json = row['fname_json'].replace('../../','./')
    dirname    = row['dirname_json'].replace('../../','./')
    
    fname_df_raw = glob.glob( os.path.join( dirname, '*raw_entire*.csv' ) )[0]

    start_time1, stop_time1, start_time2, stop_time2 = row[['start_time1', 'stop_time1', 'start_time2', 'stop_time2']].values
    
    # print(start_time1, stop_time1, start_time2, stop_time2)
    
    df_raw = pd.read_csv(fname_df_raw, index_col=0)

    # print(df_raw.shape)

    df_snippet_raw_tmp1 = df_raw.iloc[start_time1: stop_time1]
    df_snippet_raw_tmp2 = df_raw.iloc[start_time2: stop_time2]
        
    df_snippet_raw = pd.concat([df_snippet_raw_tmp1, df_snippet_raw_tmp2]).reset_index().drop('index', axis=1)

    df_snippet_scaled = get_df_scaled_from_df_raw( df_snippet_raw )

    idx_counter = 0

    df_snippet_scaled.to_csv( os.path.join( dirname, fname_json.split('/')[-1].replace('.json','_df_scaled_snippet_idx{}.csv'.format(idx_counter)) ) )
	
    ########

	### BEGINNING ###

    avail_snippets_at_beginning = int(df_snippet_raw_tmp1.index.min()/2700)
    print( 'Available snippets at beginning: {}'.format( avail_snippets_at_beginning ) )

    for n in range(avail_snippets_at_beginning):

        idx_start = df_snippet_raw_tmp1.index.min() - ((n+1)*2700)
        idx_stop  = df_snippet_raw_tmp1.index.min() - ((n)*2700)

        # print(n, idx_start, idx_stop)

        df_snippet_raw_tmp3 = df_raw.iloc[idx_start: idx_stop].reset_index().drop('index', axis=1)
		
        df_snippet_scaled_tmp3 = get_df_scaled_from_df_raw(  df_snippet_raw_tmp3 )

        idx_counter += 1

        df_snippet_scaled_tmp3.to_csv( os.path.join( dirname, fname_json.split('/')[-1].replace('.json', '_df_scaled_snippet_idx{}.csv'.format(idx_counter)) ) )

	### END ###

    if len(df_snippet_raw_tmp2) > 0:
        avail_snippets_at_end = int( ( num_frames_json - df_snippet_raw_tmp2.index.max() ) / 2700 )
        print( 'Available snippets at end: {}'.format( avail_snippets_at_end ) )

        for n in range(avail_snippets_at_end):
	
            idx_start = df_snippet_raw_tmp2.index.max() + ((n)*2700) + 1
            idx_stop  = df_snippet_raw_tmp2.index.max() + ((n+1)*2700) + 1
	
            # print(n, idx_start, idx_stop)
	
            df_snippet_raw_tmp3 = df_raw.iloc[idx_start: idx_stop].reset_index().drop('index', axis=1)
    		
            df_snippet_scaled_tmp3 = get_df_scaled_from_df_raw(  df_snippet_raw_tmp3 )

            idx_counter += 1
	
            df_snippet_scaled_tmp3.to_csv( os.path.join( dirname, fname_json.split('/')[-1].replace('.json', '_df_scaled_snippet_idx{}.csv'.format(idx_counter)) ) )

    else:
        avail_snippets_at_end = int( ( num_frames_json - df_snippet_raw_tmp1.index.max() ) / 2700 )
        print( 'Available snippets at end: {}'.format( avail_snippets_at_end ) )

        for n in range(avail_snippets_at_end):
	
            idx_start = df_snippet_raw_tmp1.index.max() + ((n)*2700) + 1
            idx_stop  = df_snippet_raw_tmp1.index.max() + ((n+1)*2700) + 1
	
            # print(n, idx_start, idx_stop)
	
            df_snippet_raw_tmp3 = df_raw.iloc[idx_start: idx_stop].reset_index().drop('index', axis=1)
			
            df_snippet_scaled_tmp3 = get_df_scaled_from_df_raw(  df_snippet_raw_tmp3 )
	
            idx_counter += 1
	
            df_snippet_scaled_tmp3.to_csv( os.path.join( dirname, fname_json.split('/')[-1].replace('.json', '_df_scaled_snippet_idx{}.csv'.format(idx_counter)) ) )

    print('****')