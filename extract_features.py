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
    'left_ankle', 'right_ankle',
)


# In[15]:


lms_using = (
    'left_wrist', 'right_wrist',
    'left_ankle', 'right_ankle',
    'left_shoulder', 'right_shoulder',
    'left_hip', 'right_hip',
    'left_elbow', 'right_elbow',
    'left_knee', 'right_knee',
)


lm_pairs = [
    
#     # head/face orientation - fencing reflex
#     ['nose', 'left_shoulder'],
#     ['nose', 'right_shoulder'],

    # touching face?
    ['left_wrist', 'nose'], 
    ['right_wrist', 'nose'],

    # wrists/ankles touching?
    ['left_wrist', 'right_wrist'], 
    ['left_ankle', 'right_ankle'],     

    # upper limb mobility
    ['left_shoulder', 'left_wrist'],
    ['right_shoulder', 'right_wrist'],

    # lower limb mobility
    ['left_hip', 'left_ankle'],
    ['right_hip', 'right_ankle'],

    # same-side symmetry
    ['left_wrist','left_ankle'],
    ['right_wrist','right_ankle'],
    ['left_elbow', 'left_knee'],
    ['right_elbow', 'right_knee'],
    
#     # torso movement
#     ['left_shoulder', 'right_hip'],
#     ['right_shoulder', 'left_hip'],

]

lm_triples = [

    # upper limb mobility
    ['left_hip', 'left_shoulder', 'left_elbow'],
    ['right_hip', 'right_shoulder', 'right_elbow'],
    ['left_shoulder','left_elbow','left_wrist'],
    ['right_shoulder','right_elbow','right_wrist'],
    
    # lower limb mobility
    ['left_shoulder', 'left_hip', 'left_knee'],
    ['right_shoulder', 'right_hip', 'right_knee'],
    ['left_hip','left_knee','left_ankle'],
    ['right_hip','right_knee','right_ankle'],
    
]

lm_pairs_polar = [
    # anchor, distal

    # UPPER BODY
    ['nose','left_wrist'],
    ['nose','right_wrist'],

    ['left_shoulder', 'left_elbow'],
    ['left_shoulder', 'left_wrist'],

    ['right_shoulder', 'right_elbow'],
    ['right_shoulder', 'right_wrist'],

    # LOWER BODY    
    ['left_hip', 'left_knee'],
    ['left_hip', 'left_ankle'],

    ['right_hip', 'right_knee'],
    ['right_hip', 'right_ankle'],
]

##### #####

def calculate_angle(
        proximal : float, 
        medial   : float, 
        distal   : float
    ):
    """
    Calculate the angle given 3 keypoints (proximal, medial, and distal).
    """
    
    radians = np.arctan2(distal[1]-medial[1], distal[0]-medial[0]) - np.arctan2(proximal[1]-medial[1], proximal[0]-medial[0])
    
    angle = np.abs((radians * 180.0) / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle

    return angle

##### #####

def get_df_polar_coords(df, anchor_lm, distal_lm):
    
    df_anchor_points = df[ [col for col in df.columns if anchor_lm in col] ]

    centroid_anchor  = df_anchor_points.mean(axis=0).values
    
    df_distal_points = df[ [col for col in df.columns if distal_lm in col] ]
    
    polar_coords = []

    if 'left' in distal_lm:
    
        for _, row in df_distal_points.iterrows():
            
            x,y,z = row.values
            
            dx = x - centroid_anchor[0]
            dy = y - centroid_anchor[1]
            dz = z - centroid_anchor[2]
            
            r_2d = math.hypot( dx, dy )
            r_3d = math.hypot( dx, dy, dz )
            
            theta_rad = np.arctan2(dy, dx) # Angle from the positive x-axis
            theta_deg = math.degrees(theta_rad)
        
            polar_coords.append( (r_2d, r_3d, theta_rad) )
            # polar_coords.append( (r_2d, r_3d, theta_deg) )

    else:
    
        centroid_anchor[0] = -1.0 * centroid_anchor[0]
        for _, row in df_distal_points.iterrows():
            
            x,y,z = row.values
            x *= -1.0
            
            dx = x - centroid_anchor[0]
            dy = y - centroid_anchor[1]
            dz = z - centroid_anchor[2]
            
            r_2d = math.hypot( dx, dy )
            r_3d = math.hypot( dx, dy, dz )
            
            theta_rad = np.arctan2(dy, dx) # Angle from the positive x-axis
            theta_deg = math.degrees(theta_rad)
        
            polar_coords.append( (r_2d, r_3d, theta_rad) )
            # polar_coords.append( (r_2d, r_3d, theta_deg) )
    
    df_polar_coords = pd.DataFrame(polar_coords, columns=['r_2d','r_3d','theta_rad'])
    # df_polar_coords = pd.DataFrame(polar_coords, columns=['r_2d','r_3d','theta_deg'])

    return df_polar_coords

##### #####

def get_lr_diff_features(df_tmp):

    df_tmp_left  = df_tmp[[col for col in df_tmp.columns if ('left' in col) & ('right' not in col)]]
    df_tmp_right = df_tmp[[col for col in df_tmp.columns if ('right' in col) & ('left' not in col)]]
    
    df_tmp_LRDiff = pd.DataFrame( df_tmp_left.values - df_tmp_right.values )

    cols_LRDiff = [ col.replace('left_','') for col in df_tmp_left.columns if 'left' in col]
    df_tmp_LRDiff.columns = cols_LRDiff

    return df_tmp_LRDiff
    
##### #####

def get_points_for_lm( df_tmp, lm, dims=3):
    cols = [f'{lm}_{coord}' for coord in ['x','y','z'][:dims]]
    points = df_tmp[cols]
    return points.values
    
    
# def get_polar_coords( lm, points, centroid=(0.0, 0.0, 0.0) ):
# 
#     polar_coords = []
#     
#     if 'left' in lm:
#         for x, y, z in points:
#             dx = x - centroid[0]
#             dy = y - centroid[1]
#             dz = z - centroid[2]
#     
#             r_2d = math.hypot( dx, dy )
#             r_3d = math.hypot( dx, dy, dz )
#             
#             theta_rad = np.arctan2(dy, dx) # Angle from the positive x-axis
#             theta_deg = math.degrees(theta_rad)
# 
#             polar_coords.append( (r_2d, r_3d, theta_deg) )
#     else:
#         centroid = (-1.0*centroid[0], centroid[1], centroid[2])
#         for x, y, z in points:
#             x = -1.0*x
#             dx = x - centroid[0]
#             dy = y - centroid[1]
#             dz = z - centroid[2]
# 
#             r_2d = math.hypot( dx, dy )
#             r_3d = math.hypot( dx, dy, dz )
#             
#             theta_rad = np.arctan2(dy, dx) # Angle from the positive x-axis
#             theta_deg = math.degrees(theta_rad)
#             
#             polar_coords.append( (r_2d, r_3d, theta_deg) )
#     
#     return polar_coords


def get_dfs(fname_df_snippet_scaled):
    
    df_snippet_scaled = pd.read_csv(
        fname_df_snippet_scaled, 
        index_col=0
    )

    df_pos_3d = df_snippet_scaled[ [col for col in df_snippet_scaled.columns if '_'.join( col.split('_')[:-1] ) in lms_using] ]
    df_pos_2d = df_pos_3d[ [col for col in df_pos_3d.columns if '_z' not in col] ]
    
    # FRAMEWISE DISTANCES
    lm2dists_framewise_2d = {}
    lm2dists_framewise_3d = {}
    
    for lm in lms_using:
    
        cols = [f'{lm}_{coord}' for coord in ['x','y','z']]
        
        dists_framewise_2d = [0.0]
        dists_framewise_3d = [0.0]
        
        for i in range(1,len(df_snippet_scaled)):
            
            prev_pnt = df_snippet_scaled.iloc[i-1][cols].values 
            curr_pnt = df_snippet_scaled.iloc[i][cols].values
            
            dists_framewise_2d.append( np.linalg.norm( curr_pnt[:2] - prev_pnt[:2] ) )
            dists_framewise_3d.append( np.linalg.norm( curr_pnt - prev_pnt ) )
        
        lm2dists_framewise_2d[f'{lm}_2d'] = dists_framewise_2d
        lm2dists_framewise_3d[f'{lm}_3d'] = dists_framewise_3d

    df_lm2dists_framewise_2d = pd.DataFrame(lm2dists_framewise_2d)
    df_lm2dists_framewise_3d = pd.DataFrame(lm2dists_framewise_3d)

    # CENTROID DISTANCES
    lm2dists_centroid_2d = {}
    lm2dists_centroid_3d = {}
    
    for lm in lms_using:
    
        cols = [f'{lm}_{coord}' for coord in ['x','y','z']]
        centroid = df_snippet_scaled[cols].mean()
        
        dists_2d = []
        dists_3d = []
        
        for i in range(len(df_snippet_scaled)):
    
            curr_pnt = df_snippet_scaled.iloc[i][cols].values
            
            dists_2d.append( np.linalg.norm( curr_pnt[:2] - centroid[:2] ) )
            dists_3d.append( np.linalg.norm( curr_pnt - centroid ) )
        
        lm2dists_centroid_2d[f'{lm}_2d'] = dists_2d
        lm2dists_centroid_3d[f'{lm}_3d'] = dists_3d

    df_lm2dists_centroid_2d = pd.DataFrame(lm2dists_centroid_2d)
    df_lm2dists_centroid_3d = pd.DataFrame(lm2dists_centroid_3d)

    # JOINT-JOINT DISTANCES
    lm_pair2dists_2d = {}
    lm_pair2dists_3d = {}
    
    for lm_pair in lm_pairs:
        lm1, lm2 = lm_pair
    
        cols1 = [f'{lm1}_{coord}' for coord in ['x','y','z']]
        cols2 = [f'{lm2}_{coord}' for coord in ['x','y','z']]    
            
        dists_2d = []
        dists_3d = []
        
        for i in range( len(df_snippet_scaled) ):
            
            pnt1 = df_snippet_scaled.iloc[i][cols1].values
            pnt2 = df_snippet_scaled.iloc[i][cols2].values
    
            dists_2d.append( np.linalg.norm( pnt1[:2] - pnt2[:2] ) )
            dists_3d.append( np.linalg.norm( pnt1 - pnt2 ) )
    
        lm_pair2dists_2d[f'{lm1}_to_{lm2}_2d'] = dists_2d
        lm_pair2dists_3d[f'{lm1}_to_{lm2}_3d'] = dists_3d

    df_lm_pair2dists_2d = pd.DataFrame(lm_pair2dists_2d)    
    df_lm_pair2dists_3d = pd.DataFrame(lm_pair2dists_3d)

    # JOINT ANGLES
    lm_triple2angles = {}

    for lm_triple in lm_triples:
        
        lm1, lm2, lm3 = lm_triple
        cols1 = [f'{lm1}_{coord}' for coord in ['x','y']]
        cols2 = [f'{lm2}_{coord}' for coord in ['x','y']]    
        cols3 = [f'{lm3}_{coord}' for coord in ['x','y']]
    
        angles = []
        if 'left' in lm1:
            for i in range( len(df_snippet_scaled) ):

                pnt1 = df_snippet_scaled.iloc[i][cols1].values

                pnt2 = df_snippet_scaled.iloc[i][cols2].values

                pnt3 = df_snippet_scaled.iloc[i][cols3].values

                angle = calculate_angle( pnt1, pnt2, pnt3 )
                angles.append(angle)
        else:
            for i in range( len(df_snippet_scaled) ):

                pnt1 = df_snippet_scaled.iloc[i][cols1].values
                pnt1[0] = pnt1[0] * -1.0

                pnt2 = df_snippet_scaled.iloc[i][cols2].values
                pnt2[0] = pnt2[0] * -1.0

                pnt3 = df_snippet_scaled.iloc[i][cols3].values
                pnt3[0] = pnt3[0] * -1.0

                angle = calculate_angle( pnt1, pnt2, pnt3 )
                angles.append(angle)    

        lm_triple2angles[f'{lm1}_to_{lm2}_to_{lm3}'] = angles


    df_lm_triple2angles = pd.DataFrame(lm_triple2angles)

#     for col in df_lm_triple2angles.columns:
#         df_lm_triple2angles[col] = np.unwrap( df_lm_triple2angles[col] )


#     # POLAR COORDINATES V1
#     lm2polar_coords = {}
#     for lm in lms_using:
#         points_lm = get_points_for_lm( df_snippet_scaled, lm )
#         centroid_lm = points_lm.mean(axis=0)
#         polar_coords = get_polar_coords( lm, points_lm, centroid_lm[2] )
#         df_polar_coords = pd.DataFrame(polar_coords, columns=[f'{lm}_r_2d', f'{lm}_r_3d', f'{lm}_theta_deg'])
#         lm2polar_coords[lm] = df_polar_coords
#     df_polar_coords = pd.concat( list(lm2polar_coords.values()), axis=1 )


    dfs_polar_coords = []
    for lm_pair_polar in tqdm( lm_pairs_polar ):
        anchor_lm, distal_lm = lm_pair_polar
        lm_pair_polar_str = '{}_to_{}'.format( anchor_lm, distal_lm )
        df_polar_coords_tmp = get_df_polar_coords( df_snippet_scaled, anchor_lm, distal_lm )
        df_polar_coords_tmp.columns = ['{}_{}'.format(lm_pair_polar_str, col) for col in df_polar_coords_tmp.columns]
        dfs_polar_coords.append( df_polar_coords_tmp )

    df_polar_coords = pd.concat( dfs_polar_coords, axis=1 )
    
#     # LRDiff
#     df_lm2dists_framewise_2d_LRDiff = get_lr_diff_features(df_lm2dists_framewise_2d)
#     df_lm2dists_framewise_3d_LRDiff = get_lr_diff_features(df_lm2dists_framewise_3d)
#     
#     df_lm2dists_centroid_2d_LRDiff = get_lr_diff_features(df_lm2dists_centroid_2d)
#     df_lm2dists_centroid_3d_LRDiff = get_lr_diff_features(df_lm2dists_centroid_3d)
# 
#     df_lm_pair2dists_2d_LRDiff = get_lr_diff_features(df_lm_pair2dists_2d)
#     df_lm_pair2dists_3d_LRDiff = get_lr_diff_features(df_lm_pair2dists_3d)
# 
#     df_lm_triple2angles_LRDiff = get_lr_diff_features(df_lm_triple2angles)
# 
# 	# POLAR COORDINATES LR DIFF V1
#     df_polar_coords_LRDiff = get_lr_diff_features(df_polar_coords)
# 	
# 	# POLAR COORDINATES LR DIFF V2
#     df_polar_coords_LRDiff = get_lr_diff_features(df_polar_coords)

    
    return  df_pos_2d, df_pos_3d, \
            df_lm2dists_framewise_2d, df_lm2dists_framewise_3d, \
            df_lm2dists_centroid_2d, df_lm2dists_centroid_3d, \
            df_lm_pair2dists_2d, df_lm_pair2dists_3d, \
            df_lm_triple2angles, \
            df_polar_coords,
#             df_lm2dists_framewise_2d_LRDiff, df_lm2dists_framewise_3d_LRDiff, \
#             df_lm2dists_centroid_2d_LRDiff, df_lm2dists_centroid_3d_LRDiff, \
#             df_lm_pair2dists_2d_LRDiff, df_lm_pair2dists_3d_LRDiff, \
#             df_lm_triple2angles_LRDiff, \
#             df_polar_coords_LRDiff


    
df_descs = [
    'pos_2d', 'pos_3d',
    'lm2dists_framewise_2d', 'lm2dists_framewise_3d',
    'lm2dists_centroid_2d', 'lm2dists_centroid_3d',
    'lm_pair2dists_2d', 'lm_pair2dists_3d',
    'lm_triple2angles',
    'polar_coords',
#     'lm2dists_framewise_2d_LRDiff', 'lm2dists_framewise_3d_LRDiff',
#     'lm2dists_centroid_2d_LRDiff', 'lm2dists_centroid_3d_LRDiff',
#     'lm_pair2dists_2d_LRDiff', 'lm_pair2dists_3d_LRDiff',
#     'lm_triple2angles_LRDiff',
#     'polar_coords_LRDiff'
]



df_meta = pd.read_csv('./sensor_fusion/df_meta_withJSON_withTimes_goodDetections_min2700.csv', index_col=0)

for _, row in df_meta.iloc[ (arg_idx - 1) * 8 : (arg_idx ) * 8 ].iterrows():

    fname_json = row['fname_json'].replace('../../','./')
    dirname    = row['dirname_json'].replace('../../','./')
    
    fnames_dfs_snippet_scaled = ast.literal_eval( row['fnames_scaled_snippets'][:] )

    print(fnames_dfs_snippet_scaled)
    
    fnames_dfs = [os.path.join(dirname, fname) for fname in fnames_dfs_snippet_scaled]
               
    for fname_df_snippet_scaled in fnames_dfs:
    
        dfs = get_dfs(fname_df_snippet_scaled)
        
        for df_tmp, df_desc in zip(dfs, df_descs):
	                
            save_path = fname_df_snippet_scaled.replace('.csv', '_{}.csv'.format(df_desc) )

            df_tmp.columns = [ '{}__{}'.format(df_desc, col) for col in df_tmp.columns ]
	        
            df_tmp.to_csv( save_path )	
	