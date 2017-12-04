#!/usr/bin/env python
import cv2
import numpy as np
import pandas as pd
import rospkg

from itertools import starmap
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('stopsign')

IMAGE_RATE = 11 # hz

EXACT_FILE_IN = '%s/data/003_manual_labels/exact.csv' % (pkg_path,)
ALL_FILE_IN = '%s/data/003_manual_labels/all.csv' % (pkg_path,)

EXACT_FILE_OUT = '%s/data/004_knn_labels/exact.csv' % (pkg_path,)
ALL_FILE_OUT = '%s/data/004_knn_labels/all.csv' % (pkg_path,)

start_image_id = 0
end_image_id = 2189

IMAGE_BASE_STRING = '%s/data/002_original_images/%s' % (pkg_path, 'frame%04d.jpg')

col_names = []
col_names.append('class'.ljust(7))
for i in range(32):
    col_names.append('descr%02d' % (i,))

kp_names = ['x', 'y', 'size', 'angle', 'respons', 'octave', 'classid']
kp_names = [x.ljust(7) for x in kp_names]

# cv2.KeyPoint([x, y, _size[, _angle[, _response[, _octave[, _class_id]]]]])

def get_image(image_id):
    filename = IMAGE_BASE_STRING % (image_id,)
    return cv2.imread(filename, cv2.IMREAD_COLOR)

def render_and_capture(image_id, stop_df, confused_df, nostop_df):
    img = get_image(image_id)

    skps = list(starmap(cv2.KeyPoint, stop_df[kp_names].values.tolist()))
    if len(skps) > 0:
        img = cv2.drawKeypoints(
                    img,
                    skps,
                    color=(255,0,0),
                    flags=0)
    else:
        print('No stopsign points')
    noskps = list(starmap(cv2.KeyPoint, nostop_df[kp_names].values.tolist()))
    if len(noskps) > 0:
        img = cv2.drawKeypoints(
                    img,
                    noskps,
                    color=(255,255,0),
                    flags=0)
    else:
        print('No not-stopsign points')
    confusedkps = list(starmap(cv2.KeyPoint, confused_df[kp_names].values.tolist()))
    if len(confusedkps) > 0:
        img = cv2.drawKeypoints(
                    img,
                    confusedkps,
                    color=(0,255,0),
                    flags=0)
    else:
        print('No confused points')
    cv2.imshow('review', img)
    # cv2.setMouseCallback('preview', click_and_crop)
    val = cv2.waitKey(10 * 1000)
    return val % 256

def ask_user(image_id, stop_df, confused_df, nostop_df, recursion=0):
    '''
    Render an image with the given keypoints out of the dataframe. 

    If the user hits (a)ccept, accept the current stop and non-stop df. Invalid if
    there are confused df. Return
    If the user hits (n)ostop, write all keypoints as no stopsign, Return

    If the user hits (s), expect stopsign somewhere. Before or after, look for
    stopsign region selection. Click and drag region. Using keyframe location
    from df, reselect points into stop and nostop regions.

    Redraw, then restart ask user process with recursion incremented
    '''
    if recursion >= 10:
        nostop_df.append(confused_df)
        return stop_df, nostop_df

    if image_id < 1785:
        nostop_df.append(stop_df)
        nostop_df.append(confused_df)
        stop_df.drop(stop_df.index, inplace=True)
        return stop_df, nostop_df

    user_key = render_and_capture(image_id, stop_df, confused_df, nostop_df)

    if user_key == ord('a'):
        if len(confused_df) == 0:
            return stop_df, nostop_df
    if user_key == ord('n') :
        nostop_df.append(stop_df)
        nostop_df.append(confused_df)
        stop_df.drop(stop_df.index, inplace=True)
        return stop_df, nostop_df
    if user_key == ord('s') or user_key == ord('a'):
        # polygon things
        raise NotImplementedError() # TODO(buckbaskin): fix this
        return stop_df, nostop_df
    else:
        raise ValueError('Please use a valid key (a, s, n)')


# Process
if __name__ == '__main__':
    # learn from exact file

    exact_df = pd.read_csv(EXACT_FILE_IN, header=0)
    # class
    exact_df[col_names[0]] = exact_df[col_names[0]].apply(lambda cls: cls / 1000.0)
    # response
    exact_df[kp_names[4]] = exact_df[kp_names[4]].apply(lambda res: res / 100000000.0)
    # angle
    exact_df[kp_names[3]] = exact_df[kp_names[3]].apply(lambda ang: ang / 1000.0)
    print(exact_df.describe())

    y = exact_df[col_names[:1]].as_matrix()
    X = exact_df[col_names[1:]].as_matrix()

    print('Begin Learning')
    neigh = KNeighborsRegressor(n_neighbors=5, weights='distance', algorithm='ball_tree')
    neigh.fit(X, y)
    print('Done with Initial Fit')
    print('Begin reading in all data from CSV')

    all_df = pd.read_csv(ALL_FILE_IN, header=0)

    print('Read in ALL data')

    new_all_df = pd.DataFrame()

    print('Iterate through Images')
    # iterate through all file by image and reclassify
    for image_id in range(start_image_id, end_image_id):
        if image_id % 20 == 0:
            print('image %d' % (image_id,))
        imagedf = all_df.loc[all_df['imageid'] == image_id]

        # For each keypoint in the image:
        #   use knn to classify it again
        new_y = neigh.predict(imagedf[col_names[1:]].as_matrix())
        old_y = imagedf[col_names[:1]].as_matrix()
        delta_y = np.absolute(new_y - old_y)
        
        imagedf['delta_y'] = delta_y
        confused_df = imagedf.loc[imagedf['delta_y'] > 0.2]
        confused_kp = confused_df[kp_names].as_matrix() # yellow
        stable_df = imagedf.loc[imagedf['delta_y'] <= 0.2]
        stop_df = stable_df.loc[stable_df[col_names[0]] > 0.5]
        # stop_kp = stop_df[kp_names].as_matrix() # blue
        nostop_df = stable_df.loc[stable_df[col_names[0]] <= 0.5]
        # nostop_kp = nostop_df[kp_names].as_matrix() # green

        # render image w/ colorized keypoints
        # s to accept, click-drag + n to rerender octogon and eventually select
        # yellow points highlights when it shouldn't be a quick skip
        new_stop_df, new_nostop_df = ask_user(image_id, stop_df, confused_df, nostop_df)

        # put rows back
        new_all_df.append(new_stop_df)
        new_all_df.append(new_nostop_df)

    # write the all-good list to disk
    new_all_df.to_csv(ALL_FILE_OUT)
