#!/usr/bin/env python
import cv2
import numpy as np
import pandas as pd
import rospkg

from itertools import starmap
# from matplotlib import pyplot as plt
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

minx = -10000
miny = -10000
maxx = 10000
maxy = 10000
user_contour = False
contour = []

def rebuild_contour():
    global minx, miny, maxx, maxy
    x1 = minx
    x2 = int(2.0/3 * minx + 1.0/3 * maxx)
    x3 = int(1.0/3 * minx + 2.0/3 * maxx)
    x4 = maxx
    y1 = miny
    y2 = int(2.0/3 * miny + 1.0/3 * maxy)
    y3 = int(1.0/3 * miny + 2.0/3 * maxy)
    y4 = maxy
    global contour
    contour = np.array([[x2, y1], [x3, y1], [x4, y2], [x4, y3],
                        [x3, y4], [x2, y4], [x1, y3], [x1, y2]], np.int32)

rebuild_contour()

def click_and_crop(event, x, y, flags, param):
    global minx, miny, maxx, maxy, user_contour
    user_contour = True
    if event == cv2.EVENT_LBUTTONDOWN:
        minx = x
        miny = y
        print('min crop')
    elif event == cv2.EVENT_LBUTTONUP:
        maxx = x
        maxy = y
        print('max crop')
    rebuild_contour()

def get_image(image_id):
    filename = IMAGE_BASE_STRING % (image_id,)
    return cv2.imread(filename, cv2.IMREAD_COLOR)

def render_and_capture(image_id, stop_df, confused_df, nostop_df):
    img = get_image(image_id)

    try:
        skps = list(starmap(cv2.KeyPoint, stop_df[kp_names].values.tolist()))
        if len(skps) > 0:
            img = cv2.drawKeypoints(
                        img,
                        skps,
                        color=(255,0,0),
                        flags=0)
        else:
            print('No stopsign points')
    except KeyError:
        print('error building skps')

    try:
        noskps = list(starmap(cv2.KeyPoint, nostop_df[kp_names].values.tolist()))
        if len(noskps) > 0:
            img = cv2.drawKeypoints(
                        img,
                        noskps,
                        color=(255,255,0),
                        flags=0)
        else:
            print('No not-stopsign points')
    except KeyError:
        print('error building noskps')

    try:
        confusedkps = list(starmap(cv2.KeyPoint, confused_df[kp_names].values.tolist()))
        if len(confusedkps) > 0:
            img = cv2.drawKeypoints(
                        img,
                        confusedkps,
                        color=(0,255,0),
                        flags=0)
        else:
            print('No confused points')
    except KeyError:
        print('error building confusedkps')

    cv2.imshow('review', img)
    cv2.setMouseCallback('review', click_and_crop)
    val = cv2.waitKey(0)
    return val % 256

def match_contour(row):
    return cv2.pointPolygonTest(
        contour,
        (row['x'.ljust(7)], row['y'.ljust(7)],),
        False)

def seek_user_classification(image_id, stop_df, confused_df, nostop_df):
    img = get_image(image_id)
    # force user selection of a polygon
    while not user_contour and False:
        pass

    kp = list(starmap(cv2.KeyPoint, stop_df[kp_names].values.tolist()))
    kp.extend(list(starmap(cv2.KeyPoint, confused_df[kp_names].values.tolist())))
    kp.extend(list(starmap(cv2.KeyPoint, nostop_df[kp_names].values.tolist())))

    # draw polygon
    print('s -> accept polyline as region\n\nOR\n')
    print('Use mouse to reselect the region')
    print('n -> refresh polyline as region')
    print('---')

    for i in range(20):
        imgur = cv2.drawKeypoints(
            img,
            filter(lambda x: cv2.pointPolygonTest(contour, x.pt, False) >= 0, kp),
            color=(0,255,0),
            flags=0)
        cv2.polylines(imgur, [contour], True, (79*i % 255, 0, 255))
        cv2.imshow('preview', imgur)
        cv2.setMouseCallback('preview', click_and_crop)
        val = cv2.waitKey(0)
        if val % 256 == ord('s'):
            break

    # I can get the contour and make a function that boolean values based on the
    #   inclusion of the contour based on x, y
    # TODO(return the partitioned dataframes based on the match_contour function)

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
        print('seek user classification')
        new_stop_df, new_nostop_df = seek_user_classification(image_id, stop_df, confused_df, nostop_df)

        print('rerendered image')
        val = render_and_capture(image_id, new_stop_df, pd.DataFrame(), new_nostop_df)

        import sys
        sys.exit(1)
        return new_stop_df, new_nostop_df
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
    #TODO fix this, change back to step 1
    for image_id in range(1790, end_image_id, 1):
        if image_id % 50 == 0 and image_id > 0:
            print('writing image up to %d' % (image_id - 1,))
            new_all_df.to_csv(ALL_FILE_OUT)
            print('done writing')
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
