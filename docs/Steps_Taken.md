# Process Documentation

## Preprocessing

Raw File: `raw/single_i.bag.orig.active`

The file was then `reindex`ed using the following command:

`rosbag reindex raw/single_i.bag.active`

From there the reindexed bag was moved to the correct folder and `fix`ed using the following command:

`rosbag fix fixed_bags/single_i.bag.active fixed_bags/single_i_fixed.bag`

From there, `jpeg` images were extracted using the launch file:

`launch/convert_bag_to_jpeg.launch`

This writes all of the images published to the camera topic to the
`original_images/` folder.

## Manual Labeling

Using the following program, labels were estimated for all keypoints in all
images. This is a very noisy process, but there a number of exact images within
the dataset. This process generates 2 CSV files. The first is the exact data.
For each image actually previewed, with stopsigns selected if needed, the exact
label for each keypoint and descriptor is stored in the CSV file. The second is
the approximate dataset. This is much larger. The exact information is stored.
Additionally, the intervening images are stored as if every keypoint matches the
class label of the previous exact image (stopsign or not). These are indicated
with a lower confidence (.25 or .75 respectively).
