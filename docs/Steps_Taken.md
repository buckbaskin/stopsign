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
images. This was done by matching keypoint locations with an octagonal contour
specified by the user clicking and dragging. All points could be marked as not
a stopsign by entering a keypress. Additionally, all frames before the
stopsign was revealed were marked as entirely not stopsign.
