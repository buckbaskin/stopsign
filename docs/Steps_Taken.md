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
