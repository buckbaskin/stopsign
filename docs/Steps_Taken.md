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

Using the `manually_annotate_images.py` program, labels were estimated for all keypoints in all
images. This was done by matching keypoint locations with an octagonal contour
specified by the user clicking and dragging. All points could be marked as not
a stopsign by entering a keypress. Additionally, all frames before the
stopsign was revealed were marked as entirely not stopsign.

## Classifying an Image

A keypoint descriptor classifier was trained based on the manual labeling and saved using `scikit-learn` and `joblib`'s model persistence. This was then loaded and run to classify an image by counting the number of stopsigns identified. This approach was far more performant that any possible second machine learning classifying the image based on the location and classification of all of the keypoints.

## Creating the Demo

Each image from the original video was loaded and classified offline. Each set of images was stitched together using ffmpeg to make demo videos based on each set of images (original, gaussian noise and labelled demo). During the running of the demo, the performance of the algorithm was shown to be entirely inadequate. Using the `scikit-learn` implementation to predict an entire image at once took more than 30 seconds per image in an application where the snowplow might need to check 30 images per second.
