# New Theory of Operations

## The theory

If I can label images as stopsign or not stopsign, then I can try to learn which
features in the images correspond to stopsigns. I can then apply that learning
algorithm to identify stopsigns (and potentially other obstacles) in the vision
stream.

## The Process (In English)

Using the process described here (https://arxiv.org/pdf/1703.02391.pdf) I can
use a small dataset of handlabelled images and then use a noisy dataset to
extract further learning.

Label a handful of images, going through and identifying stopsign or not for
each keypoint (one image per second). In between these points, images will be
classified all keypoints as either stopsign or not based on previous overall
image classification. Learn from those images. Manually reclassify images
that do not match the training label as needed (image by image, keypoint by
keypoint). Relearn the dataset. Validate/Test the dataset against a reserve time
period with some stopsign data and some not stopsign data.

Potentially add gaussian noise to images later. Potentially add brightness,
reduce brightness, otherwise perturb images and extend the training dataset to
try to account for variations in lighting conditions.

## Processing Steps (In Bullet Points)
1. Reindex and fix rosbag
2. Extract all images/frames from rosbag (on topics ...)
3. Use OpenCV ORB to find keypoints and characterize keypoints for each image
4. Run a script to sample 1 image per second and label it with stopsign, no
stopsign. See script (...). It interpolates between images hand-labeled by
keeping the last label. This will get a general sense for the correct labels.
5. Build a dataset of feature vectors and labels, where the class is either
stopsign or not in the entire image. For each keypoint in the image, add the
keypoint, the feature vector the class information, the image id. 
6. Train a SVD to identify keypoints that are in stopsign images. This should
include lots of noise because not every keypoint will actually be on a stopsign,
and some total images may be mislabled.
7. Iterate through the keypoints data, grouped by image. Estimate confidence in
the image being a stopsign by having a certain percent of keypoints classified
as a stopsign. If the confidence is low for an image labeled stopsign, ask the
user. If the confidence is high for an image labeled not stopsign, ask the user.
This process can be repeated for different tolerances. Rewrite the keypoints
data to reflect user input as a new training set. This step identifies examples
that may need new labels while requiring fewer labels to be set by hand.
8. Now with better image labels, train a new SVD. Test against reserved images?

### Process Notes

Stopsign flips open between 70 and 75 sec into data, with some images that have
the stopsign hidden. The images are published at about 31hz on average. The bag
runs for about 119 seconds.
