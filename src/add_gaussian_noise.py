#!/usr/bin/env python
import cv2
import numpy as np
import rospkg

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('stopsign')

IN_IMAGE = '%s/data/002_original_images/%s' % (pkg_path, 'frame%04d.jpg')

OUT_IMAGE = '%s/data/006_gaussian_images/%s' % (pkg_path, 'frame%04d.jpg')

start_image_id = 1820
end_image_id = 2189

def get_image(image_id):
    filename = IN_IMAGE % (image_id,)
    return cv2.imread(filename, cv2.IMREAD_COLOR)

def set_image(img, image_id):
    filename = OUT_IMAGE % (image_id,)
    cv2.imwrite(filename, img)


def add_gaussian(og_img):
    mean = 0
    stdev = 15
    noise = np.random.normal(mean, stdev, og_img.shape).reshape(og_img.shape)
    return (og_img + noise).astype('uint8')

if __name__ == '__main__':
    np.random.seed(12345)
    for image_id in range(start_image_id, end_image_id):
        og_img = get_image(image_id)
        noise_img = add_gaussian(og_img)

        cv2.imshow('original', og_img)
        cv2.waitKey(0)
        cv2.imshow('noisey', noise_img)
        cv2.waitKey(0)
        set_image(noise_img, image_id)
        break