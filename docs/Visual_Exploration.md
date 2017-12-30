# Visual Exploration

This document outlines the steps that I'm taking to explore the data. For now at least, the document will largely be sequential notes and will be cleaned up later.

## Obtaining and Cleaning Data

The data comes from 200 random images selected from the new video dataset with stopsigns. Stopsigns were manually labelled and constituent keypoints were identified using an octagonal polygon. From there, the dataset was split into 12861 positive examples and 1003498 negative examples. This split was done with `split_pos_neg.py`.

The data represents 256 bitwise comparisons in the ORB descriptor features (32 x 8 bits per byte). Each original value is broken out into 8 columns in `bytes_to_bits.py`.