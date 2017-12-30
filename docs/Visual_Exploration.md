# Visual Exploration

This document outlines the steps that I'm taking to explore the data. For now at least, the document will largely be sequential notes and will be cleaned up later.

## Obtaining and Cleaning Data

The data comes from 200 random images selected from the new video dataset with stopsigns. Stopsigns were manually labelled and constituent keypoints were identified using an octagonal polygon. From there, the dataset was split into 12861 positive examples and 1003498 negative examples. This split was done with `split_pos_neg.py`.

The data represents 256 bitwise comparisons in the ORB descriptor features (32 x 8 bits per byte). Each original value is broken out into 8 columns in `bytes_to_bits.py`.

## 1D "Visualizing" the distribution of bits

This step can be achieved by summarizing the bitwise files.
The mean of each bit's value is the percent of 1s in the dataset for that bit.

This does not visualize correlations, but it does offer some insight into the naive-bayes idea of how new vectors could be classified.
For each column, the probability of a 1 would be the mean and the probability of 0 would be (1-mean).
The probability of each positive or negative would be the multiplication of the probabilities.
The positive or negative class with a higher probability wins.
Most probabilities are close to 0.5, so many bits aren't very useful, but there is at least one positive bit with a mean of 0.381930 (d31b2) so some variation in probability could be expected.

## 2D Visualizing the correlation between bits

`pandas` has a .corr function that gives the correlation matrix :). And `matplotlib`/`pyplot` has a function to plot the correlation matrix :).

Now to just get that bytes->bits thing working for the much larger negative dataset.
This will likely require making ~80 smaller datasets.

## Making Smaller Datasets

`pandas` has a `sample` function that is used to subdivide the negative dataset into 5 different negative datasets of the same size as the positive dataset. This is effectively equivalent to random undersampling.

## Learning with boolean algorithms

- Boosted Decision Trees
- Logistic Regression