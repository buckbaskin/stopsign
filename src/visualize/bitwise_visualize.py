#!/usr/bin/env python3

import pandas as pd

import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt

pkg_path = '/home/buck/ros_ws/src/stopsign'
START_FILE = '%s/data/015_visualize/positive_bits_200.csv' % (pkg_path,)

bit_label = 'd%02db%01d'

descriptors = []
for i in range(32):
    for bit_index in range(0, 8):
        descriptors.append('d%02db%01d' % (i, bit_index,))

klass = ['class']

df = pd.read_csv(START_FILE, header=0)

# print(df.describe())

correlation_matrix = df.corr()
plt.matshow(correlation_matrix)
plt.savefig('positive_corr_matrix.png')

print(correlation_matrix.describe())
