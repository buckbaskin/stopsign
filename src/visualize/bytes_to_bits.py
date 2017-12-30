#!/usr/bin/env python3

import pandas as pd

from bitstring import BitArray

pkg_path = '/home/buck/ros_ws/src/stopsign'
START_FILE = '%s/data/015_visualize/negative_200.csv' % (pkg_path,)
BIT_FILE = '%s/data/015_visualize/negative_bits_200.csv' % (pkg_path,)

descriptors = []
for i in range(32):
    descriptors.append('descr%02d' % (i,))

klass = ['class'.ljust(7)]

print('read_csv')

df = pd.read_csv(START_FILE, header=0)

print('relabel df')

df['class'] = df['class  ']
df = df.drop(columns=['class  ',])
df = df.drop(columns=['angle  ', 'classid', 'octave ', 'x'.ljust(7), 'y'.ljust(7), 'respons', 'size   ', 'imageid'])

bit_label = 'd%02db%01d'
for desc_index, descriptor in enumerate(descriptors):
    for bit_index in range(0, 8):
        new_label = bit_label % (desc_index, bit_index,)
        df[new_label] = df[descriptor].apply(lambda x: (x // 2**bit_index) % 2)
    df = df.drop(columns=[descriptor])
    print('done with % 2d / 32' % (desc_index + 1,))

print('write to csv')

print(df.describe())
df.to_csv(BIT_FILE)