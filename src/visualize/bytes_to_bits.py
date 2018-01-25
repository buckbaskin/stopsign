#!/usr/bin/env python3

import pandas as pd

from bitstring import BitArray

pkg_path = '/home/buck/ros_ws/src/stopsign'
POSITIVE_FILE = '%s/data/017_the_500/positive_500.csv' % (pkg_path,)
NEGATIVE_FILE = '%s/data/017_the_500/negative_500_%s.csv' % (pkg_path, '%d',)
BIT_FILE = '%s/data/017_the_500/%s' % (pkg_path, '%s_bits_500%s.csv',)

descriptors = []
for i in range(32):
    descriptors.append('descr%02d' % (i,))

klass = ['class'.ljust(7)]

all_files = [POSITIVE_FILE]
for i in range(5):
    all_files.append(NEGATIVE_FILE % (i,))

for index, START_FILE in enumerate(all_files):
    print('read_csv')
    print(START_FILE)

    df = pd.read_csv(START_FILE, header=0)

    print('relabel df %d' % (len(df),))

    df['class'] = df['class  ']
    df = df.drop(columns=['class  ',])
    df = df.drop(columns=['angle  ', 'classid', 'octave ', 'x'.ljust(7), 'y'.ljust(7), 'respons', 'size   ', 'imageid'])

    bit_label = 'd%02db%01d'
    for desc_index, descriptor in enumerate(descriptors):
        for bit_index in range(0, 8):
            new_label = bit_label % (desc_index, bit_index,)
            df[new_label] = df[descriptor].apply(lambda x: (x // 2**bit_index) % 2)
        df = df.drop(columns=[descriptor])
        if desc_index % 8 == 0:
            print('done with % 2d / 32' % (desc_index + 1,))

    print('write to csv')

    # print(df.describe())
    if index == 0:
        OUT_FILE = BIT_FILE % ('positive', '',)
    else:
        OUT_FILE = BIT_FILE % ('negative', '_%d' % (index - 1),)
    print(OUT_FILE)
    df.to_csv(OUT_FILE, index=False)
