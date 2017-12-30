#!/usr/bin/env python3

import pandas as pd

pkg_path = '/home/buck/ros_ws/src/stopsign'
START_FILE = '%s/data/015_visualize/clean_200.csv' % (pkg_path,)
POSITIVE_FILE = '%s/data/015_visualize/positive_200.csv' % (pkg_path,)
NEGATIVE_FILE = '%s/data/015_visualize/negative_200.csv' % (pkg_path,)

klass = ['class'.ljust(7)]

df = pd.read_csv(START_FILE, header=0)

neg, pos = df.groupby(by=klass)

nclass_, neg = neg
pclass_, pos = pos

pos.to_csv(POSITIVE_FILE)
neg.to_csv(NEGATIVE_FILE)