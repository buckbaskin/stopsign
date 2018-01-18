#!/usr/bin/env python3

import pandas as pd

pkg_path = '/home/buck/ros_ws/src/stopsign'
START_FILE = '%s/data/017_the_500/clean_500.csv' % (pkg_path,)
POSITIVE_FILE = '%s/data/017_the_500/positive_500.csv' % (pkg_path,)
NEGATIVE_FILE = '%s/data/017_the_500/negative_500_%s.csv' % (pkg_path, '%d',)

klass = ['class'.ljust(7)]

df = pd.read_csv(START_FILE, header=0)
print('done reading')

neg, pos = df.groupby(by=klass)

nclass_, neg = neg
pclass_, pos = pos

print('done groupby')

pos.to_csv(POSITIVE_FILE, index=False)
print('positive written')

for i in range(0, 5):
    little_neg = neg.sample(n=len(pos))
    little_neg.to_csv(NEGATIVE_FILE % (i,), index=False)
    print('negative %d written' % (i,))
