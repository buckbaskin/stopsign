#!/usr/bin/env python3

import pandas as pd

pkg_path = '/home/buck/ros_ws/src/stopsign'

IMAGE_RATE = 30 # hz

IN_FILE = '%s/data/017_the_500/all_500.csv' % (pkg_path,)
OUT_FILE = '%s/data/017_the_500/clean_500.csv' % (pkg_path,)

if __name__ == '__main__':
    df = pd.read_csv(IN_FILE, header=0)
    print('done reading. apply')
    df['class  '] = df['class  '].apply(lambda cls: cls / 1000.0)
    df['angle  '] = df['angle  '].apply(lambda ang: ang / 1000.0)
    df['respons'] = df['respons'].apply(lambda res: res / 100000000.0)
    print('done applying. write')
    df.to_csv(OUT_FILE)
