# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/expressive_tacotron
'''

from __future__ import print_function

from utils import load_spectrograms
import os
from data_load import load_data
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

NUM_JOBS = 4

# Utility function
def f(fpath):
    fname, mel, mag = load_spectrograms(fpath)
    np.save("mels/{}".format(fname.replace("wav", "npy")), mel)
    np.save("mags/{}".format(fname.replace("wav", "npy")), mag)
    return None

# Load data
fpaths, _ = load_data() # list

# Creates folders
if not os.path.exists("mels"): os.mkdir("mels")
if not os.path.exists("mags"): os.mkdir("mags")

# Creates pool
p = Pool(NUM_JOBS)

total_files = len(fpaths)
with tqdm(total=total_files) as pbar:
	for _ in tqdm(p.imap_unordered(f, fpaths)):
		pbar.update()