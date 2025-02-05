# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 16:05:05 2024

@author: Simon
"""
import sys; sys.path.append('../..')
from meg_utils import decoding


import os
import unittest
import numpy as np
import scipy
from scipy import io
from tqdm import tqdm
from tdlm import plotting
import matplotlib.pyplot as plt
import tdlm

class TestDecoding(unittest.TestCase):

    def test_votingclassifier(self):
        pass
