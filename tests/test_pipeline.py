#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
import mne
from meg_utils.pipeline import FilterStep


class TestFilterStep(unittest.TestCase):
    def test_filter_epochs(self):
        """FilterStep accepts Epochs and returns Epochs."""
        info = mne.create_info(3, 100., 'grad')
        epochs = mne.EpochsArray(np.random.randn(5, 3, 100), info, verbose=False)
        out = FilterStep(1, 20, verbose=False).transform(epochs)
        self.assertIsInstance(out, mne.BaseEpochs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
