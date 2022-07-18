import pytest
import copy
import numpy as np

from hmrGC.dixon_imaging.helper import calculate_pdff_percent

def calculate_pdff_percent_test():
    W = np.array([100, 10])
    F = np.array([0, 20])
    S = np.array([0, 70])
    PDFF_silicone = calculate_pdff_percent(W, F, S)
    PDFF_no_silicone = calculate_pdff_percent(W, F)
    assert PDFF_silicone[0] == PDFF_no_silicone[0]
    assert PDFF_silicone[1] != PDFF_no_silicone[1]
