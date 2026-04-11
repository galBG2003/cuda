import cupy as cp
import numpy as np

def get_module(data):
    try:
        return cp.get_array_module(data)
    except ImportError:
        return np