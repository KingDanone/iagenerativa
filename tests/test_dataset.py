import numpy as np
import torch

from src.dataset import LMDataset


def test_lm_shift():
    toks = np.arange(20, dtype=np.uint16)
    ds = LMDataset(toks, block_size=4)
    x, y = ds[0]
    assert x.tolist() == [0, 1, 2, 3]
    assert y.tolist() == [1, 2, 3, 4]
    x2, y2 = ds[5]
    assert (y2[:-1] == x2[1:]).all()


def test_shapes_and_len():
    toks = np.arange(100, dtype=np.uint16)
    ds = LMDataset(toks, block_size=16)
    assert len(ds) == 100 - 16 - 1
    x, y = ds[10]
    assert x.shape == (16,) and y.shape == (16,)
    assert x.dtype == torch.int64
