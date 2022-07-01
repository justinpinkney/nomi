import os
from pathlib import Path
import time
import pytest

from nomi.imdump import Dumper

import torch

def make_test_data(device):
    b, c, h, w = 10, 3, 256, 256
    dummy_data = torch.ones(b, c, h, w).to(device)
    return 255*dummy_data.permute(0,2,3,1)

device_params = pytest.mark.parametrize(
    ("device"),
    [   
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")),
    ],  
)

@device_params
def test_dumper_with_abs_filenames(tmp_path, device):
    """Save images with absolute filenames"""
    dummy_data = make_test_data(device)
    filenames = [tmp_path/f"{x}.jpg" for x in range(dummy_data.shape[0])]
    dumper = Dumper(n_workers=2)

    dumper.save(dummy_data, filenames=filenames)
    dumper.finish()

    assert all([x.exists() for x in filenames])

@device_params
def test_error_bad_data(tmp_path, device):
    """Error if data is not formatted correctly"""
    dummy_data = make_test_data(device)
    filenames = [tmp_path/f"{x}.jpg" for x in range(dummy_data.shape[0])]
    dumper = Dumper(n_workers=2)

    dummy_data = dummy_data.permute(0, 2, 3, 1)
    with pytest.raises(Exception):
        # Exception is only raised on subsequent calls to save so do 1 by 1
        for d, f in zip(dummy_data, filenames):
            dumper.save(d, filenames=f)

@device_params
def test_dumper_with_rel_filenames(tmp_path, device):
    """Provide relative filenames and base directory"""
    dummy_data = make_test_data(device)
    filenames = [f"{x}.jpg" for x in range(dummy_data.shape[0])]
    dumper = Dumper(n_workers=2)

    dumper.save(dummy_data, filenames=filenames, base_dir=tmp_path)
    dumper.finish()

    assert all([(tmp_path/x).exists() for x in filenames])

@device_params
def test_with_auto_naming(tmp_path, device):
    """With no filenames just increment a counter"""
    dummy_data = make_test_data(device)
    dumper = Dumper(n_workers=2)
    dumper.save(dummy_data, base_dir=tmp_path)
    # Make sure names are unique by writing twice
    dumper.save(dummy_data, base_dir=tmp_path)
    dumper.finish()

    n_files = len(list(tmp_path.glob("*")))
    assert n_files == 2*dummy_data.shape[0]

@device_params
def test_no_overwrite(tmp_path, device):
    """Error on attempted overwrite"""
    dummy_data = make_test_data(device)
    filenames = [tmp_path/f"{x}.jpg" for x in range(dummy_data.shape[0])]
    dumper = Dumper(n_workers=2)

    dumper.save(dummy_data, filenames=filenames)
    dumper.finish()

    with pytest.raises(Exception):
        # Exception is only raised on subsequent calls to save so do 1 by 1
        for d, f in zip(dummy_data, filenames):
            dumper.save(d, filenames=f)

@device_params
def test_allow_overwrite(tmp_path, device):
    """Allow overwrite with argument"""
    dummy_data = make_test_data(device)
    filenames = [tmp_path/f"{x}.jpg" for x in range(dummy_data.shape[0])]
    dumper = Dumper(n_workers=2)

    dumper.save(dummy_data, filenames=filenames)
    dumper.save(dummy_data, filenames=filenames, allow_overwrite=True)
    dumper.finish()

    assert all([x.exists() for x in filenames])

@device_params
def test_allow_finish_then_save(tmp_path, device):
    """Allow calling save again after finishing"""
    dummy_data = make_test_data(device)
    filenames = [tmp_path/f"{x}.jpg" for x in range(dummy_data.shape[0])]
    filenames2 = [tmp_path/f"2-{x}.jpg" for x in range(dummy_data.shape[0])]
    dumper = Dumper(n_workers=2)

    dumper.save(dummy_data, filenames=filenames)
    dumper.finish()
    dumper.save(dummy_data, filenames=filenames2)
    dumper.finish()

    assert all([x.exists() for x in filenames])
    assert all([x.exists() for x in filenames2])