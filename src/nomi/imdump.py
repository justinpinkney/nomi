from pathlib import Path
from typing import List, Optional, Union
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import weakref
import numpy as np
import torch

def save_fn(image_data, outname, allow_overwrite=False):
    try:
        if allow_overwrite and Path(outname).exists():
            raise OSError(f"Filename {outname} already exists")
        im = Image.fromarray(image_data)
        im.save(outname)
    except Exception as e:
        print(e)
        return e


class Dumper():
    """Saves tensors to batches of images"""
    def __init__(self, n_workers:Optional[int]=None) -> None:
        self.n_workers = n_workers
        self._create_new_executor()
        self.count = 0
        self.failed = False

    def save(self,
        images:torch.Tensor,
        filenames:Optional[List[Union[Path,str]]]=None,
        base_dir:Optional[Path]=None,
        allow_overwrite:bool=False,
    ):
        """Save tensors to images with given filenames
        if no filenames give them increasing numeric values
        """
        # Assume all scaling and conversion is done by the user
        # TODO accept numpy array too
        if self._executor._shutdown:
            self._create_new_executor()
        if filenames is None:
            n_files = images.shape[0]
            filenames = [f"{self.count + idx:08}.png" for idx in range(n_files)]

        if base_dir is not None:
            filenames = [Path(base_dir)/x for x in filenames]

        to_save = images.detach().to("cpu").numpy().astype(np.uint8)
        for im, name in zip(to_save, filenames):
            submit_fn = lambda x, y: save_fn(x, y, allow_overwrite=allow_overwrite)
            future = self._executor.submit(submit_fn, im, name)
            future.add_done_callback(self._check_result)
            self.count += 1

        if self.failed:
            raise Exception("failed")

    def finish(self):
        """Block until all saving is completed"""
        #https://docs.python.org/3/library/weakref.html?highlight=weakref#comparing-finalizers-with-del-methods
        self._finaliser()

    @property
    def finished(self):
        return not self._finaliser.alive

    def _check_result(self, future):
        if future.result() is not None:
            self.failed = True

    def _create_new_executor(self):
        """Make it easy to create a new executor"""
        n_workers = self.n_workers
        if n_workers==0:
            raise NotImplementedError(f"Non-threaded saving not implemented")

        self._executor = ThreadPoolExecutor(max_workers=n_workers)
        self._finaliser = weakref.finalize(self, self._executor.shutdown, wait=True)