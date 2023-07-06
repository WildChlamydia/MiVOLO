"""
Code adapted from timm https://github.com/huggingface/pytorch-image-models

Modifications and additions for mivolo by / Copyright 2023, Irina Tolstykh, Maxim Kuprashevich
"""

import logging
from contextlib import suppress
from functools import partial
from itertools import repeat

import numpy as np
import torch
import torch.utils.data
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.dataset import IterableImageDataset
from timm.data.loader import PrefetchLoader, _worker_init
from timm.data.transforms_factory import create_transform

_logger = logging.getLogger(__name__)


def fast_collate(batch, target_dtype=torch.uint8):
    """A fast collation function optimized for uint8 images (np array or torch) and target_dtype targets (labels)"""
    assert isinstance(batch[0], tuple)
    batch_size = len(batch)
    if isinstance(batch[0][0], np.ndarray):
        targets = torch.tensor([b[1] for b in batch], dtype=target_dtype)
        assert len(targets) == batch_size
        tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.uint8)
        for i in range(batch_size):
            tensor[i] += torch.from_numpy(batch[i][0])
        return tensor, targets
    else:
        raise ValueError(f"Incorrect batch type: {type(batch[0][0])}")


def adapt_to_chs(x, n):
    if not isinstance(x, (tuple, list)):
        x = tuple(repeat(x, n))
    elif len(x) != n:
        # doubled channels
        if len(x) * 2 == n:
            x = np.concatenate((x, x))
            _logger.warning(f"Pretrained mean/std different shape than model (doubled channes), using concat: {x}.")
        else:
            x_mean = np.mean(x).item()
            x = (x_mean,) * n
            _logger.warning(f"Pretrained mean/std different shape than model, using avg value {x}.")
    else:
        assert len(x) == n, "normalization stats must match image channels"
    return x


class PrefetchLoaderForMultiInput(PrefetchLoader):
    def __init__(
        self,
        loader,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        channels=3,
        device=torch.device("cuda"),
        img_dtype=torch.float32,
    ):

        mean = adapt_to_chs(mean, channels)
        std = adapt_to_chs(std, channels)
        normalization_shape = (1, channels, 1, 1)

        self.loader = loader
        self.device = device
        self.img_dtype = img_dtype
        self.mean = torch.tensor([x * 255 for x in mean], device=device, dtype=img_dtype).view(normalization_shape)
        self.std = torch.tensor([x * 255 for x in std], device=device, dtype=img_dtype).view(normalization_shape)

        self.is_cuda = torch.cuda.is_available() and device.type == "cuda"

    def __iter__(self):
        first = True
        if self.is_cuda:
            stream = torch.cuda.Stream()
            stream_context = partial(torch.cuda.stream, stream=stream)
        else:
            stream = None
            stream_context = suppress

        for next_input, next_target in self.loader:

            with stream_context():
                next_input = next_input.to(device=self.device, non_blocking=True)
                next_target = next_target.to(device=self.device, non_blocking=True)
                next_input = next_input.to(self.img_dtype).sub_(self.mean).div_(self.std)

            if not first:
                yield input, target  # noqa: F823, F821
            else:
                first = False

            if stream is not None:
                torch.cuda.current_stream().wait_stream(stream)

            input = next_input
            target = next_target

        yield input, target


def create_loader(
    dataset,
    input_size,
    batch_size,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
    num_workers=1,
    crop_pct=None,
    crop_mode=None,
    pin_memory=False,
    img_dtype=torch.float32,
    device=torch.device("cuda"),
    persistent_workers=True,
    worker_seeding="all",
    target_type=torch.int64,
):

    transform = create_transform(
        input_size,
        is_training=False,
        use_prefetcher=True,
        mean=mean,
        std=std,
        crop_pct=crop_pct,
        crop_mode=crop_mode,
    )
    dataset.transform = transform

    if isinstance(dataset, IterableImageDataset):
        # give Iterable datasets early knowledge of num_workers so that sample estimates
        # are correct before worker processes are launched
        dataset.set_loader_cfg(num_workers=num_workers)
        raise ValueError("Incorrect dataset type: IterableImageDataset")

    loader_class = torch.utils.data.DataLoader
    loader_args = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        sampler=None,
        collate_fn=lambda batch: fast_collate(batch, target_dtype=target_type),
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=partial(_worker_init, worker_seeding=worker_seeding),
        persistent_workers=persistent_workers,
    )
    try:
        loader = loader_class(dataset, **loader_args)
    except TypeError:
        loader_args.pop("persistent_workers")  # only in Pytorch 1.7+
        loader = loader_class(dataset, **loader_args)

    loader = PrefetchLoaderForMultiInput(
        loader,
        mean=mean,
        std=std,
        channels=input_size[0],
        device=device,
        img_dtype=img_dtype,
    )

    return loader
