import six
from rich import print
import torch
import random
import math
import pickle
import os
import numpy as np
from lib.utils import timeit
from lib.common import TrainArg, gState
from lib.comm import get_local_rank, get_rank, all_gather, synchronize, get_world_size


def Cifar(args):
    args: TrainArg = args
    import os
    if args.num_classes == 100:
        from torchvision.datasets.cifar import CIFAR100 as CIFARDATASET
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif args.num_classes == 10:
        from torchvision.datasets.cifar import CIFAR10 as CIFARDATASET
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    else:
        raise ValueError(
            "Unsupported Number of Classes: %d".format(args.num_classes))

    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    import torchvision.transforms as T

    root = args.data
    print("[green]load data from {}".format(root))
    print("[green]statistics: {} {}".format(mean, std))

    class Identity:
        def __call__(self, image):
            return image

    class Cutout:
        def __init__(self, size=16, p=0.5):
            self.size = size
            self.half_size = size // 2
            self.p = p

        def __call__(self, image):
            if torch.rand([1]).item() > self.p:
                return image

            left = torch.randint(-self.half_size,
                                 image.size(1) - self.half_size, [1]).item()
            top = torch.randint(-self.half_size, image.size(2) -
                                self.half_size, [1]).item()
            right = min(image.size(1), left + self.size)
            bottom = min(image.size(2), top + self.size)

            image[:, max(0, left): right, max(0, top): bottom] = 0
            return image

    train_transform = T.Compose([
        T.RandomCrop(size=(32, 32), padding=4),
        T.RandomHorizontalFlip(),
        T.AutoAugment(T.AutoAugmentPolicy.CIFAR10) if args.aa else Identity(),
        T.ToTensor(),
        T.Normalize(mean, std),
        Cutout(size=16 if args.num_classes == 10 else 8)
    ])

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std)
    ])

    if get_rank() == 0:
        trainset = CIFARDATASET(root, True,  transform=train_transform, download=True)
        synchronize()
    else:
        synchronize()
        trainset = CIFARDATASET(root, True,  transform=train_transform, download=False)

    valset = CIFARDATASET(root, False,  transform=test_transform)

    train_sampler = DistributedSampler(trainset, shuffle=True, drop_last=True)
    val_sampler = DistributedSampler(valset, shuffle=False, drop_last=False)
    train_loader = DataLoader(dataset=trainset,
                              sampler=train_sampler,
                              batch_size=args.batch_size,
                              num_workers=args.workers)
    val_loader = DataLoader(dataset=valset,
                            sampler=val_sampler,
                            batch_size=args.batch_size,
                            num_workers=args.workers)

    train_loader._size = len(train_loader) * args.batch_size
    val_loader._size = len(val_loader) * args.batch_size

    print("[yellow]",
          "trainset: {}\n"
          "train_sampler: {}\n"
          "train_loader: {}\n"
          .format(len(trainset), len(train_sampler), len(train_loader)))

    def _reset(j=None):
        if j is None:
            j = gState.global_epoch + 1
        train_sampler.set_epoch(j)
    train_loader.reset = _reset
    val_loader.reset = lambda: None

    if args.start_epoch != 0:
        train_loader.reset(args.start_epoch)

    if not args.evaluate:
        gState.train_loader_len = math.ceil(
            train_loader._size/args.batch_size)

    gState.val_loader_len = math.ceil(
        val_loader._size/args.batch_size)

    return train_loader, val_loader


class Basic_ITER(object):
    def __init__(self, batch_size, rank, world_size, file: str, shuffle=True, seed=0):
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rand = random.Random(0)
        self.seed = gState.global_epoch + seed

    def __iter__(self):
        self.i = 0
        if self.shuffle:
            self.shuffle_with_seed(self.seed)
            print("[red]Shuffle with seed: {}".format(self.seed))
            self.seed += 1
        return self

    def __len__(self):
        return self.data_set_len

    def next(self, *args, **kwargs):
        return self.__next__(*args, **kwargs)

    def shuffle_with_seed(self, seed):
        self.rand.seed(seed)
        self.all_id = list(range(self.data_set_len))
        self.rand.shuffle(self.all_id)
        self.ids = self.all_id[
            self.data_set_len * self.rank // self.world_size:
            self.data_set_len * (self.rank + 1) // self.world_size]
        self.n = len(self.ids)


def loads_data(buf):
    return pickle.loads(buf)


class LMDB_ITER(Basic_ITER):
    def __init__(self, batch_size, rank, world_size, file, shuffle=True, seed=0):
        import lmdb
        import os.path as osp
        from torch.utils import data

        def loads_data(buf):
            return pickle.loads(buf)

        class ImageFolderLMDB(data.Dataset):
            def __init__(self,
                         db_path):
                self.db_path = db_path
                sz = os.stat(self.db_path).st_size
                self.env = lmdb.open(
                    db_path, map_size=sz,
                    subdir=osp.isdir(db_path),
                    readonly=True, lock=False,
                    readahead=True, meminit=False)
                with self.env.begin(write=False) as txn:
                    self._length = loads_data(txn.get(b'__len__'))
                    self._keys = loads_data(txn.get(b'__keys__'))
                self.trans = None

            def __getitem__(self, index):
                env = self.env
                if self.trans is None:
                    with env.begin(write=False) as txn:
                        byteflow = txn.get(self._keys[index])
                else:
                    byteflow = self.trans.get(self._keys[index])
                unpacked = loads_data(byteflow)
                im, lab = unpacked
                im = np.frombuffer(im, dtype=np.uint8)
                lab = np.int32(lab)
                return im, lab

            def __len__(self):
                return self._length

        self.db = ImageFolderLMDB(file)
        self.data_set_len = len(self.db)
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rand = random.Random(0)
        self.shuffle_with_seed(0)
        self.seed = gState.global_epoch + seed
        self.n = len(self.ids)

    def __next__(self):
        batch = []
        labels = []

        if self.i >= self.n:
            self.__iter__()
            raise StopIteration
        with self.db.env.begin(write=False) as txn:
            self.db.trans = txn
            for _ in range(self.batch_size):
                buf, label = self.db[self.ids[self.i % self.n]]
                batch.append(buf)
                labels.append(label)
                self.i += 1
            self.db.trans = None
        return (batch, labels)

    next = __next__

def ImageNet_Loader(inputIterator, args: TrainArg):
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali import types, fn
    import os
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.plugin.pytorch import LastBatchPolicy

    def TP(batch_size, num_threads, device_id, EII, crop=224):
        pipe = Pipeline(batch_size, num_threads, device_id)
        with pipe:
            jpegs, labels = fn.external_source(
                source=EII, num_outputs=2)
            images = fn.decoders.image_random_crop(
                jpegs, device="cpu",
                output_type=types.RGB,
                random_aspect_ratio=[0.8, 1.25],
                random_area=[0.1, 1.0],
                num_attempts=100)
            images = images.gpu()
            images = fn.resize(
                images, resize_x=crop, resize_y=crop,
                interp_type=types.INTERP_TRIANGULAR)
            coin = fn.random.coin_flip(probability=0.5)
            images = images.gpu()
            images = fn.crop_mirror_normalize(
                images, output_layout=types.NCHW,
                dtype=types.FLOAT, crop=(crop, crop),
                mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                mirror=coin)
            pipe.set_outputs(images, labels)
        return pipe

    def VP(batch_size, num_threads, device_id, EII, crop=224, size=256):
        pipe = Pipeline(batch_size, num_threads, device_id)
        with pipe:
            jpegs, labels = fn.external_source(
                source=EII, num_outputs=2)
            images = fn.decoders.image(
                jpegs, device="cpu", output_type=types.RGB)
            images = images.gpu()
            images = fn.resize(
                images, resize_shorter=size,
                interp_type=types.INTERP_TRIANGULAR)
            images = fn.crop_mirror_normalize(
                images, output_layout=types.NCHW,
                dtype=types.FLOAT, crop=(crop, crop),
                mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
            pipe.set_outputs(images, labels)
        return pipe

    def get_loaders(train_file,
                    val_file,
                    batch_size,
                    num_threads,
                    crop, size,
                    *,
                    rank=0,
                    device_id=0,
                    world_size=1,
                    val_only=False):
        if not val_only:
            eii_train = inputIterator(
                batch_size, rank, world_size, train_file, shuffle=True, seed=args.seed)
            pip_train = TP(batch_size, num_threads,
                           device_id, eii_train, crop)

        eii_val = inputIterator(
            batch_size, rank, world_size, val_file, shuffle=False, seed=args.seed)
        pip_val = VP(batch_size, num_threads,
                     device_id, eii_val, crop, size)

        return DALIClassificationIterator(pip_train, last_batch_padded=True,
                                          last_batch_policy=LastBatchPolicy.PARTIAL) if not val_only else None,\
            DALIClassificationIterator(pip_val, last_batch_padded=True,
                                       last_batch_policy=LastBatchPolicy.PARTIAL),\
            eii_train.n if not val_only else 0, eii_val.n

    if (args.arch == "inception_v3"):
        raise RuntimeError("Inception_v3 is not supported by this example.")
    else:
        crop_size = 224
        val_size = 256

    train_data = args.train_data
    val_data = args.val_data
    print("[green]build dataloaders")
    synchronize()
    train_loader, val_loader, gState.train_loader_len, gState.val_loader_len = get_loaders(
        train_data,
        val_data,
        batch_size=args.batch_size,
        num_threads=args.workers,
        crop=crop_size, size=val_size,
        device_id=get_local_rank(),
        rank=get_rank(),
        world_size=get_world_size(),
        val_only=args.evaluate
    )
    print("[green]loader's length: {} {}".format(
        gState.train_loader_len, gState.val_loader_len))
    if not args.evaluate:
        gState.train_loader_len = math.ceil(
            gState.train_loader_len/args.batch_size)
    gState.val_loader_len = math.ceil(
        gState.val_loader_len/args.batch_size)
    return train_loader, val_loader


def from_args(args: TrainArg):
    if args.cifar:
        return Cifar(args)
    else:
        data_root = args.lmdb_path or args.data
        args.train_data = os.path.join(data_root, 'train.lmdb')
        args.val_data = os.path.join(data_root, 'val.lmdb')
        return ImageNet_Loader(LMDB_ITER, args)


if __name__ == "__main__":
    args = TrainArg(
        data="/data1/yiming/dataset/I100",
    )
    trl, val = from_args(args)

    for epoch in range(args.epochs):
        gState.global_epoch = epoch

        for i, data in enumerate(trl):
            gState.global_step = epoch * gState.train_loader_len + i
            pass

        for data in val:
            pass

        trl.reset()
        val.reset()
