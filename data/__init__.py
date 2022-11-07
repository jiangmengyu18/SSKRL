from importlib import import_module
import torch.utils.data as data
from torch.utils.data import DataLoader
from .multiscalesrdata import *

class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            trainset = SRData(args, benchmark=False)
            self.loader_train = DataLoader(
                trainset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.n_threads,
                drop_last=True,
                pin_memory=True,
            )

        if args.data_test in ['Set5', 'Set14', 'B100', 'Manga109', 'Urban100', 'test', 'test_aniso']:
            testset = SRData(args, benchmark=True)
        elif args.data_test in ['DIV2KRK']:
            testset = HRLRData(args, benchmark=True)

        self.loader_test = DataLoader(
                testset,
                batch_size=1,
                shuffle=False,
                num_workers=args.n_threads,
                drop_last=False,
                pin_memory=True,
            )

