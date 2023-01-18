from dataset import *
import pickle
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.distributed as dist
import os
import argparse
import numpy as np
from  transformers import RobertaTokenizer
from train_utils import SequentialDistributedSampler

def get_train_valid_sampler(trainset):
    size = len(trainset)
    idx = list(range(size))
    return SubsetRandomSampler(idx)


def load_vocab(args = None):
    label_vocab = pickle.load(open(args.data_dir + '/MELD/label_vocab.pkl', 'rb'))

    return label_vocab

def get_IEMOCAP_loaders_DDP(dataset_name = 'IEMOCAP', batch_size=32, num_workers=0, pin_memory=False, args = None):
    # distributed data loader
    tokenizer = RobertaTokenizer.from_pretrained(args.bert_tokenizer_dir)
    if dist.get_rank() == 0:
        print('building vocab.. ')
    label_vocab = load_vocab(args)
    if dist.get_rank() == 0:
        print('building datasets..')

    trainset = IEMOCAPDataset(dataset_name, 'train',   label_vocab, args, tokenizer)
    train_sampler = DistributedSampler(trainset, num_replicas=dist.get_world_size(), rank=dist.get_rank())

    train_loader = DataLoader(trainset,
                            batch_size=batch_size,
                            sampler=train_sampler,
                            collate_fn=trainset.collate_fn,
                            num_workers=num_workers,
                            pin_memory=pin_memory)

    devset = IEMOCAPDataset(dataset_name, 'dev',  label_vocab, args, tokenizer)
    dev_sampler = SequentialDistributedSampler(devset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    valid_loader = DataLoader(devset,
                            sampler=dev_sampler,
                            batch_size=batch_size,
                            collate_fn=devset.collate_fn,
                            num_workers=num_workers,
                            pin_memory=pin_memory)

    testset = IEMOCAPDataset(dataset_name, 'test',   label_vocab, args, tokenizer)
    test_sampler = SequentialDistributedSampler(testset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    test_loader = DataLoader(testset,
                            sampler=test_sampler,
                            batch_size=batch_size,
                            collate_fn=testset.collate_fn,
                            num_workers=num_workers,
                            pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader,  label_vocab, len(trainset), len(devset), len(testset)

def get_IEMOCAP_loaders(dataset_name = 'IEMOCAP', batch_size=32, num_workers=0, pin_memory=False, args = None):
    if args.local_rank != -1:
        return get_IEMOCAP_loaders_DDP(dataset_name=args.dataset_name, batch_size=batch_size, num_workers=0, args = args)
    else:
        tokenizer = RobertaTokenizer.from_pretrained(args.bert_tokenizer_dir)
        print('building vocab.. ')
        label_vocab = load_vocab(args)
        print('building datasets..')
        trainset = IEMOCAPDataset(dataset_name, 'train',   label_vocab, args, tokenizer)
        train_sampler = get_train_valid_sampler(trainset)

        train_loader = DataLoader(trainset,
                                batch_size=batch_size,
                                sampler=train_sampler,
                                collate_fn=trainset.collate_fn,
                                num_workers=num_workers,
                                pin_memory=pin_memory)

        devset = IEMOCAPDataset(dataset_name, 'dev', label_vocab, args, tokenizer)
        valid_loader = DataLoader(devset,
                                batch_size=batch_size,
                                collate_fn=devset.collate_fn,
                                num_workers=num_workers,
                                pin_memory=pin_memory)

        testset = IEMOCAPDataset(dataset_name, 'test',   label_vocab, args, tokenizer)
        test_loader = DataLoader(testset,
                                batch_size=batch_size,
                                collate_fn=testset.collate_fn,
                                num_workers=num_workers,
                                pin_memory=pin_memory)

        return train_loader, valid_loader, test_loader, label_vocab, len(trainset), len(devset), len(testset)


