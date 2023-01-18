import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
import torch.distributed as dist
import pandas as pd
import json
import copy
import numpy as np
import random
from pandas import DataFrame
from tqdm import tqdm
import queue
import pickle


class LMDataset(Dataset):

    def __init__(self, split = 'train',  args = None, tokenizer = None):
        self.args = args
        self.data = self.read(split)

        self.len = len(self.data)

    def read(self,  split):
        # print("process %d loading %s set..."%(dist.get_rank(), split))
        raw_data = pickle.load(open('data/%s/%s_data.pkl'%(self.args.dataset_name, split), 'rb'))
        # process dialogue
        samples = []
        for d in raw_data:
            content_ids = d['content_ids']
            parsing_mask = np.array(d['parsing_mask'], dtype=np.int)
            attention_mask = np.ones_like(content_ids)
            samples.append({
                'content_ids':content_ids,
                'parsing_mask':parsing_mask,
                'attention_mask':attention_mask
            })
        if split == 'train':
            random.shuffle(samples)
        return samples

    def __getitem__(self, index):
        '''

        :param index:
        :return:
            text_ids:
            token_types:
            label
        '''
        return {'input_ids': torch.LongTensor(self.data[index]['content_ids']), 
               'parsing_mask': torch.LongTensor(self.data[index]['parsing_mask']), 
               'attention_mask': torch.FloatTensor(self.data[index]['attention_mask'])}

    def __len__(self):
        return self.len



