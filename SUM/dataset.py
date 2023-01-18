import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
import pickle, pandas as pd
import json
import copy
import numpy as np
import random
from pandas import DataFrame
import queue

class parsing_link:
    def __init__(self,num = -1,step = 0):
        self.num = num
        self.step = step


def get_parsing(parsing, y,content_len,parsing_mask, max_hop,  relative_mode):
    # t = len(parsing_mask)
    # extend
    new_parsing_mask = np.zeros((content_len[y+1], content_len[y+1]), dtype = np.int32)
    if content_len[y] > 0:
        new_parsing_mask[:content_len[y], :content_len[y]] += parsing_mask
    
    # new_mask = have_relation(new_mask, y, y,content_len, 1)
    new_parsing_mask[content_len[y]:content_len[y+1], content_len[y]:content_len[y+1]] = 1

    q = queue.Queue()
    tmp = parsing_link(y, 1)
    q.put(tmp)
    while q.qsize() > 0:
        tmp = q.get()
        for r in parsing:
            if r['y'] == tmp.num:
                new_parsing_mask[content_len[y]:content_len[y+1], content_len[r['x']]:content_len[r['x']+1]] = tmp.step
                if relative_mode == 'bi':
                    new_parsing_mask[content_len[r['x']]:content_len[r['x']+1], content_len[y]:content_len[y+1]] = -tmp.step
                elif relative_mode == 'symm':
                    new_parsing_mask[content_len[r['x']]:content_len[r['x']+1], content_len[y]:content_len[y+1]] = tmp.step


                # have_relation(parsing_mask, y, r['x'], content_len, deposite)
                if tmp.step < max_hop:
                    tmp = parsing_link(r['x'], tmp.step+1)
                    q.put(tmp)
            if r['y'] > tmp.num:
                break

    return new_parsing_mask

class SamsumDataset(Dataset):

    def __init__(self,  split = 'train', tokenizer = None, args = None):

        self.args = args
        self.data = self.read(split, tokenizer)

        self.len = len(self.data)

    def read(self, split, tokenizer):
        with open(self.args.data_path + '/%s_data.json'%(split), encoding='utf-8') as f:
            raw_data = json.load(f)
        with open(self.args.data_path + '/%s_data_parsing.json'%(split),encoding='utf-8') as par:
            raw_parsing = json.load(par)
        # process dialogue
        samples = []
        max_hop = self.args.max_hop
        for d, p in zip(raw_data, raw_parsing):
        # for d in raw_data:
            content_len = [0]
            parsing = p['relations']
            parsing_mask = None # discourse connection
            content_ids = []
            for i, u in enumerate(d['dialogue']):
                cur_content_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('</s> ' + u['text']))
                utterance_len = len(cur_content_ids)
                content_len.append((content_len[i])+utterance_len)
                parsing_mask= get_parsing(parsing, i, content_len, parsing_mask,  max_hop, self.args.relative_mode)
                content_ids = content_ids + cur_content_ids

            label = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('<s> '+d['summary']))

            # if len(content_ids)>=self.args.max_sent_len:
            #     print(f'leng>{self.args.max_sent_len}')
            content_ids = content_ids[:self.args.max_source_length]
            content_ids[0] = tokenizer.convert_tokens_to_ids('<s>')
            # speaker_ids.insert(0,0)
            label = label[:self.args.max_target_length-1]
            label +=  tokenizer.convert_tokens_to_ids(tokenizer.tokenize('</s>'))


            tmp_parsing_mask = copy.deepcopy(parsing_mask[:self.args.max_source_length, :self.args.max_source_length])
            if self.args.relative_mode == 'bi':
                # if bi, we need to shift the index
                tmp_parsing_mask = tmp_parsing_mask + max_hop + 1
            # t_parsing_mask = np.zeros((len(tmp_parsing_mask) + 1, len(tmp_parsing_mask) + 1))
            # t_parsing_mask[-len(tmp_parsing_mask):, -len(tmp_parsing_mask):] += tmp_parsing_mask
            # t_parsing_mask[0, :] = t_parsing_mask[-1, :]
            # t_parsing_mask[:, 0] = t_parsing_mask[:, -1]
            # t_parsing_mask[0, 0] = 1


            attention_mask = np.ones_like(content_ids)

            # for i in range(0, len(tmp_parsing_mask)):
            #     tmp_parsing_mask[i][0] = 1
            #     tmp_parsing_mask[0][i] = 1
            # if len(content_ids)>max_len:
            #     max_len=len(content_ids)
            #     print(f'max_len is {max_len}')

            samples.append({
                'content_ids': content_ids, #content_ids[self.args.max_sent_len:],
                'labels': label,
                'parsing_mask': tmp_parsing_mask,
                'attention_mask': attention_mask
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
        return torch.LongTensor(self.data[index]['content_ids']), \
               torch.LongTensor(self.data[index]['labels']), torch.LongTensor(self.data[index]['parsing_mask']),torch.FloatTensor(self.data[index]['attention_mask'])

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        '''

        :param data:
            content_ids
            token_types
            labels

        :return:

        '''
        content_ids = pad_sequence([d[0] for d in data], batch_first = True, padding_value = 1) # (B, T, )
        labels = pad_sequence([d[1] for d in data], batch_first = True, padding_value=-100)

        c = content_ids.size()
        q = []
        for d in data :
            tmp = pad(d[2],pad=(0,c[1]-d[2].size()[0],0,c[1]-d[2].size()[0]))
            q.append(tmp)
        parsing_mask = torch.stack(q,dim=0)



        attention_mask = pad_sequence([d[3] for d in data], batch_first = True)

        sample = {}
        sample['input_ids'] = content_ids
        sample['disc_connect_ids'] = parsing_mask
        sample['labels'] = labels
        sample['attention_mask'] = attention_mask
        # print(sample)
        # exit()
        return sample
