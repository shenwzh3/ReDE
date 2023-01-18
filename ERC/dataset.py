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


def get_parsing(parsing, y,content_len,parsing_mask, max_hop, relative_mode):

    new_parsing_mask = np.zeros((content_len[y+1], content_len[y+1]), dtype = np.int32)
    if content_len[y] > 0:
        new_parsing_mask[:content_len[y], :content_len[y]] += parsing_mask
    
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

class IEMOCAPDataset(Dataset):

    def __init__(self, dataset_name = 'IEMOCAP', split = 'train',  label_vocab=None, args = None, tokenizer = None):
        self.label_vocab = label_vocab
        self.args = args
        self.data = self.read(dataset_name, split, tokenizer)

        self.len = len(self.data)

    def read(self, dataset_name, split, tokenizer):
        with open(self.args.data_dir + '/MELD/%s_data.json' % split, encoding='utf-8') as f:
            raw_data = json.load(f)
        with open(self.args.data_dir + '/MELD/%s_data_parsing2.json' % split, encoding='utf-8') as par:
            raw_parsing = json.load(par)
        # process dialogue
        samples = []
        max_hop = self.args.max_hop
        for d, p in zip(raw_data, raw_parsing):
            content_len = [0]
            parsing = p['relations']
            parsing_mask = None # discourse connection
            for i, u in enumerate(d):
                content_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('</s> ' + u['text']))
                utterance_len = len(content_ids)
                content_len.append((content_len[i])+utterance_len)
                parsing_mask = get_parsing(parsing, i, content_len, parsing_mask, max_hop, self.args.relative_mode)
                if 'label' in u.keys():
                    for j in range(1, i + 1):
                        content_ids_temp = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('</s> '+d[i - j]['text']))
                        content_ids = content_ids_temp + content_ids 


                    content_ids = content_ids[-self.args.max_sent_len:]


                    tmp_parsing_mask = copy.deepcopy(parsing_mask[-self.args.max_sent_len:, -self.args.max_sent_len:])
                    tmp_parsing_mask[0,:] = tmp_parsing_mask[-1,:]
                    tmp_parsing_mask[:,0] = tmp_parsing_mask[:,-1]
                    tmp_parsing_mask[0,0] = 1

                    if self.args.relative_mode == 'bi':
                        tmp_parsing_mask = tmp_parsing_mask + max_hop + 1


                    attention_mask = np.ones_like(content_ids)
                

                    samples.append({
                        'content_ids': content_ids, #content_ids[self.args.max_sent_len:],
                        'parsing_mask': tmp_parsing_mask,
                        'labels': self.label_vocab['stoi'][u['label']],
                        'length': int(utterance_len),
                        'seq_len': len(content_ids),
                        'attention_mask': attention_mask
                    })
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
               self.data[index]['labels'], self.data[index]['length'],self.data[index]['seq_len'], \
               torch.LongTensor(self.data[index]['parsing_mask']), torch.FloatTensor(self.data[index]['attention_mask'])

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
        labels = torch.LongTensor([d[1] for d in data])
        length = torch.LongTensor([d[2] for d in data])
        seq_len = torch.LongTensor([d[3] for d in data])

        c = content_ids.size()
        q = []
        for d in data :
            tmp = pad(d[4],pad=(0,c[1]-d[4].size()[0],0,c[1]-d[4].size()[0]))
            q.append(tmp)
        parsing_mask = torch.stack(q,dim=0)


        attention_mask = pad_sequence([d[5] for d in data], batch_first = True)

        return content_ids, parsing_mask,labels,length,seq_len,attention_mask
