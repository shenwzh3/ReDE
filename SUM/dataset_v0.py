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

# 创建a与b有关
# deposite 是已经抛弃的长度
# def have_relation(relation,a,b, content_len, step):
#     relation[content_len[a]]
#     for i in range(content_len[a+1] - content_len[a]):
#         for j in range(content_len[b+1] - content_len[b]):
#             if i + content_len[a] - deposite >=0 and j + content_len[b] - deposite>=0:
#                 relation [i + content_len[a] - deposite] [j + content_len[b] - deposite] = 1

def get_parsing(parsing, y,content_len,parsing_mask, rel_mask,  max_hop, discourse_rel_vocab):
    # t = len(parsing_mask)
    # extend
    new_parsing_mask = np.zeros((content_len[y+1], content_len[y+1]), dtype = np.int32)
    new_rel_mask = np.zeros((content_len[y+1], content_len[y+1]), dtype = np.int32)
    if content_len[y] > 0:
        new_parsing_mask[:content_len[y], :content_len[y]] += parsing_mask
        new_rel_mask[:content_len[y], :content_len[y]] += rel_mask
    
    # new_mask = have_relation(new_mask, y, y,content_len, 1)
    new_parsing_mask[content_len[y]:content_len[y+1], content_len[y]:content_len[y+1]] = 1
    new_rel_mask[content_len[y]:content_len[y+1], content_len[y]:content_len[y+1]] = 1

    q = queue.Queue()
    tmp = parsing_link(y, 1)
    q.put(tmp)
    while q.qsize() > 0:
        tmp = q.get()
        for r in parsing:
            if r['y'] == tmp.num:
                new_parsing_mask[content_len[y]:content_len[y+1], content_len[r['x']]:content_len[r['x']+1]] = tmp.step
                if r['y'] == y:
                    new_rel_mask[content_len[y]:content_len[y+1], content_len[r['x']]:content_len[r['x']+1]] = discourse_rel_vocab['stoi'][r['type']]
                    head_rel = r['type']
                else:
                    new_rel_mask[content_len[y]:content_len[y+1], content_len[r['x']]:content_len[r['x']+1]] = discourse_rel_vocab['stoi'][head_rel + '_' + r['type']]
                # have_relation(parsing_mask, y, r['x'], content_len, deposite)
                if tmp.step < max_hop+1:
                    tmp = parsing_link(r['x'], tmp.step+1)
                    q.put(tmp)
            if r['y'] > tmp.num:
                break

    return new_parsing_mask, new_rel_mask

class SamsumDataset(Dataset):

    def __init__(self, split = 'train', tokenizer = None, args = None):
        self.discourse_rel_vocab =  pickle.load(open('data/discouse_rel_vocab.pkl', 'rb'))
        self.args = args
        self.data = self.read(split, tokenizer)
        self.len = len(self.data)

    def read(self, split, tokenizer):
        with open('data/%s_data.json'% split, encoding='utf-8') as f: 
            raw_data = json.load(f)
        with open('data/Discourse/%s_data_parsing2.json'% split,encoding='utf-8') as par:
            raw_parsing = json.load(par)
        # process dialogue
        samples = []
        max_hop = self.args.max_hop
        for d, p in zip(raw_data, raw_parsing):
            content_len = [0]
            parsing = p['relations']
            parsing_mask = None # discourse connection
            rel_mask = None # discourse relation
            for i, u in enumerate(d):
                content_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('</s> ' + u['text']))
                utterance_len = len(content_ids)
                content_len.append((content_len[i])+utterance_len)
                parsing_mask, rel_mask = get_parsing(parsing, i, content_len, parsing_mask, rel_mask, max_hop, self.discourse_rel_vocab)
                if 'label' in u.keys():
                    token_types = [1 for k in range(len(content_ids))]
                    speaker_ids = [self.speaker_vocab['stoi'][u['speaker']]+1 for k in range(len(content_ids))]
                    for j in range(1, i + 1):
                        content_ids_temp = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('</s> '+d[i - j]['text']))
                        # if u['speaker'] != d[i - j]['speaker']:
                        #     token_type_temp = [1 for k in range(len(content_ids_temp))]
                        # else:
                        #     token_type_temp = [0 for k in range(len(content_ids_temp))]
                        # 使用parsing
                        content_ids = content_ids_temp + content_ids #    修改过，加了个tab
                        token_types = [0 for k in range(len(content_ids_temp))] + token_types
                        speaker_ids = [self.speaker_vocab['stoi'][d[i - j]['speaker']]+1 for k in range(len(content_ids_temp))] + speaker_ids

                    # if len(content_ids)>=self.args.max_sent_len:
                    #     print(f'leng>{self.args.max_sent_len}')
                    content_ids = content_ids[-self.args.max_sent_len:]
                    speaker_ids = speaker_ids[-self.args.max_sent_len:]
                    content_ids[0] = tokenizer.convert_tokens_to_ids('<s>')
                    speaker_ids[0] = self.speaker_vocab['stoi'][u['speaker']]+1

                    # ps = len(parsing_mask)
                    # c = len(content_ids)
                    # if ps > c:
                    #     deposite = deposite + ps - c
                    #     parsing_mask = [parsing_mask[i][ps-c:] for i in range(ps-c,ps)]
                    tmp_parsing_mask = copy.deepcopy(parsing_mask[-self.args.max_sent_len:, -self.args.max_sent_len:])
                    tmp_parsing_mask[0,:] = tmp_parsing_mask[-1,:]
                    tmp_parsing_mask[:,0] = tmp_parsing_mask[:,-1]
                    tmp_parsing_mask[0,0] = 1

                    tmp_rel_mask = copy.deepcopy(rel_mask[-self.args.max_sent_len:, -self.args.max_sent_len:])
                    tmp_rel_mask[0,:] = tmp_rel_mask[-1,:]
                    tmp_rel_mask[:,0] = tmp_rel_mask[:,-1]
                    tmp_rel_mask[0,0] = 1

                    attention_mask = np.ones_like(content_ids)
                    
                    # for i in range(0, len(tmp_parsing_mask)):
                    #     tmp_parsing_mask[i][0] = 1
                    #     tmp_parsing_mask[0][i] = 1
                    # if len(content_ids)>max_len:
                    #     max_len=len(content_ids)
                    #     print(f'max_len is {max_len}')

                    samples.append({
                        'content_ids': content_ids, #content_ids[self.args.max_sent_len:],
                        'speaker_ids': speaker_ids,
                        'token_types': token_types[-self.args.max_sent_len:],
                        'parsing_mask': tmp_parsing_mask,
                        'rel_mask': tmp_rel_mask,
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
               torch.LongTensor(self.data[index]['speaker_ids']), \
               torch.LongTensor(self.data[index]['token_types']), \
               self.data[index]['labels'], self.data[index]['length'],self.data[index]['seq_len'], \
               torch.LongTensor(self.data[index]['parsing_mask']), torch.LongTensor(self.data[index]['rel_mask']), torch.FloatTensor(self.data[index]['attention_mask'])

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
        content_ids = pad_sequence([d[0] for d in data], batch_first = True) # (B, T, )
        speaker_ids = pad_sequence([d[1] for d in data], batch_first = True)
        token_types = pad_sequence([d[2] for d in data], batch_first = True) # (B, T, )
        labels = torch.LongTensor([d[3] for d in data])
        length = torch.LongTensor([d[4] for d in data])
        seq_len = torch.LongTensor([d[5] for d in data])

        c = content_ids.size()
        q = []
        for d in data :
            tmp = pad(d[6],pad=(0,c[1]-d[6].size()[0],0,c[1]-d[6].size()[0]))
            q.append(tmp)
        parsing_mask = torch.stack(q,dim=0)

        q = []
        for d in data :
            tmp = pad(d[7],pad=(0,c[1]-d[7].size()[0],0,c[1]-d[7].size()[0]))
            q.append(tmp)
        rel_mask = torch.stack(q,dim=0)

        attention_mask = pad_sequence([d[8] for d in data], batch_first = True)

        return content_ids, speaker_ids, token_types, parsing_mask ,rel_mask,labels,length,seq_len,attention_mask
