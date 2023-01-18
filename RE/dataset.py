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

def get_parsing(parsing, y, content_len,parsing_mask, max_hop, relative_mode):
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

class REDataset(Dataset):

    def __init__(self, split='train', label_vocab=None, args = None, tokenizer = None):
        self.label_vocab = label_vocab
        self.args = args
        self.data = self.read(split, tokenizer)

        self.len = len(self.data)

    def read(self, split, tokenizer):
        with open(self.args.data_dir + 'DialogRE_v1/%s_data.json' % split, encoding='utf-8') as f:
            raw_data = json.load(f)
        with open(self.args.data_dir + 'DialogRE_v1/%s_data_parsing.json' % split, encoding='utf-8') as par:
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
            speaker_ids = []
            for i, u in enumerate(d['dialog']):
                cur_content_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('</s> ' + u['text']))
                utterance_len = len(cur_content_ids)
                content_ids = content_ids + cur_content_ids
                content_len.append((content_len[i])+utterance_len)

                cur_speaker_ids = int(u['speaker'][-1]) - int('0') + 1
                speaker_ids = speaker_ids + [cur_speaker_ids for k in range(len(cur_content_ids))]
                parsing_mask = get_parsing(parsing, i, content_len,parsing_mask, max_hop, self.args.relative_mode)
            # truncate before appending the arguments
            content_ids = content_ids[: self.args.max_sent_len]
            speaker_ids = speaker_ids[:self.args.max_sent_len]

            for j, r in enumerate(d['relations']):
                cur_subject_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('</s> '+ r['y']))
                cur_object_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('</s> ' + r['x'] + '</s>'))
                r_content_ids = content_ids + cur_subject_ids + cur_object_ids # r_content_ids = [CLS]d[SEP]a_1[SEP]a_2[SEP]
                r_speaker_ids = speaker_ids + [0 for k in range((len(cur_subject_ids) + len(cur_object_ids)))]
                # if len(content_ids)>=self.args.max_sent_len:
                #     print(f'leng>{self.args.max_sent_len}')
                new_parsing_mask = np.zeros((len(r_content_ids), len(r_content_ids)), dtype = np.int32)
                new_parsing_mask[:len(content_ids), :len(content_ids)] += parsing_mask[:len(content_ids), :len(content_ids)]
                new_parsing_mask[len(content_ids) :, :] = 1
                if self.args.relative_mode == 'bi':
                    new_parsing_mask[ :, len(content_ids):] = -1
                    new_parsing_mask[ len(content_ids):, len(content_ids):] = 1
                elif self.args.relative_mode == 'symm':
                    new_parsing_mask[ :, len(content_ids):] = 1
                # new_parsing_mask[:, len(content_ids):] = 1

                r_content_ids[0] = tokenizer.convert_tokens_to_ids('<s>') # since the first utterance begins with '</s>'
                # r_speaker_ids[0] = 0
                # r_token_types.insert(0, 0)
                label_list = [ 0 for i in range(37)]
                for rid in r['rid']:
                    label_list[rid - 1] = 1

                tmp_parsing_mask = new_parsing_mask
                if self.args.relative_mode == 'bi':
                    # if bi, we need to shift the index
                    tmp_parsing_mask = tmp_parsing_mask + max_hop + 1



                attention_mask = np.ones_like(r_content_ids)
                '''        
                # for i in range(0, len(tmp_parsing_mask)):
                #     tmp_parsing_mask[i][0] = 1
                #     tmp_parsing_mask[0][i] = 1
                # if len(content_ids)>max_len:
                #     max_len=len(content_ids)
                #     print(f'max_len is {max_len}')
                '''
                samples.append({
                    'content_ids': r_content_ids, #content_ids[self.args.max_sent_len:],
                    'labels': label_list,
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
               self.data[index]['labels'],\
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
        labels = torch.FloatTensor([d[1] for d in data])

        c = content_ids.size()
        q = []
        for d in data :
            tmp = pad(d[2],pad=(0,c[1]-d[2].size()[0],0,c[1]-d[2].size()[0]))
            q.append(tmp)
        parsing_mask = torch.stack(q,dim=0)

        attention_mask = pad_sequence([d[3] for d in data], batch_first = True)

        return content_ids, labels, parsing_mask, attention_mask
