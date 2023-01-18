import json
from os import write
from tqdm import tqdm
import queue
import numpy as np
from transformers import RobertaTokenizer
import random
import pickle
import time
import os


class parsing_link:
    def __init__(self,num = -1,step = 0):
        self.num = num
        self.step = step

def get_parsing(parsing, y,content_len,parsing_mask,   max_hop, relative_mode):
    # t = len(parsing_mask)
    # extend
    new_parsing_mask = np.zeros((content_len[y+1], content_len[y+1]), dtype = np.int)
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

def read( split, dataset_name, tokenizer, max_hop, relative_mode, max_seq_length):
    with open('data/%s/%s_data.json'%(dataset_name, split), encoding='utf-8') as f: 
        raw_data = json.load(f)
    
    
    # process dialogue
    samples = []
    max_hop = max_hop
    for d in tqdm(raw_data):
        content_len = [0]
        dialog = d['dialog']
        parsing = d['relations']
        parsing_mask = None # discourse connection
        content_ids = []
        for i, u in enumerate(dialog):
            content_ids += tokenizer.convert_tokens_to_ids(tokenizer.tokenize('</s> ' + u['text']))
            utterance_len = len(content_ids)
            content_len.append((content_len[i])+utterance_len)
            parsing_mask = get_parsing(parsing, i, content_len, parsing_mask, max_hop, relative_mode)
            if content_len[-1] > max_seq_length:
                break
        
        content_ids = content_ids[:max_seq_length]
        content_ids[0] = tokenizer.convert_tokens_to_ids('<s>')
        parsing_mask = parsing_mask[:len(content_ids),:len(content_ids)]
        if relative_mode == 'bi':
            # if bi, we need to shift the index.
            parsing_mask = parsing_mask + max_hop + 1
        # attention_mask = np.ones_like(content_ids)
        samples.append({
            'content_ids': content_ids, #content_ids[self.args.max_sent_len:],
            'parsing_mask': parsing_mask.tolist(),
            # 'attention_mask': attention_mask
        })
    print('%s size:'%(split), len(samples))
    if split == 'train':
        random.shuffle(samples)
    return samples

def process_and_write(split, dataset_name, tokenizer, max_hop, relative_mode, max_seq_length):
    print('processing %s set...'%(split))
    
    process_start = time.time()
    train_set = read(split, dataset_name, tokenizer=tokenizer, max_hop=max_hop, relative_mode=relative_mode, max_seq_length=max_seq_length)
    process_end = time.time()

    print('processing %s set cost'%(split), process_end-process_start,'s')



    print('pickling %s set...'%(split))
    train_obj = pickle.dumps(train_set)
    pickle_end = time.time()
    print('pickling %s set cost'%(split), pickle_end-process_end,'s')

    if not os.path.exists('data/%s'%(dataset_name)):
        os.makedirs('data/%s'%(dataset_name))

    print('writing %s set...'%(split))
    with open('data/%s/%s_data.pkl'%(dataset_name,split), 'wb') as f:
        f.write(train_obj)
    write_pickle_end = time.time()
    print('writing %s set to pickle cost'%(split), write_pickle_end-pickle_end, 's')

    del train_set
    del train_obj

    load_pickle_start = time.time()
    print('loading %s set from pickle file...'%(split))
    train_set = pickle.load(open('data/%s/%s_data.pkl'%(dataset_name, split),'rb'))
    load_pickle_end = time.time()
    print('loading %s set from pickle cost'%(split), load_pickle_end-load_pickle_start,'s')

    del train_set


if __name__ == "__main__":
    max_hop = 7
    relative_mode = 'bi'
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    max_seq_length = 512
    

    process_and_write('train','all', tokenizer=tokenizer, max_hop=max_hop, relative_mode=relative_mode, max_seq_length=max_seq_length)
    process_and_write('dev','all', tokenizer=tokenizer, max_hop=max_hop, relative_mode=relative_mode, max_seq_length=max_seq_length)




