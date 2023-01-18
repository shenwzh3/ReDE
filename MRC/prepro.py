
# coding=utf-8

import json
from numpy.lib.twodim_base import mask_indices
from numpy.ma.core import doc_note
from transformers import RobertaTokenizerFast
import copy
from tqdm import tqdm
import numpy as np
import queue
from torch.utils.data import Dataset
import torch

class Utterance():
    def __init__(self, start_char_index, end_char_index, head):
        '''
        start_char_index: the start char index in context
        end_char_index: the start end index in context
        head: the head in context
        '''
        self.start_char_index = start_char_index
        self.end_char_index = end_char_index
        self.head = head
        self.start_token_index = None
        self.end_token_index = None


class parsing_link:
    def __init__(self,num = -1,step = 0):
        self.num = num
        self.step = step

def get_adj_mat(utterance_list, max_hop, relative_mode):
    '''
    build the adj matrix for utterances
    '''

    mat = np.zeros((len(utterance_list), len(utterance_list)), dtype = np.int32)
    for i,u in enumerate(utterance_list):
        mat[i,i] = 1
        if i != 0:
            q = queue.Queue()
            tmp = parsing_link(i, 1)
            q.put(tmp)
            while q.qsize() > 0:
                tmp = q.get()
                # print(i, tmp.num, utterance_list[tmp.num].head)
                mat[i, utterance_list[tmp.num].head] = tmp.step
                mat[utterance_list[tmp.num].head, i] = tmp.step
                if utterance_list[tmp.num].head != 0:
                    tmp = parsing_link(utterance_list[tmp.num].head, tmp.step+1)
                    q.put(tmp)
    
    # print(mat)
    return mat

    
        

def read(split, args):
    '''
    read all processed qa from data file 
    '''
    with open(args.data_path + '%s.json'%(split)) as f:
        file = json.load(f)
    data = file['data']['dialogues']
    # for d in tqdm(data):
    examples = []

    # for d in tqdm(data):
    for d in data:
        dialog = d['edus']
        relations = d['relations']
        qas = d['qas']
        origin_qas = copy.deepcopy(qas)
        context = "" # context with </s> token to seperate each utterance
        origin_context = "" # context without </s> tokn, utterances concatenated with a space
        utterance_list = []
        for i,u in enumerate(dialog):

            tmp_start_char_index = len(context)
            tmp_start_char_index_origin = len(origin_context)
            tmp_context = "</s> " + u['speaker']+': ' + u['text']
            context += tmp_context
            context += " "
            tmp_end_char_index = len(context) - 1 # since we add a space in the rare of the sentence

            tmp_context_origin = u['speaker']+': ' + u['text']
            origin_context += tmp_context_origin
            origin_context += " "
            tmp_end_char_index_origin = len(origin_context) - 1

            # push current utterance to the utterance list
            find = False
            for r in relations:
                if i == r['y']:
                    if r['x'] >= r['y']:
                        continue
                    utterance_list.append(Utterance(tmp_start_char_index, tmp_end_char_index, r['x']))
                    find = True
                    break
            if not find:
                if i == 0:
                    utterance_list.append(Utterance(tmp_start_char_index, tmp_end_char_index, None))
                else:
                    utterance_list.append(Utterance(tmp_start_char_index, tmp_end_char_index, 0))

            # add an offset the start position of answers 
            for j,q in enumerate(origin_qas):
                for k,ans in enumerate(q['answers']):
                    if ans['answer_start'] >= tmp_start_char_index_origin:
                        qas[j]['answers'][k]['answer_start'] += 5 # add the offset of "</s> "

        # lower the context
        context = context[:-1].lower() # for the space in the rare of the context
        origin_context = origin_context[:-1].lower()

        # build the utterance relative adj matrix
        adj_mat = get_adj_mat(utterance_list, args.max_hop, args.relative_mode)

        # push all above to qas info
        for q in qas:
            examples.append({
                'context': context,
                'question':q['question'],
                'id': q['id'],
                'answers': {
                    'answer_start': [answer["answer_start"] for answer in q["answers"]],
                    'text': [answer["text"] for answer in q["answers"]]
                },
                'adj_mat':adj_mat,
                'utterance_list':utterance_list
            })
    # examples = {
    #     'context':context_list,
    #     'question':question_list,
    #     'id':id_list,
    #     'answers':answers_list,
    #     'adj_mat':adj_mat_list,
    #     'dialogs':dialog_list
    # }

    return examples


def get_parsing_mask(offsets, adj_mat, utterance_list, args):
    """
    args:
        offsets: list(tuple), containing the offset mapping of context, while other tokens in input_ids are annotated with None
        adj_mat: n*n numpy array, n is the number of utterances
        utterance_list
    """
    parsing_mask = np.zeros((len(offsets), len(offsets)), dtype = np.int32)
    context_start_index = 0
    context_end_index = len(offsets) - 1

    while offsets[context_start_index] is None:
        context_start_index += 1

    while offsets[context_end_index] is None:
        context_end_index -= 1
    context_end_index += 1

    # process the question
    parsing_mask[:context_start_index, :] = 1
    if args.relative_mode == 'bi':
        parsing_mask[context_start_index :, : context_start_index] = -1
    elif args.relative_mode == 'symm':
        parsing_mask[context_start_index :, : context_start_index] = 1

    # map each token in context span to the utterance, and map each utterance's start_token_index and end_token_index to corresponded token
    token_to_utterance_mapping = []
    cur_utterance = 0
    utterance_list[cur_utterance].start_token_index = context_start_index
    for i in range(context_start_index, context_end_index):
        while offsets[i][0] > utterance_list[cur_utterance].end_char_index:
            cur_utterance += 1
            utterance_list[cur_utterance].start_token_index = i
        utterance_list[cur_utterance].end_token_index = i
        token_to_utterance_mapping.append(cur_utterance)

    start_utterance = token_to_utterance_mapping[0]
    end_utterance = token_to_utterance_mapping[-1]

    assert utterance_list[start_utterance].start_token_index == context_start_index
    assert utterance_list[end_utterance].end_token_index == context_end_index - 1

    # broadcast adj_mat[i,j] to corresponded indices in parsing_mask
    for j in range(start_utterance, end_utterance + 1):
        for i in range(j, end_utterance + 1):
            parsing_mask[utterance_list[i].start_token_index: utterance_list[i].end_token_index+1, utterance_list[j].start_token_index: utterance_list[j].end_token_index+1] = \
                (int(adj_mat[i,j]) if int(adj_mat[i,j]) <= args.max_hop else 0)
            if args.relative_mode == 'bi':
                parsing_mask[utterance_list[j].start_token_index: utterance_list[j].end_token_index+1, utterance_list[i].start_token_index: utterance_list[i].end_token_index+1] = \
                (-int(adj_mat[j,i]) if int(adj_mat[j,i]) <= args.max_hop else 0)
            elif args.relative_mode == 'symm':
                parsing_mask[utterance_list[j].start_token_index: utterance_list[j].end_token_index+1, utterance_list[i].start_token_index: utterance_list[i].end_token_index+1] = \
                (int(adj_mat[j,i]) if int(adj_mat[j,i]) <= args.max_hop else 0)

    if args.relative_mode == 'bi':
        parsing_mask = parsing_mask + args.max_hop + 1

    # process the paddings
    parsing_mask[context_end_index:, :] = 0
    parsing_mask[:, context_end_index:] = 0

    return parsing_mask
    

    
class QADataset(Dataset):

    def __init__(self, examples, tokenizer, args):
        self.data = self.preprocess(examples, tokenizer, args)
        self.args = args

        self.len = len(self.data['input_ids'])

                
    def preprocess(self, examples, tokenizer, data_args):
        pad_on_right = tokenizer.padding_side == "right" # default true
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length) # cover all sequence lengths
        pad_on_right = True

        tokenized_examples = tokenizer(
                [e['question'] if pad_on_right else e['context'] for e in examples], # default question
                [e['context'] if pad_on_right else e['question'] for e in examples], # default context
                truncation="only_second" if pad_on_right else "only_first", # default only second
                max_length=max_seq_length,
                stride=data_args.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length" if data_args.pad_to_max_length else False,
            )     

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # finished here
        
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        tokenized_examples["example_id"] = []
        tokenized_examples["offset_mapping"] = []
        tokenized_examples['disc_connect_ids'] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            offsets = offset_mapping[i]

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0


            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[sample_index]['answers']
            tokenized_examples["example_id"].append(examples[sample_index]["id"])

            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):#??? why
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)
                
            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"].append([
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(offset_mapping[i])
            ])

            # build the disc connect matrix
            parsing_mask = get_parsing_mask(offsets = tokenized_examples["offset_mapping"][i], 
                                            adj_mat = examples[sample_index]["adj_mat"],
                                            utterance_list = examples[sample_index]['utterance_list'],
                                            args = data_args)
            tokenized_examples['disc_connect_ids'].append(parsing_mask)
        
        return tokenized_examples

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        '''

        :param index:
        :return:
            text_ids:
            token_types:
            label
        '''
        # print(self.data['start_positions'])
        item = {
            'input_ids': torch.LongTensor(self.data['input_ids'][index]),
            'start_positions': self.data['start_positions'][index],
            'end_positions': self.data['end_positions'][index],
            'example_id': self.data['example_id'][index],
            'offset_mapping': self.data['offset_mapping'][index],
            'disc_connect_ids':  torch.LongTensor(self.data['disc_connect_ids'][index]),
            'attention_mask': torch.FloatTensor(self.data['attention_mask'][index])
        }

        return item


def get_dataset_and_examples(split, tokenizer, args):
    examples = read(split, args)
    qadataset = QADataset(examples, tokenizer, args)

    return qadataset, examples

    

if __name__ == '__main__':
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')
    max_hop = 7
    relative_mode = 'bi'
