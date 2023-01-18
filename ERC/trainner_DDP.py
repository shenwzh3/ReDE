import numpy as np, argparse, time, pickle, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import IEMOCAPDataset
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, \
    precision_recall_fscore_support
from transformers.utils.dummy_pt_objects import AutoModelForTableQuestionAnswering
from utils import person_embed
from tqdm import tqdm
import torch.distributed as dist


def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]


def model_step(model, data, loss_function, args, train = False):
    content_ids, parsing_mask, label, utterance_len, seq_len, attention_mask = data

    content_ids = content_ids.to(args.local_rank)
    label = label.to(args.local_rank)
    utterance_len = utterance_len.to(args.local_rank)
    seq_len = seq_len.to(args.local_rank)
    parsing_mask = parsing_mask.to(args.local_rank)
    attention_mask = attention_mask.to(args.local_rank)
    # print(cnt)
    # cnt +=1 
    # print(content_ids.size())
    # print(parsing_mask.size())
    # print('content_ids',content_ids.size())
    # print('parsing_mask', parsing_mask.size())
    # print('label',label.size())

    log_prob = model(content_ids, parsing_mask, attention_mask) # (B, D)
    # print(label)
    loss = loss_function(log_prob, label)
    loss = loss.mean()

    if train:
        loss = loss / args.grad_accumulate_step
        loss.backward()

    label = label
    pred = torch.argmax(log_prob, 1)

    return loss.detach(), label.detach(), pred.detach()


def train_model_DDP(model, loss_function, dataloader,  epoch, cuda, args, data_size, optimizer=None):

    assert optimizer != None
    model.train()
    step = 0
    
    dataloader.sampler.set_epoch(epoch)
    # print('----------------------------------------------')

    tr_loss = torch.tensor(0.0).to(args.local_rank)
    total_loss_scalar = 0.0
    preds, labels = [], []

    # for data in tqdm(dataloader):
    for data in dataloader:

        loss, label, pred = model_step(model, data, loss_function, args, train=True)
        preds.append(pred)
        labels.append(label)

        if (step+1) % args.grad_accumulate_step != 0:
            with model.no_sync():
                tr_loss += loss
        else:
            tr_loss += loss

        if (step+1) % args.grad_accumulate_step == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # if args.tensorboard:
            #     for param in model.named_parameters():
            #         writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()
            optimizer.zero_grad()

        step += 1
        
    optimizer.zero_grad()
    total_loss_scalar += tr_loss.item()
    avg_loss = total_loss_scalar / step

    # begin eval
    preds = distributed_concat(torch.cat(preds, dim=0), data_size).cpu().numpy()
    labels = distributed_concat(torch.cat(labels, dim=0), data_size).cpu().numpy()

    # if args.local_rank == 0:
    #     print(len(preds))
    #     print(len(labels))

    # print(preds.tolist())
    # print(labels.tolist())
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    if args.dataset_name in ['IEMOCAP', 'MELD', 'EmoryNLP']:
        avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)
    else:
        avg_fscore = round(f1_score(labels, preds, average='micro', labels=list(range(1, 7))) * 100, 2)

    return avg_loss, avg_accuracy, labels, preds, avg_fscore




def eval_model_DDP(model, loss_function, dataloader,  epoch, cuda, data_size, args):

    model.eval()
    
    tr_loss = torch.tensor(0.0).to(args.local_rank)
    total_loss_scalar = 0.0
    preds, labels = [], []
    # cnt = 0
    # print('----------------------------------------------')
    step = 0
    with torch.no_grad():
        # for data in tqdm(dataloader):
        for data in dataloader:

            loss, label, pred = model_step(model, data, loss_function, args)
            preds.append(pred)
            labels.append(label)
            tr_loss += loss

            step += 1

        total_loss_scalar += tr_loss.item()
        avg_loss = total_loss_scalar / step

        # begin eval
        preds = distributed_concat(torch.cat(preds, dim=0), data_size).cpu().numpy()
        labels = distributed_concat(torch.cat(labels, dim=0), data_size).cpu().numpy()

        # if args.local_rank == 0:
        #     print(len(preds))
        #     print(len(labels))

        # print(preds.tolist())
        # print(labels.tolist())
        avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
        if args.dataset_name in ['IEMOCAP', 'MELD', 'EmoryNLP']:
            avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)
        else:
            avg_fscore = round(f1_score(labels, preds, average='micro', labels=list(range(1, 7))) * 100, 2)


    return avg_loss, avg_accuracy, labels, preds, avg_fscore
