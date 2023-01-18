import numpy as np, argparse, time, pickle, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import get_DialogRE_loaders
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, \
    precision_recall_fscore_support
from transformers.utils.dummy_pt_objects import AutoModelForTableQuestionAnswering
from utils import person_embed
from tqdm import tqdm
import torch.distributed as dist
from evaluate_utils import evaluate


def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]


def model_step(model, data, loss_function, args, train = False):
    content_ids, label,  parsing_mask, attention_mask = data
    content_ids = content_ids.to(args.local_rank)
    label = label.to(args.local_rank)
    parsing_mask = parsing_mask.to(args.local_rank)
    attention_mask = attention_mask.to(args.local_rank)
    # print(cnt)
    # cnt +=1 
    # print(content_ids.size())
    # print(parsing_mask.size())
    # print('content_ids',content_ids.size())
    # print('parsing_mask', parsing_mask.size())
    # print('label',label.size())

    log_prob = model(content_ids, parsing_mask, attention_mask) # (B, C)
    # print(label)
    loss = loss_function(log_prob, label)
    loss = loss.mean()

    if train:
        loss = loss / args.grad_accumulate_step
        loss.backward()


    return loss.detach(), label.detach(), log_prob.detach()


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
    preds = list(1 / (1 + np.exp(-preds)))
    labels = distributed_concat(torch.cat(labels, dim=0), data_size).cpu().numpy()
    labels = list(labels)

    precision, recall, f_1, preds, labels = evaluate(preds, labels)
    # if args.local_rank == 0:
    #     print(len(preds))
    #     print(len(labels))

    # print(preds.tolist())
    # print(labels.tolist())

    return avg_loss, precision, recall, f_1, labels, preds




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
        preds = list(1 / (1 + np.exp(-preds)))
        labels = distributed_concat(torch.cat(labels, dim=0), data_size).cpu().numpy()
        labels = list(labels)

        precision, recall, f_1, preds, labels = evaluate(preds, labels)
        # if args.local_rank == 0:
        #     print(len(preds))
        #     print(len(labels))

        # print(preds.tolist())
        # print(labels.tolist())

    return avg_loss, precision, recall, f_1, labels, preds
