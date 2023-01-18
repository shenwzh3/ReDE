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


def train_model(model, loss_function, dataloader,  epoch, cuda, args, optimizer=None):
    losses, preds, labels = [], [], []
    scores, vids = [], []


    assert optimizer != None
    model.train()
    step = 0
    
    # print('----------------------------------------------')
    # for data in tqdm(dataloader):
    for data in dataloader:

        # text_ids, text_feature, speaker_ids, labels, umask = [d.cuda() for d in data] if cuda else data
        content_ids, parsing_mask, label, utterance_len, seq_len, attention_mask = data
        # speaker_vec = person_embed(speaker_ids, person_vec)
        if cuda:
            content_ids = content_ids.cuda()
            label = label.cuda()
            utterance_len = utterance_len.cuda()
            seq_len = seq_len.cuda()
            parsing_mask = parsing_mask.cuda()
            attention_mask = attention_mask.cuda()
        # print(cnt)
        # cnt +=1 
        # print(content_ids.size())
        # print(parsing_mask.size())
        # print('content_ids',content_ids.size())
        # print('parsing_mask', parsing_mask.size())
        # print('label',label.size())



        log_prob = model(content_ids, parsing_mask, attention_mask)
        # print(label)
        loss = loss_function(log_prob, label)
        loss = loss / args.grad_accumulate_step


        label = label.cpu().numpy()
        pred = torch.argmax(log_prob, 1).cpu().numpy()
        preds.append(pred)
        labels.append(label)
        losses.append(loss.item())


        loss_val = loss.item()
        loss.backward()

        if (step+1) % args.grad_accumulate_step == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()
            optimizer.zero_grad()

        step += 1

    

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []

    # print(preds.tolist())
    # print(labels.tolist())
    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    if args.dataset_name in ['IEMOCAP', 'MELD', 'EmoryNLP']:
        avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)
    else:
        avg_fscore = round(f1_score(labels, preds, average='micro', labels=list(range(1, 7))) * 100, 2)

    return avg_loss, avg_accuracy, labels, preds, avg_fscore




def eval_model(model, loss_function, dataloader,  epoch, cuda, args):
    losses, preds, labels = [], [], []
    scores, vids = [], []


    model.eval()
    
    # cnt = 0
    # print('----------------------------------------------')
    with torch.no_grad():
        # for data in tqdm(dataloader):
        for data in dataloader:

            content_ids, parsing_mask, label, utterance_len, seq_len, attention_mask = data

            if cuda:
                content_ids = content_ids.cuda()
                label = label.cuda()
                utterance_len = utterance_len.cuda()
                seq_len = seq_len.cuda()
                parsing_mask = parsing_mask.cuda()
                attention_mask = attention_mask.cuda()
            # print(cnt)
            # cnt +=1 
            # print(content_ids.size())
            # print(parsing_mask.size())
            # print('content_ids',content_ids.size())
            # print('parsing_mask', parsing_mask.size())
            # print('label',label.size())



            log_prob = model(content_ids, parsing_mask, attention_mask)
            # print(label)
            loss = loss_function(log_prob, label)


            label = label.cpu().numpy()
            pred = torch.argmax(log_prob, 1).cpu().numpy()
            preds.append(pred)
            labels.append(label)
            losses.append(loss.item())

        if preds != []:
            preds = np.concatenate(preds)
            labels = np.concatenate(labels)
        else:
            return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []

        # print(preds.tolist())
        # print(labels.tolist())
        avg_loss = round(np.sum(losses) / len(losses), 4)
        avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
        if args.dataset_name in ['IEMOCAP', 'MELD', 'EmoryNLP']:
            avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)
        else:
            avg_fscore = round(f1_score(labels, preds, average='micro', labels=list(range(1, 7))) * 100, 2)

    return avg_loss, avg_accuracy, labels, preds, avg_fscore
