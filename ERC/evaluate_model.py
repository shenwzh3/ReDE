import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3' # 指定使用的GPU
import numpy as np, argparse, time, pickle, random
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import IEMOCAPDataset
from model import RobertaERC
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, \
    precision_recall_fscore_support
from trainer import  train_model, eval_model
from trainner_DDP import train_model_DDP, eval_model_DDP
from dataset import IEMOCAPDataset
from dataloader import get_IEMOCAP_loaders
from transformers import AdamW
import warnings
warnings.filterwarnings('ignore')



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path',type=str,default='optimal_model.pkl')
    path = parser.parse_args().path

    optimal_model = torch.load(path)

    args = optimal_model['args']
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    cuda = args.cuda
    if args.cuda and args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        # to use DDP run python -m torch.distributed.launch --nproc_per_node 4 run.py

    if args.cuda:
        if args.local_rank == -1 or args.local_rank == 0:
            print('Running on GPU')
    else:
        print('Running on CPU')

    train_loader, valid_loader, test_loader, speaker_vocab, label_vocab, train_size, dev_size, test_size = get_IEMOCAP_loaders(dataset_name=args.dataset_name, batch_size=args.batch_size, num_workers=0, args=args)
    n_classes = len(label_vocab['itos'])
    model = RobertaERC(args, n_classes)

    model.load_state_dict(optimal_model['model_state_dict'])
    model.eval()
    loss_function = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = AdamW(model.parameters() , lr=args.lr)

    if args.local_rank == -1:
        valid_loss, valid_acc, _, _, valid_fscore = eval_model(model, loss_function, valid_loader, 0, cuda, args)
        test_loss, test_acc, test_label, test_pred, test_fscore = eval_model(model, loss_function, test_loader, 0, cuda, args)
    else:

        valid_loss, valid_acc, _, _, valid_fscore = eval_model_DDP(model, loss_function, valid_loader, 0, cuda,dev_size, args)
        test_loss, test_acc, test_label, test_pred, test_fscore = eval_model_DDP(model, loss_function, test_loader, 0,cuda, test_size, args)
    print(' valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}'. \
            format(valid_loss, valid_acc, valid_fscore, test_loss, test_acc,
                   test_fscore))