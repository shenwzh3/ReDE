import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3' # 指定使用的GPU
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


# We use seed = 100 for reproduction of the results reported in the paper.
# seed = 100

import logging

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def seed_everything(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    # path = './saved/IEMOCAP/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_model_dir', type=str, default='')
    parser.add_argument('--bert_tokenizer_dir', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='./data/')

    parser.add_argument('--bert_dim', type = int, default=1024)
    parser.add_argument('--hidden_dim', type = int, default=300)
    parser.add_argument('--mlp_layers', type=int, default=2, help='Number of output mlp layers.')

    parser.add_argument('--max_sent_len', type=int, default=200,
                        help='max content length for each text, if set to 0, then the max length has no constrain')

    parser.add_argument('--no_cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--dataset_name', default='MELD', type= str, help='dataset name, IEMOCAP or MELD or DailyDialog or EmoryNLP')

    parser.add_argument('--windowp', type=int, default=15,
                        help='context window size for constructing edges in graph model for past utterances')

    parser.add_argument('--windowf', type=int, default=0,
                        help='context window size for constructing edges in graph model for future utterances')

    parser.add_argument('--max_hop', type=int, default=7,
                        help='max acceptable number of hops in discourse dependency relations')

    parser.add_argument('--relative_mode', type=str, choices= ['uni', 'bi', 'symm'], default='bi',
                        help='relative embedding mode: uni-directional, bi-directional, symmetric')

    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')

    parser.add_argument('--lr', type=float, default=5e-6, metavar='LR', help='learning rate')


    parser.add_argument('--dropout', type=float, default=0, metavar='dropout', help='dropout rate')

    parser.add_argument('--batch_size', type=int, default=1, metavar='BS', help='batch size')

    parser.add_argument('--epochs', type=int, default=20, metavar='E', help='number of epochs')

    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')

    # parser.add_argument('--grad_accumulate', action='store_true', default=False, help='use gradient accumulation')

    parser.add_argument('--grad_accumulate_step', type=int, default=1, help="accumulate gradient every t steps")

    parser.add_argument('--local_rank', type=int, default=-1)

    parser.add_argument('--seed' , type=int , default = 100)

    args = parser.parse_args()
    seed = args.seed


    if args.local_rank == -1 or args.local_rank == 0:
        print(args)

    seed_everything(seed)

    args.cuda = torch.cuda.is_available() and not args.no_cuda


    if args.cuda and args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        # to use DDP run python -m torch.distributed.launch --nproc_per_node 4 run.py


    if args.cuda:
        if args.local_rank == -1 or args.local_rank == 0:
            print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter()

    # print('building model..')
    # model = BertERC(args, 6)

    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size


    train_loader, valid_loader, test_loader, speaker_vocab, label_vocab, train_size, dev_size, test_size = get_IEMOCAP_loaders(dataset_name=args.dataset_name, batch_size=batch_size, num_workers=0, args = args)
    if args.local_rank == -1 or args.local_rank == 0:
        print('train_size:', train_size)
        print('dev_size:', dev_size)
        print('test_size:', test_size)

    n_classes = len(label_vocab['itos'])

    if args.local_rank == -1 or args.local_rank == 0:
        print('building model..')
    model = RobertaERC(args, n_classes)

    if cuda:
        if args.local_rank == -1:
            print('Mutli-GPU...........')
            model.cuda()
            model = nn.DataParallel(model,device_ids = range(torch.cuda.device_count()))
        else:
            # use distributed parallel
            if args.local_rank == 0:
                print('use DDP...')
            model = model.to(args.local_rank)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)



    loss_function = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = AdamW(model.parameters() , lr=args.lr)



    best_fscore,best_acc, best_loss, best_label, best_pred, best_mask = None,None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []
    best_acc = 63.
    best_fscore = 0.



    cur_best_on_valid = 0.0
    for e in range(n_epochs):
        start_time = time.time()
        if args.local_rank == -1:
            train_loss, train_acc, _, _, train_fscore = train_model(model, loss_function,train_loader, e, cuda,args, optimizer)
            valid_loss, valid_acc, _, _, valid_fscore= eval_model(model, loss_function, valid_loader, e, cuda, args)
            test_loss, test_acc, test_label, test_pred, test_fscore= eval_model(model,loss_function, test_loader,e, cuda, args)
        else:
            train_loss, train_acc, _, _, train_fscore = train_model_DDP(model, loss_function,train_loader, e, cuda, args, train_size,  optimizer)
            valid_loss, valid_acc, _, _, valid_fscore= eval_model_DDP(model, loss_function, valid_loader, e, cuda, dev_size, args)
            test_loss, test_acc, test_label, test_pred, test_fscore= eval_model_DDP(model,loss_function, test_loader,e, cuda, test_size, args)

        all_fscore.append([valid_fscore, test_fscore])

        if valid_fscore > cur_best_on_valid:
            cur_best_on_valid = valid_fscore
            torch.save({'args':args,'model_state_dict':model.state_dict()},'optimal_model.pkl')

        if args.local_rank == -1 or args.local_rank == 0:
            print(
                'epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'. \
                format(e + 1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc,
                    test_fscore, round(time.time() - start_time, 2)))

    if args.tensorboard:
        writer.close()

    # logger.info('finish training!')

    if args.local_rank == -1 or args.local_rank == 0:
        print('Test performance..')
        all_fscore = sorted(all_fscore, key=lambda x: x[0], reverse=True)
        print('Best F-Score based on validation:', all_fscore[0][1])
        print('Best F-Score based on test:', max([f[1] for f in all_fscore]))

        with open('results.txt','a') as f:
            f.write('Test performance..\n' + 'Best F-Score based on validation: ' + str(all_fscore[0][1])+'\n' + 'Best F-Score based on test: ' + str(max([f[1] for f in all_fscore])) + '\n')

