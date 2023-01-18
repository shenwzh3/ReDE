import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np, itertools, random, copy, math
from transformers import BertModel, BertConfig
from transformers import AutoTokenizer, AutoModelWithLMHead
from modeling_DisRoberta import RobertaModel_disc
from modeling_DisRoberta_sm import RobertaModel_disc_singleemb
from transformers import RobertaModel


# For methods and models related to DialogueGCN jump to line 516
class RobertaERC(nn.Module):

    def __init__(self, args, num_class):
        super().__init__()
        self.args = args
        # gcn layer

        self.dropout = nn.Dropout(args.dropout)

        self.roberta = RobertaModel_disc_singleemb.from_pretrained(args.bert_model_dir)
        # self.roberta = RobertaModel.from_pretrained(args.home_dir + args.bert_model_dir)
        # self.bert = BertModel.from_pretrained('/ERC/models/chinese_wwm_ext_pytorch/bert_pytorch.bin')
        in_dim = args.bert_dim

        # output mlp layers
        layers = [nn.Linear(in_dim, args.hidden_dim), nn.ReLU()]
        for _ in range(args.mlp_layers - 1):
            layers += [nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU()]
        layers += [nn.Linear(args.hidden_dim, num_class)]

        self.out_mlp = nn.Sequential(*layers)

    def forward(self, content_ids, disc_connect_ids, attention_mask):
        # lastHidden = self.roberta(content_ids, disc_connect_ids = disc_connect_ids, token_type_ids = token_type_ids)[0][:,1,:] #(N , D)
        lastHidden = self.roberta(content_ids, disc_connect_ids=disc_connect_ids, attention_mask=attention_mask)[0][:, 1, :]  # (N, D)
        # lastHidden = self.roberta(content_ids)[1]

        ##### 这是取last hid的方法
        # lastHiddenN = [lastHidden[i,seq_len[i]-utterance_len[i]:seq_len[i],:] for i in range(len(utterance_len))]
        # # pooling 一下
        # text_feature = [torch.max(lastHiddenN[i],dim=0).values for i in range(len(lastHiddenN))]
        # t = [np.array(text_feature[i].detach().cpu()) for i in range(len(text_feature))]
        # final_feature = torch.from_numpy(np.array(t))
        # final_feature = final_feature.cuda()
        ##### 这是取last hid的方法

        final_feature = self.dropout(lastHidden)

        # pooling

        outputs = self.out_mlp(final_feature)  # (N, D)

        return outputs
