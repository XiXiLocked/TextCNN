# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule


class TextCNN(BasicModule):

    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.config = config
        self.out_channel = config.out_channel
        
        D_embed = config.word_embedding_dimension
        CH_out = config.out_channel
        size_sentence = config.sentence_max_size
        self.embedding = nn.Embedding(config.word_num, D_embed)

        #self.conv2 = nn.Conv2d(1, 1, (2, config.word_embedding_dimension))
        self.conv3 = nn.Conv2d(1, CH_out, (3, D_embed))
        self.conv4 = nn.Conv2d(1, CH_out, (4, D_embed))
        self.conv5 = nn.Conv2d(1, CH_out, (5, D_embed))
        #self.conv6 = nn.Conv2d(1, 1, (6, config.word_embedding_dimension))
        #self.Max2_pool = nn.MaxPool2d((self.config.sentence_max_size-2+1, 1))
        self.Max3_pool = nn.MaxPool2d((size_sentence-3+1, 1))
        self.Max4_pool = nn.MaxPool2d((size_sentence-4+1, 1))
        self.Max5_pool = nn.MaxPool2d((size_sentence-5+1, 1))
        #self.Max6_pool = nn.MaxPool2d((self.config.sentence_max_size-6+1, 1))
        self.dropout = nn.Dropout(0.5)
        self.linear1 = nn.Linear(3*CH_out, config.label_num)

    def forward(self, x):
        batch = x.shape[0]
        x = self.embedding(x)
        # Convolution
        #x2 = F.relu(self.conv2(x))

        x3 = F.relu(self.conv3(x))
        x4 = F.relu(self.conv4(x))
        x5 = F.relu(self.conv5(x))
        #x6 = F.relu(self.conv6(x))
        
        # Pooling
        #x2 = self.Max2_pool(x2)
        x3 = self.Max3_pool(x3)
        x4 = self.Max4_pool(x4)
        x5 = self.Max5_pool(x5)

        #x6 = self.Max6_pool(x6)

        # capture and concatenate the features
        #x = torch.cat((x2,x3,x4,x5,x6), -1)
        x = torch.cat((
        x3.view(-1,self.out_channel),
        x4.view(-1,self.out_channel),
        x5.view(-1,self.out_channel)), 1)
        #x = x.view(batch, 1, -1)
        x = self.dropout(x)

        # project the features to the labels
        x = self.linear1(x)
        x = x.view(-1, self.config.label_num)

        return x


if __name__ == '__main__':
    print('running the TextCNN...')