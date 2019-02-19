# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from config import Config
from model import TextCNN
from data import TextDataset
import argparse

import torchtext
from torchtext.data import Field

torch.manual_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epoch', type=int, default=2)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--out_channel', type=int, default=100)
parser.add_argument('--label_num', type=int, default=3)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()


torch.manual_seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)

# Create the configuration
config = Config(sentence_max_size=50,
                batch_size=args.batch_size,
                word_num=25000,
                label_num= args.label_num,
                learning_rate=args.lr,
                cuda=args.gpu,
                epoch=args.epoch,
                out_channel=args.out_channel)

             

# load SST dataset
def sst(text_field, label_field,  **kargs):
    train_data, dev_data, test_data = torchtext.datasets.SST.splits(text_field, label_field)#, fine_grained=True)
    text_field.build_vocab(train_data, dev_data, test_data)
    label_field.build_vocab(train_data, dev_data, test_data)
    train_iter, dev_iter, test_iter = torchtext.data.BucketIterator.splits(
                                        (train_data, dev_data, test_data), 
                                        batch_sizes=(args.batch_size, 
                                                     len(dev_data), 
                                                     len(test_data)),
                                        **kargs)

    return train_iter, dev_iter, test_iter 
             
text_field = torchtext.data.Field(lower=True)
label_field = torchtext.data.Field(sequential=False)
training_iter, valid_iter, test_iter= sst(text_field, label_field,device=-1, repeat=False)
#training_set = TextDataset(path='data/train',)

#training_iter = data.DataLoader(dataset=training_set,
#                                batch_size=config.batch_size)

model = TextCNN(config)

if torch.cuda.is_available():
    model.cuda()
    embeds = embeds.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=config.lr,momentum=0.9)

count = 0
loss_sum = 0
 
# Train the model
for epoch in range(config.epoch):
    for batch in training_iter:
        data = batch.text;
        label = batch.label;
        data.data.t_(), label.data.sub_(1)

        if config.cuda and torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()
        #input_data = embeds(autograd.Variable(data))
        input_data = autograd.Variable(data)
        out = model(input_data.unsqueeze(1))
        loss = criterion(out, autograd.Variable(label))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            norm = model.linear1.weight.norm()
            if norm>3 :
              model.linear1.weight =torch.nn.Parameter((3.0/norm)* model.linear1.weight)
            loss_sum += loss
            count += 1
    
            if count % 100 == 0:
                print("epoch", epoch, end='  ')
                print("The loss is: %.5f" % (loss_sum/100))
    
                loss_sum = 0
                count = 0
                corrects = (torch.max(out, 1)[1].view(label.size()).data == label.data).sum()
                accuracy = float(corrects)/config.batch_size * 100.0
                print("accuracy=%f, "%( accuracy))
        
       
    # save the model in every epoch
    #model.save('checkpoints/epoch{}.ckpt'.format(epoch))
    
    
confusionMatrix =[]
for i in range(config.label_num):
  confusionMatrix.append([])
  for j in range(config.label_num):
    confusionMatrix[i].append(0)

loss_total = 0
count_valid = 0
for batch  in valid_iter:
    data = batch.text;
    label = batch.label;
    data.data.t_(), label.data.sub_(1)
    if config.cuda and torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()
    with torch.no_grad():
        #input_data = embeds(autograd.Variable(data))
        input_data = autograd.Variable(data)
        out = model(input_data.unsqueeze(1))
        _,pred = torch.max(out,1)
        loss = criterion(out, autograd.Variable(label))
        loss_total +=loss
        count_valid+=1
        for p,l in zip(list(pred),list(label)):
            confusionMatrix[p][l]+=1

import pprint            
pprint.pprint(confusionMatrix)
print("acc :",
        sum([confusionMatrix[i][i] for i in range(config.label_num)] )/
        sum([sum(confusionMatrix[i]) for i in range(config.label_num)]))
print("loss:",loss_total/count_valid)