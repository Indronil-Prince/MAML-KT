#DKT
# @Original Author: jarvis.zhang
# @Last Modified by: Indronil-Prince

import torch
import torch.nn as nn
import numpy as np
import itertools


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.rnn = nn.RNN(input_dim,
                          hidden_dim,
                          layer_dim,
                          batch_first=True,
                          nonlinearity='tanh')
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.sig = nn.Sigmoid()
        self.device = device

    def forward(self, x):  # shape of input: [batch_size, length, questions * 2]
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=self.device)  # shape: [num_layers * num_directions, batch_size, hidden_size]
        out, hn = self.rnn(x, h0)  # shape of out: [batch_size, length, hidden_size]
        res = self.sig(self.fc(out))  # shape of res: [batch_size, length, question]
        return res

#Readdata
class DataReader():
    def __init__(self, train_path, test_path, maxstep, numofques):
        self.train_path = train_path
        self.test_path = test_path
        self.maxstep = maxstep
        self.numofques = numofques

    def getData(self, file_path):
        data = []
        with open(file_path, 'r') as file:
            for len, ques, ans in itertools.zip_longest(*[file] * 3):
                len = int(len.strip().strip(','))
                ques = [int(q) for q in ques.strip().strip(',').split(',')]
                ans = [int(a) for a in ans.strip().strip(',').split(',')]
                slices = len//self.maxstep + (1 if len % self.maxstep > 0 else 0)
                for i in range(slices):
                    temp = temp = np.zeros(shape=[self.maxstep, 2 * self.numofques])
                    if len > 0:
                        if len >= self.maxstep:
                            steps = self.maxstep
                        else:
                            steps = len
                        for j in range(steps):
                            if ans[i*self.maxstep + j] == 1:
                                temp[j][ques[i*self.maxstep + j]] = 1
                            else:
                                temp[j][ques[i*self.maxstep + j] + self.numofques] = 1
                        len = len - self.maxstep
                    data.append(temp.tolist())
            # print('done: ' + str(np.array(data).shape))
        return data

    def getTrainData(self):
        # print('loading train data...')
        trainData = self.getData(self.train_path)
        return np.array(trainData)

    def getTestData(self):
        # print('loading test data...')
        testData = self.getData(self.test_path)
        return np.array(testData)

#Dataloader
# -*- coding: utf-8 -*-
# @Author: jarvis.zhang
# @Date:   2020-05-08 16:21:21
# @Last Modified by:   jarvis.zhang
# @Last Modified time: 2020-05-10 11:47:28
import torch
import torch.utils.data as Data

dts = 2017
setid = 1
stu = 50
que = 30

def getDataLoader(batch_size, num_of_questions, max_step, que, setid, stu):
    handle = DataReader('C:\\Users\\ibpri\\Downloads\\Knowledge Tracing\\DKVMN-No-ID\\DKVMN-main\\dataset\\assist'+str(dts)+'\\Test - '+str(stu)+'\\Set'+str(setid)+'\\assist'+str(dts)+'_train_new_'+str(stu)+'_set'+str(setid)+'.txt',
                        'C:\\Users\\ibpri\\Downloads\\Knowledge Tracing\\DKVMN-No-ID\\DKVMN-main\\dataset\\assist'+str(dts)+'\\Test - '+str(stu)+'\\Set'+str(setid)+'\\assist'+str(dts)+'_test_new_'+str(que)+'_set'+str(setid)+'.txt', max_step,
                        num_of_questions)
    dtrain = torch.tensor(handle.getTrainData().astype(float).tolist(),
                          dtype=torch.float32)
    dtest = torch.tensor(handle.getTestData().astype(float).tolist(),
                         dtype=torch.float32)
    trainLoader = Data.DataLoader(dtrain, batch_size=batch_size, shuffle=True)
    testLoader = Data.DataLoader(dtest, batch_size=batch_size, shuffle=False)
    return trainLoader, testLoader

#eval
# -*- coding: utf-8 -*-
# @Author: jarvis.zhang
# @Date:   2020-05-09 13:42:11
# @Last Modified by:   jarvis.zhang
# @Last Modified time: 2020-05-10 13:33:06
import tqdm
import torch
import logging

import torch.nn as nn
from sklearn import metrics

logger = logging.getLogger('main.eval')


def performance(ground_truth, prediction):
    fpr, tpr, thresholds = metrics.roc_curve(ground_truth.detach().cpu().numpy(),
                                             prediction.detach().cpu().numpy())
    auc = metrics.auc(fpr, tpr)

    acc = metrics.accuracy_score(ground_truth.detach().cpu().numpy(),
                                 torch.round(prediction).detach().cpu().numpy())

    f1 = metrics.f1_score(ground_truth.detach().cpu().numpy(),
                          torch.round(prediction).detach().cpu().numpy())
    recall = metrics.recall_score(ground_truth.detach().cpu().numpy(),
                                  torch.round(prediction).detach().cpu().numpy())
    precision = metrics.precision_score(
        ground_truth.detach().cpu().numpy(),
        torch.round(prediction).detach().cpu().numpy())
    
    logger.info('auc: ' + str(auc) + 'acc: ' + str(acc) + ' f1: ' + str(f1) + ' recall: ' +
                str(recall) + ' precision: ' + str(precision))
    print(
        # 'auc: ' + str(auc) + 
        'acc: ' + str(acc) 
        # + ' f1: ' + str(f1) + ' recall: ' +
                # str(recall) + ' precision: ' + str(precision)
                )


class lossFunc(nn.Module):
    def __init__(self, num_of_questions, max_step, device):
        super(lossFunc, self).__init__()
        self.crossEntropy = nn.BCELoss()
        self.num_of_questions = num_of_questions
        self.max_step = max_step
        self.device = device

    def forward(self, pred, batch):
        loss = 0
        prediction = torch.tensor([], device=self.device)
        ground_truth = torch.tensor([], device=self.device)
        for student in range(pred.shape[0]):
            delta = batch[student][:, 0:self.num_of_questions] + batch[
                student][:, self.num_of_questions:]  # shape: [length, questions]
            temp = pred[student][:self.max_step - 1].mm(delta[1:].t())
            index = torch.tensor([[i for i in range(self.max_step - 1)]],
                                 dtype=torch.long, device=self.device)
            p = temp.gather(0, index)[0]
            a = (((batch[student][:, 0:self.num_of_questions] -
                   batch[student][:, self.num_of_questions:]).sum(1) + 1) //
                 2)[1:]
            for i in range(len(p) - 1, -1, -1):
                if p[i] > 0:
                    p = p[:i + 1]
                    a = a[:i + 1]
                    break
            loss += self.crossEntropy(p, a)
            prediction = torch.cat([prediction, p])
            ground_truth = torch.cat([ground_truth, a])
        return loss, prediction, ground_truth


def train_epoch(model, trainLoader, optimizer, loss_func, device):
    model.to(device)
    for batch in tqdm.tqdm(trainLoader, desc='Training:    ', mininterval=2):
        batch = batch.to(device)
        pred = model(batch)
        loss, prediction, ground_truth = loss_func(pred, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, optimizer


def test_epoch(model, testLoader, loss_func, device):
    model.to(device)
    ground_truth = torch.tensor([], device=device)
    prediction = torch.tensor([], device=device)
    for batch in tqdm.tqdm(testLoader, desc='Testing:     ', mininterval=2):
        batch = batch.to(device)
        pred = model(batch)
        loss, p, a = loss_func(pred, batch)
        prediction = torch.cat([prediction, p])
        ground_truth = torch.cat([ground_truth, a])
    performance(ground_truth, prediction)

#run
# -*- coding: utf-8 -*-
# @Author: jarvis.zhang
# @Date:   2020-05-09 21:50:46
# @Last Modified by:   jarvis.zhang
# @Last Modified time: 2020-05-10 13:20:09
"""
Usage:
    run.py (rnn|sakt) --hidden=<h> [options]

Options:
    --length=<int>                      max length of question sequence [default: 50]
    --questions=<int>                   num of question [default: 124]
    --lr=<float>                        learning rate [default: 0.001]
    --bs=<int>                          batch size [default: 64]
    --seed=<int>                        random seed [default: 59]
    --epochs=<int>                      number of epochs [default: 10]
    --cuda=<int>                        use GPU id [default: 0]
    --hidden=<int>                      dimention of hidden state [default: 128]
    --layers=<int>                      layers of rnn or transformer [default: 1]
    --heads=<int>                       head number of transformer [default: 8]
    --dropout=<float>                   dropout rate [default: 0.1]
    --model=<string>                    model type
"""

import os
import random
import logging
import torch

import torch.optim as optim
import numpy as np

from datetime import datetime

def setup_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#args = docopt(__doc__)
length = 50
questions = 200
lr = 0.001
bs = 64
seed = 59
epochs = 10
cuda = 0
hidden = 128
layers = 1
heads = 8
dropout = 0.1
# if args['rnn']:
model_type = 'RNN'
# elif args['sakt']:
# model_type = 'SAKT'

# logger = logging.getLogger('main')
# logger.setLevel(level=logging.DEBUG)
# date = datetime.now()
# handler = logging.FileHandler(
#     f'log/{date.year}_{date.month}_{date.day}_{model_type}_result.log')
# handler.setLevel(logging.INFO)
# formatter = logging.Formatter(
#     '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)

#logger.info(list(args.items()))

setup_seed(seed)

if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print("Assist"+str(dts)+" DKT with "+str(stu)+" Students\n")
for set in range(5,6):
    print(f"---- RUN for {set} Student Set----\n")
    trainLoader, testLoader = getDataLoader(bs, questions, length, 20, set, stu)

    if model_type == 'RNN':
        model = RNNModel(questions * 2, hidden, layers, questions, device)
        # elif model_type == 'SAKT':
        #     from model.SAKT.model import SAKTModel
        #     model = SAKTModel(heads, length, hidden, questions, dropout)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_func = lossFunc(questions, length, device)
        for epoch in range(epochs):
            print('epoch: ' + str(epoch))
            model, optimizer = train_epoch(model, trainLoader, optimizer,
                                            loss_func, device)
            logger.info(f'epoch {epoch}')

    for que in range(3,31):
        trainLoader, testLoader = getDataLoader(bs, questions, length, que, set, stu)
        print("--------------------------------Questions = "+str(que)+" ----------------------------------------")
        test_epoch(model, testLoader, loss_func, device)

