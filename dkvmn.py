#Model
import torch
import torch.nn as nn
from model.memory import DKVMN


class MODEL(nn.Module):

    def __init__(self, n_question, batch_size, q_embed_dim, qa_embed_dim, memory_size, final_fc_dim):
        super(MODEL, self).__init__()
        self.n_question = n_question
        self.batch_size = batch_size
        self.q_embed_dim = q_embed_dim
        self.qa_embed_dim = qa_embed_dim
        self.memory_size = memory_size
        self.memory_key_state_dim = q_embed_dim
        self.memory_value_state_dim = qa_embed_dim
        self.final_fc_dim = final_fc_dim

        self.read_embed_linear = nn.Linear(self.memory_value_state_dim + self.memory_key_state_dim, self.final_fc_dim, bias=True)
        self.predict_linear = nn.Linear(self.final_fc_dim, 1, bias=True)
        self.init_memory_key = nn.Parameter(torch.randn(self.memory_size, self.memory_key_state_dim))
        nn.init.kaiming_normal_(self.init_memory_key)
        self.init_memory_value = nn.Parameter(torch.randn(self.memory_size, self.memory_value_state_dim))
        nn.init.kaiming_normal_(self.init_memory_value)

        self.mem = DKVMN(memory_size=self.memory_size,
                         memory_key_state_dim=self.memory_key_state_dim,
                         memory_value_state_dim=self.memory_value_state_dim, init_memory_key=self.init_memory_key)

        self.q_embed = nn.Embedding(self.n_question + 1, self.q_embed_dim, padding_idx=0)
        self.qa_embed = nn.Embedding(2 * self.n_question + 1, self.qa_embed_dim, padding_idx=0)

    def init_params(self):
        nn.init.kaiming_normal_(self.predict_linear.weight)
        nn.init.kaiming_normal_(self.read_embed_linear.weight)
        nn.init.constant_(self.read_embed_linear.bias, 0)
        nn.init.constant_(self.predict_linear.bias, 0)

    def init_embeddings(self):
        nn.init.kaiming_normal_(self.q_embed.weight)
        nn.init.kaiming_normal_(self.qa_embed.weight)

    def forward(self, q_data, qa_data, target):
        batch_size = q_data.shape[0]
        seqlen = q_data.shape[1]
        q_embed_data = self.q_embed(q_data)
        qa_embed_data = self.qa_embed(qa_data)

        memory_value = nn.Parameter(torch.cat([self.init_memory_value.unsqueeze(0) for _ in range(batch_size)], 0).data)
        self.mem.init_value_memory(memory_value)

        slice_q_embed_data = torch.chunk(q_embed_data, seqlen, 1)
        slice_qa_embed_data = torch.chunk(qa_embed_data, seqlen, 1)

        value_read_content_l = []
        input_embed_l = []
        for i in range(seqlen):
            # Attention
            q = slice_q_embed_data[i].squeeze(1)
            correlation_weight = self.mem.attention(q)

            # Read Process
            read_content = self.mem.read(correlation_weight)
            value_read_content_l.append(read_content)
            input_embed_l.append(q)

            # Write Process
            qa = slice_qa_embed_data[i].squeeze(1)
            self.mem.write(correlation_weight, qa)

        all_read_value_content = torch.cat([value_read_content_l[i].unsqueeze(1) for i in range(seqlen)], 1)
        input_embed_content = torch.cat([input_embed_l[i].unsqueeze(1) for i in range(seqlen)], 1)

        predict_input = torch.cat([all_read_value_content, input_embed_content], 2)
        read_content_embed = torch.tanh(self.read_embed_linear(predict_input.view(batch_size * seqlen, -1)))

        pred = self.predict_linear(read_content_embed)
        target_1d = target.view(-1, 1)  # [batch_size * seq_len, 1]
        mask = target_1d.ge(1)  # [batch_size * seq_len, 1]
        pred_1d = pred.view(-1, 1)  # [batch_size * seq_len, 1]

        filtered_pred = torch.masked_select(pred_1d, mask)
        filtered_target = torch.masked_select(target_1d, mask) - 1
        loss = torch.nn.functional.binary_cross_entropy_with_logits(filtered_pred, filtered_target.float())

        return loss, torch.sigmoid(filtered_pred), filtered_target.float()

#Memory
import torch
from torch import nn


class DKVMNHeadGroup(nn.Module):
    def __init__(self, memory_size, memory_state_dim, is_write):
        super(DKVMNHeadGroup, self).__init__()
        """"
        Parameters
            memory_size:        scalar
            memory_state_dim:   scalar
            is_write:           boolean
        """
        self.memory_size = memory_size
        self.memory_state_dim = memory_state_dim
        self.is_write = is_write
        if self.is_write:
            self.erase = torch.nn.Linear(self.memory_state_dim, self.memory_state_dim, bias=True)
            self.add = torch.nn.Linear(self.memory_state_dim, self.memory_state_dim, bias=True)
            nn.init.kaiming_normal_(self.erase.weight)
            nn.init.kaiming_normal_(self.add.weight)
            nn.init.constant_(self.erase.bias, 0)
            nn.init.constant_(self.add.bias, 0)

    def addressing(self, control_input, memory):
        """
        Parameters
            control_input:          Shape (batch_size, control_state_dim)
            memory:                 Shape (memory_size, memory_state_dim)
        Returns
            correlation_weight:     Shape (batch_size, memory_size)
        """
        similarity_score = torch.matmul(control_input, torch.t(memory))
        correlation_weight = torch.nn.functional.softmax(similarity_score, dim=1)  # Shape: (batch_size, memory_size)
        return correlation_weight

    def read(self, memory, control_input=None, read_weight=None):
        """
        Parameters
            control_input:  Shape (batch_size, control_state_dim)
            memory:         Shape (batch_size, memory_size, memory_state_dim)
            read_weight:    Shape (batch_size, memory_size)
        Returns
            read_content:   Shape (batch_size,  memory_state_dim)
        """
        if read_weight is None:
            read_weight = self.addressing(control_input=control_input, memory=memory)
        read_weight = read_weight.view(-1, 1)
        memory = memory.view(-1, self.memory_state_dim)
        rc = torch.mul(read_weight, memory)
        read_content = rc.view(-1, self.memory_size, self.memory_state_dim)
        read_content = torch.sum(read_content, dim=1)
        return read_content

    def write(self, control_input, memory, write_weight):
        """
        Parameters
            control_input:      Shape (batch_size, control_state_dim)
            write_weight:       Shape (batch_size, memory_size)
            memory:             Shape (batch_size, memory_size, memory_state_dim)
        Returns
            new_memory:         Shape (batch_size, memory_size, memory_state_dim)
        """
        assert self.is_write
        erase_signal = torch.sigmoid(self.erase(control_input))
        add_signal = torch.tanh(self.add(control_input))
        erase_reshape = erase_signal.view(-1, 1, self.memory_state_dim)
        add_reshape = add_signal.view(-1, 1, self.memory_state_dim)
        write_weight_reshape = write_weight.view(-1, self.memory_size, 1)
        erase_mult = torch.mul(erase_reshape, write_weight_reshape)
        add_mul = torch.mul(add_reshape, write_weight_reshape)
        new_memory = memory * (1 - erase_mult) + add_mul
        return new_memory


class DKVMN(nn.Module):
    def __init__(self, memory_size, memory_key_state_dim, memory_value_state_dim, init_memory_key):
        super(DKVMN, self).__init__()
        """
        :param memory_size:             scalar
        :param memory_key_state_dim:    scalar
        :param memory_value_state_dim:  scalar
        :param init_memory_key:         Shape (memory_size, memory_value_state_dim)
        :param init_memory_value:       Shape (batch_size, memory_size, memory_value_state_dim)
        """
        self.memory_size = memory_size
        self.memory_key_state_dim = memory_key_state_dim
        self.memory_value_state_dim = memory_value_state_dim

        self.key_head = DKVMNHeadGroup(memory_size=self.memory_size,
                                       memory_state_dim=self.memory_key_state_dim,
                                       is_write=False)

        self.value_head = DKVMNHeadGroup(memory_size=self.memory_size,
                                         memory_state_dim=self.memory_value_state_dim,
                                         is_write=True)

        self.memory_key = init_memory_key

        self.memory_value = None

    def init_value_memory(self, memory_value):
        self.memory_value = memory_value

    def attention(self, control_input):
        correlation_weight = self.key_head.addressing(control_input=control_input, memory=self.memory_key)
        return correlation_weight

    def read(self, read_weight):
        read_content = self.value_head.read(memory=self.memory_value, read_weight=read_weight)

        return read_content

    def write(self, write_weight, control_input):
        memory_value = self.value_head.write(control_input=control_input,
                                             memory=self.memory_value,
                                             write_weight=write_weight)

        self.memory_value = nn.Parameter(memory_value.data)

        return self.memory_value

#Readdata
import numpy as np
import itertools
from sklearn.model_selection import KFold


class DataReader():
    def __init__(self, train_path, test_path, maxstep, num_ques):
        self.train_path = train_path
        self.test_path = test_path
        self.maxstep = maxstep
        self.num_ques = num_ques

    def getData(self, file_path):
        datas = []
        with open(file_path, 'r') as file:
            for len, ques, ans in itertools.zip_longest(*[file] * 3):
                len = int(len.strip().strip(','))
                ques = [int(q) for q in ques.strip().strip(',').split(',')]
                ans = [int(a) for a in ans.strip().strip(',').split(',')]
                slices = len//self.maxstep + (1 if len % self.maxstep > 0 else 0)
                for i in range(slices):
                    data = np.zeros(shape=[self.maxstep, 3])  # 0 ->question and answer(1->)
                    if len > 0:                               # 1->question (1->)
                        if len >= self.maxstep:               # 2->label (0->1, 1->2)
                            steps = self.maxstep
                        else:
                            steps = len
                        for j in range(steps):
                            data[j][0] = ques[i * self.maxstep + j] + 1
                            data[j][2] = ans[i * self.maxstep + j] + 1
                            if ans[i * self.maxstep + j] == 1:
                                data[j][1] = ques[i * self.maxstep + j] + 1
                            else:
                                data[j][1] = ques[i * self.maxstep + j] + self.num_ques + 1
                        len = len - self.maxstep
                    datas.append(data.tolist())
            print('done: ' + str(np.array(datas).shape))
        return datas

    def getTrainData(self):
        print('loading train data...')
        kf = KFold(n_splits=5, shuffle=True, random_state=3)
        Data = np.array(self.getData(self.train_path))
        for train_indexes, vali_indexes in kf.split(Data):
            valiData = Data[vali_indexes].tolist()
            trainData = Data[train_indexes].tolist()
        return np.array(trainData), np.array(valiData)

    def getTestData(self):
        print('loading test data...')
        testData = self.getData(self.test_path)
        return np.array(testData)

#Dataloader
import torch
import torch.utils.data as Data
#assist2015/assist2015_train.txt assist2015/assist2015_test.txt
#assist2017/assist2017_train.txt assist2017/assist2017_test.txt
#assist2009/builder_train.csv assist2009/builder_test.csv

dts = 2017
setid = 1
stu = 50
que = 30

def getDataLoader(batch_size, num_of_questions, max_step, que, setid, stu):
    handle = DataReader('C:\\Users\\ibpri\\Downloads\\Knowledge Tracing\\DKVMN-No-ID\\DKVMN-main\\dataset\\assist'+str(dts)+'\\Test - '+str(stu)+'\\Set'+str(setid)+'\\assist'+str(dts)+'_train_new_'+str(stu)+'_set'+str(setid)+'.txt',
                        'C:\\Users\\ibpri\\Downloads\\Knowledge Tracing\\DKVMN-No-ID\\DKVMN-main\\dataset\\assist'+str(dts)+'\\Test - '+str(stu)+'\\Set'+str(setid)+'\\assist'+str(dts)+'_test_new_'+str(que)+'_set'+str(setid)+'.txt', max_step,
                        num_of_questions)
    train, vali = handle.getTrainData()
    dtrain = torch.tensor(train.astype(int).tolist(), dtype=torch.long)
    dvali = torch.tensor(vali.astype(int).tolist(), dtype=torch.long)
    dtest = torch.tensor(handle.getTestData().astype(int).tolist(),
                         dtype=torch.long)
    trainLoader = Data.DataLoader(dtrain, batch_size=batch_size, shuffle=True)
    valiLoader = Data.DataLoader(dvali, batch_size=batch_size, shuffle=True)
    testLoader = Data.DataLoader(dtest, batch_size=batch_size, shuffle=False)
    return trainLoader, valiLoader, testLoader
#eval
import tqdm
import torch
import logging
import os
from sklearn import metrics
logger = logging.getLogger('main.eval')

def __load_model__(ckpt):
    '''
    ckpt: Path of the checkpoint
    return: Checkpoint dict
    '''
    if os.path.isfile(ckpt):
        checkpoint = torch.load(ckpt)
        print("Successfully loaded checkpoint '%s'" % ckpt)
        return checkpoint
    else:
        raise Exception("No checkpoint found at '%s'" % ckpt)


def train_epoch(model, trainLoader, optimizer, device):
    model.to(device)
    for batch in tqdm.tqdm(trainLoader, desc='Training:    ', mininterval=2):
        batch = batch.to(device)
        datas = torch.chunk(batch, 3, 2)
        optimizer.zero_grad()
        loss, prediction, ground_truth = model(datas[0].squeeze(2), datas[1].squeeze(2), datas[2])
        loss.backward()
        optimizer.step()
    return model, optimizer


def test_epoch(model, testLoader, device, ckpt=None):
    model.to(device)
    if ckpt is not None:
        checkpoint = __load_model__(ckpt)
        model.load_state_dict(checkpoint['state_dict'])
    ground_truth = torch.tensor([], device=device)
    prediction = torch.tensor([], device=device)
    for batch in tqdm.tqdm(testLoader, desc='Testing:     ', mininterval=2):
        batch = batch.to(device)
        datas = torch.chunk(batch, 3, 2)
        loss, p, label = model(datas[0].squeeze(2), datas[1].squeeze(2), datas[2])
        prediction = torch.cat([prediction, p])
        ground_truth = torch.cat([ground_truth, label])
    acc = metrics.accuracy_score(torch.round(ground_truth).detach().cpu().numpy(), torch.round(prediction).detach().cpu().numpy())
    auc = metrics.roc_auc_score(ground_truth.detach().cpu().numpy(), prediction.detach().cpu().numpy())
    logger.info('auc: ' + str(auc) + ' acc: ' + str(acc))
    print(#'auc: ' + str(auc) + 
          ' acc: ' + str(acc))
    return auc

#run
"""
Usage:
    run.py  [options]

Options:
    --length=<int>                      max length of question sequence [default: 50]
    --questions=<int>                   num of question [default: 100]
    --lr=<float>                        learning rate [default: 0.001]
    --bs=<int>                          batch size [default: 64]
    --seed=<int>                        random seed [default: 59]
    --epochs=<int>                      number of epochs [default: 30]
    --cuda=<int>                        use GPU id [default: 0]
    --final_fc_dim=<int>                dimension of final dim [default: 10]
    --question_dim=<int>                dimension of question dim[default: 50]
    --question_and_answer_dim=<int>     dimension of question and answer dim [default: 100]
    --memory_size=<int>               memory size [default: 20]
    --model=<string>                    model type [default: DKVMN]
"""

import os
import random
import logging
import torch
import torch.optim as optim
import numpy as np
from datetime import datetime

# from data.dataloader import getDataLoader
from evaluation import eval


def setup_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# args = docopt(__doc__)
length = 50
questions = 200
lr = 0.001
bs = 64
seed = 59
epochs = 20
cuda = 0
final_fc_dim = 20
question_dim = 50
question_and_answer_dim = 100
memory_size = 20
model_type = 'DKVMN'

# logger = logging.getLogger('main')
# logger.setLevel(level=logging.DEBUG)
# date = datetime.now()
# handler = logging.FileHandler(f'log/{date.year}_{date.month}_{date.day}_{model_type}_result.log')
# handler.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)
# logger.info('DKVMN')
# logger.info(list(args.items()))

setup_seed(seed)

if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
# for que in range(3,21):
#     trainLoader, validationLoader, testLoader = getDataLoader(bs, questions, length, que)

from model.model import MODEL
trainLoader, validationLoader, testLoader = getDataLoader(bs, questions, length, que, setid, stu)
model = MODEL(n_question=questions, batch_size=bs, q_embed_dim=question_dim, qa_embed_dim=question_and_answer_dim,
            memory_size=memory_size, final_fc_dim=final_fc_dim)
model.init_params()
model.init_embeddings()
optimizer = optim.Adam(model.parameters(), lr=lr)
best_auc = 0
for epoch in range(epochs):
        print('epoch: ' + str(epoch+1))
        model, optimizer = eval.train_epoch(model, trainLoader, optimizer, device)
        logger.info(f'epoch {epoch+1}')
        auc = eval.test_epoch(model, validationLoader, device)
        if auc > best_auc:
            # print('best checkpoint')
            torch.save({'state_dict': model.state_dict()}, 'C:\\Users\\ibpri\\Downloads\\Knowledge Tracing\\DKVMN-No-ID\\DKVMN-main\\checkpoint\\'+model_type+'.pth.tar')
            best_auc = auc

for setid in range(1,5):
 print("-------------Run for Set ID = " + str(setid) + '--------------')
 for que in range(3, 31):
    print("--------------------------------Questions = " + str(que) + " ----------------------------------------")
    trainLoader, validationLoader, testLoader = getDataLoader(bs, questions, length, que, setid, stu)
    # Custom checkpoint loading to ignore mem.memory_value shape mismatch
    ckpt_path = 'C:\\Users\\ibpri\\Downloads\\Knowledge Tracing\\DKVMN-No-ID\\DKVMN-main\\checkpoint\\' + model_type + '.pth.tar'
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['state_dict']
    # Remove mem.memory_value from state_dict if present
    state_dict = {k: v for k, v in state_dict.items() if "mem.memory_value" not in k}
    model.load_state_dict(state_dict, strict=False)
    eval.test_epoch(model, testLoader, device)
