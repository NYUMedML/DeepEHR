# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: jshliu

Scripts of multi-event encounter level models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from sklearn import metrics
import pdb
import pickle
import json
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
# import matplotlib.gridspec as gridspec

import util
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


## ====== Data Loaders ============
class encDataset(Dataset):
    # Each record is: [[text_enc1, text_enc2, ...], [num_enc1, num_enc2, ...], disease, mask, age, gender, race, eth]
    def __init__(self, root_dir, dsName, nClassGender, nClassRace, nClassEthnic, transform=None):
        self.root_dir = root_dir
        self.ds = json.load(open(root_dir + dsName, 'r'))
        print('Loaded: ', dsName)
        self.transform = transform
        self.nClass = [nClassGender, nClassRace, nClassEthnic]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        """
        Shape of the inputs:
            - Note: 1 x n_enc x noteLength
            - Num: 1 x n_enc x dimNum
            - Disease: 1 x 3
            - Mask: 1 x 3
            - Age: 1
            - Demo ([Gender, race, eth]): 1 x 3
        """
        Note, Num, Disease, Mask, Age, gender, race, eth = self.ds[idx]

        Num = np.asarray(Num, dtype='float32')
        Disease = np.asarray(Disease, dtype='int')
        Mask = np.asarray(Mask, dtype='int')
        Age = np.asarray(Age, dtype='float32')

        gender2 = self._idx2onehot(gender, self.nClass[0])
        race2 = self._idx2onehot(race, self.nClass[1])
        eth2 = self._idx2onehot(eth, self.nClass[2])

        Demo = np.concatenate([gender2, race2, eth2])
        sample = {'Note': Note, 'Num': Num, 'Disease': Disease, 'Mask': Mask, 'Age': Age, 'Demo': Demo}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _idx2onehot(self, value_idx, max_idx):
        temp = np.zeros(max_idx)
        if value_idx > 0:
            temp[value_idx - 1] = 1
        return temp


class concatNote(object):
    """
    concatenate all notes regardless of encounter, but still maintain numerics by encounter
    """

    def __init__(self, enc_len, doc_len):
        self.enc_len = enc_len
        self.doc_len = doc_len

    def __call__(self, sample):
        Note, Num, Disease, Mask, Age, Demo = sample['Note'], sample['Num'], sample['Disease'], sample['Mask'], sample[
            'Age'], sample['Demo']
        return self._run(Note, Num, Disease, Mask, Age, Demo)

    def _run(self, Note, Num, Disease, Mask, Age, Demo):
        # Pad / truncate notes
        if len(Note) > self.enc_len:
            Num = Num[-self.enc_len:, :]
        padded_Num = torch.zeros(self.enc_len, Num.shape[1])
        padded_Num[-Num.shape[0]:, :] = torch.from_numpy(Num).float()

        note_all = [w for note in Note for w in note]
        if len(note_all) > self.doc_len:
            note_all = note_all[-self.doc_len:]
        padded_Note = torch.zeros(1, self.doc_len)
        if len(note_all) > 0:
            padded_Note[:, -len(note_all):] = torch.from_numpy(np.asarray(note_all)).long()

        return {'Note': padded_Note,
                'Num': padded_Num,
                'Disease': torch.from_numpy(Disease.reshape(1, -1)).float(),
                'Mask': torch.from_numpy(Mask.reshape(1, -1)).float(),
                'Age': torch.from_numpy(Age.reshape(1, -1)).float(),
                'Demo': torch.from_numpy(Demo.reshape(1, -1)).float()
                }

class staticDataset(Dataset):
    
        # Each record is: [[text_enc1, text_enc2, ...], [num_enc1, num_enc2, ...], disease, mask, age, gender, race, eth]
    def __init__(self, root_dir, dsName, nClassGender, nClassRace, nClassEthnic, transform=None):
        self.root_dir = root_dir
        self.ds = json.load(open(root_dir + dsName, 'r'))
        print('Loaded: ', dsName)
        self.transform = transform
        self.nClass = [nClassGender, nClassRace, nClassEthnic]
        self.max_len = 3000

    def __len__(self):
        return len(self.ds)


    def __getitem__(self, idx):

        """
        Shape of the inputs:
            - Note: 1 x n_enc x noteLength
            - Num: 1 x n_enc x dimNum
            - Disease: 1 x 3
            - Mask: 1 x 3
            - Age: 1
            - Demo ([Gender, race, eth]): 1 x 3
        """
        
        Note, Num, Disease, Mask, Age, gender, race, eth = self.ds[idx]

        Note = np.asarray([item for sublist in Note for item in sublist])
        Num = np.asarray(Num, dtype='float32').mean(axis=0)
        Disease = np.asarray(Disease, dtype='int')
        Mask = np.asarray(Mask, dtype='int')
        Age = np.asarray(Age, dtype='float32')

        if len(Note) > self.max_len:
            Note = Note[:self.max_len]
            
        else:        
            Note = np.concatenate([ np.zeros( self.max_len - Note.shape[0] ) , Note ])

        Note = torch.from_numpy(Note).long()

        gender2 = self._idx2onehot(gender, self.nClass[0])
        race2 = self._idx2onehot(race, self.nClass[1])
        eth2 = self._idx2onehot(eth, self.nClass[2])

        Demo = np.concatenate([gender2, race2, eth2])
        sample = {'Note': Note, 'Num': Num, 'Disease': Disease, 'Mask': Mask, 'Age': Age, 'Demo': Demo}

        return {'Note': Note,
                'Num': Num,
                'Disease': torch.from_numpy(Disease.reshape(1, -1)).float(),
                'Mask': torch.from_numpy(Mask.reshape(1, -1)).float(),
                'Age': torch.from_numpy(Age.reshape(1, -1)).float(),
                'Demo': torch.from_numpy(Demo.reshape(1, -1)).float()
                }

    def _idx2onehot(self, value_idx, max_idx):
    
        temp = np.zeros(max_idx)
        if value_idx > 0:
            temp[value_idx - 1] = 1
        return temp

  
 


class padOrTruncateToTensor(object):
    """
    pad or truncate sample to make it as long as the given enc_len, doc_len
    """

    def __init__(self, enc_len, doc_len):
        self.enc_len = enc_len
        self.doc_len = doc_len

    def __call__(self, sample):
        Note, Num, Disease, Mask, Age, Demo = sample['Note'], sample['Num'], sample['Disease'], sample['Mask'], sample[
            'Age'], sample['Demo']
        return self._run(Note, Num, Disease, Mask, Age, Demo)

    def _run(self, Note, Num, Disease, Mask, Age, Demo):
        # Pad / truncate notes
        if len(Note) > self.enc_len:
            Note = Note[-self.enc_len:]
            Num = Num[-self.enc_len:, :]
        padded_Note = self._pad_doc(Note, self.doc_len, self.enc_len)
        padded_Num = torch.zeros(self.enc_len, Num.shape[1])
        padded_Num[-Num.shape[0]:, :] = torch.from_numpy(Num).float()

        return {'Note': padded_Note,
                'Num': padded_Num,
                'Disease': torch.from_numpy(Disease.reshape(1, -1)).float(),
                'Mask': torch.from_numpy(Mask.reshape(1, -1)).float(),
                'Age': torch.from_numpy(Age.reshape(1, -1)).float(),
                'Demo': torch.from_numpy(Demo.reshape(1, -1)).float()
                }

    def _pad_doc(self, seq, max_len, n):
        padded_seq = torch.zeros(n, max_len)
        start = 0 if len(seq) >= n else n - len(seq)
        for i, s in enumerate(seq):
            if len(s) > max_len:
                padded_seq[start + i] = torch.from_numpy(np.asarray(s[:max_len])).long()
            else:
                if len(s) == 0:
                    continue
                padded_seq[start + i, -len(s):] = torch.from_numpy(np.asarray(s)).long()
        return padded_seq


## ======= Models =========

class Enc_SumLSTM(nn.Module):

    def __init__(self, model_paras, embedding = None):
        """
        model_paras:
            - enc_len: max number of encounters in each sample
            - doc_len: max number of words in each enc, input dimension should be [batchSize, enc_len, doc_len]
            - flg_updateEmb: whether to train embeddings or not
            - batchSize: batch size
            - rnnType: 'GRU' or 'LSTM'
            - bidir: whether to train bi-directional RNN or not
            - p_dropOut: dropOut percentage
            - lsDim: dimensions of [hidden_state, linear dimension 1, linear dimension 2...]
            - flg_cuda: use GPU or not
            - emb_dim: embedding dimension, do not need if provide embedding
            - n_words: vocabulary size,  do not need if provide embedding
        To-do:
            expand to handle more than 1 linear layers
        """

        super(Enc_SumLSTM,self).__init__()

        #self.enc_len = model_paras.get('enc_len', 15)
        self.doc_len = model_paras.get('doc_len', 20)
        flg_updateEmb = model_paras.get('flg_updateEmb', False)
        self.model_paras = model_paras
        self.rnnType = model_paras.get('rnnType', 'GRU')
        self.bidir = model_paras.get('bidir', False)
        self.p_dropOut = model_paras.get('p_dropOut', 0.8)
        self.lsDim = model_paras.get('lsDim', [128, 1]) 
        self.flg_cuda = model_paras.get('flg_cuda', True)
        
        
        if embedding is not None:        
            self.n_words = embedding.size()[0]
            self.emb_dim = embedding.size()[1]
            self.embed = nn.Embedding(self.n_words, self.emb_dim)
            self.embed.weight = nn.Parameter(embedding,requires_grad=flg_updateEmb)
        else:
            self.n_words = model_paras.get('n_words', 20000)
            self.emb_dim = model_paras.get('emb_dim', 300)
            self.embed = nn.Embedding(self.n_words, self.emb_dim)
            
        self.lstm = getattr(nn, self.rnnType)(self.emb_dim, self.lsDim[0], 1, batch_first=True, bidirectional = self.bidir, dropout= self.p_dropOut)

        if self.bidir:
            self.fc = nn.Linear(2*self.lsDim[0], self.lsDim[1])
        else:
            self.fc = nn.Linear(self.lsDim[0], self.lsDim[1])
        
        self.params = list(self.lstm.parameters()) + list(self.fc.parameters())
        if flg_updateEmb:
            self.params += list(self.embed.parameters())
        

    def init_hidden(self, batchSize, nlayer = 1):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if self.rnnType == 'LSTM':
            if self.flg_cuda:
                return  (Variable(torch.zeros(nlayer, batchSize, self.lsDim[0])).cuda(), Variable(torch.zeros(nlayer, batchSize, self.lsDim[0])).cuda() )
            else:
                return (Variable(torch.zeros(nlayer, batchSize, self.lsDim[0])), Variable(torch.zeros(nlayer, batchSize, self.lsDim[0])))
        else:
            if self.flg_cuda:
                return Variable(torch.zeros(nlayer, batchSize, self.lsDim[0])).cuda()
            else:
                return Variable(torch.zeros(nlayer, batchSize, self.lsDim[0]))
            
    def forward(self,x):
        batchSize = x.size()[0]
        x= x.view(-1,self.doc_len)

        E= self.embed(x)

        E = torch.chunk(E, batchSize, 0)
        E = torch.stack(E)
        E = E.sum(2)
        #x = x.transpose(1,2)

        if self.bidir:
            h0 = self.init_hidden(batchSize = batchSize, nlayer = 2)
            z = self.lstm(E, h0)[1]
            z = torch.cat([ z[0].squeeze() , z[1].squeeze()] , 1)

        else:
            h0 = self.init_hidden(batchSize = batchSize, nlayer = 1)
            z = self.lstm(E, h0)[0][:, -1, :]

        y_hat = self.fc(z)
        return F.sigmoid(y_hat)


class Enc_CNN_LSTM(nn.Module):

    def __init__(self, model_paras, embedding=None):
        """
        CNN on word, then LSTM with enc
        model_paras:
        - enc_len: max number of encounters in each sample
        - doc_len: max number of words in each enc, input dimension should be [batchSize, enc_len, doc_len]
        - flg_updateEmb: whether to train embeddings or not
        - batchSize: batch size
        - rnnType: 'GRU' or 'LSTM'
        - bidir: whether to train bi-directional RNN or not
        - p_dropOut: dropOut percentage
        - lsDim: dimensions of [hidden_state, multi-event dimension, 1]
        - flg_cuda: use GPU or not
        - emb_dim: embedding dimension, do not need if provide embedding
        - n_words: vocabulary size,  do not need if provide embedding
        - filters: dimension of CNN output
        - Ks: kernels
        - randn_std: std of random noise on embedding
        """

        super(Enc_CNN_LSTM, self).__init__()

        # self.enc_len = model_paras.get('enc_len', 30)
        self.doc_len = model_paras.get('doc_len', 800)
        flg_updateEmb = model_paras.get('flg_updateEmb', False)
        self.model_paras = model_paras
        self.rnnType = model_paras.get('rnnType', 'GRU')
        self.dimLSTM = model_paras.get('dimLSTM', 128)  # LSTM hidden layer dimension
        self.bidir = model_paras.get('bidir', False)
        self.p_dropOut = model_paras.get('p_dropOut', 0.8)
        self.lsDim = model_paras.get('lsDim')
        self.flg_cuda = model_paras.get('flg_cuda', True)
        self.filters = model_paras.get('filters', 128)
        self.Ks = model_paras.get('Ks', [1, 2])
        self.randn_std = model_paras.get('randn_std', None)
        self.lastRelu = model_paras.get('lastRelu', False)
        self.isViz = model_paras.get('isViz', False)
        self.flgBias = model_paras.get('flgBias', True)
        self.flg_AllLSTM = model_paras.get('flg_AllLSTM', False)

        if embedding is not None:
            self.n_words = embedding.size()[0]
            self.emb_dim = embedding.size()[1]
            self.embed = nn.Embedding(self.n_words, self.emb_dim)
            self.embed.weight = nn.Parameter(embedding, requires_grad=flg_updateEmb)
        else:
            self.n_words = model_paras.get('n_words', 20000)
            self.emb_dim = model_paras.get('emb_dim', 300)
            self.embed = nn.Embedding(self.n_words, self.emb_dim)

        self.convs = nn.ModuleList([nn.Conv1d(self.emb_dim, self.filters, K) for K in self.Ks])
        self.bn_conv = nn.ModuleList([nn.BatchNorm1d(self.filters) for K in self.Ks])

        self.lstm = getattr(nn, self.rnnType)(self.filters * len(self.Ks), self.dimLSTM, 1, batch_first=True,
                                              bidirectional=self.bidir,
                                              dropout=self.p_dropOut, bias=self.flgBias)

        self.FCs = nn.ModuleList([nn.Linear(self.lsDim[i], self.lsDim[i + 1]) for i in range(len(self.lsDim) - 1)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.lsDim[i + 1]) for i in range(len(self.lsDim) - 2)])

        self.params = list(self.lstm.parameters())
        # self.lstm.weight.data.normal_(0.0, 0.02)
        # self.lstm.bias.data.normal_(0.0, 0.01)
        # self.reg_params = list(self.lstm.)
        for c in self.convs:
            self.params += list(c.parameters())
            # n = c.kernel_size[0] * c.out_channels
            # c.weight.data.normal_(0, np.sqrt(2. / n))

        for b in self.bn_conv:
            self.params += list(b.parameters())

        for fc in self.FCs:
            self.params += list(fc.parameters())
            # fc.weight.data.normal_(0.0, 0.02)

        for bn in self.bns:
            self.params += list(bn.parameters())

        if flg_updateEmb:
            self.params += list(self.embed.parameters())

    def fc_layer(self, x, layer, bn=None):
        flg_bn = self.model_paras.get('flg_bn', True)
        p_dropOut = self.model_paras.get('p_dropOut', 0.5)

        x = layer(x)
        if flg_bn:
            x = F.relu(bn(x))
        else:
            x = F.relu(x)
        x = F.dropout(x, p_dropOut)
        return x

    def init_hidden(self, batchSize, nlayer=1):
        if self.rnnType == 'LSTM':
            if self.flg_cuda:
                return (Variable(torch.zeros(nlayer, batchSize, self.dimLSTM)).cuda(),
                        Variable(torch.zeros(nlayer, batchSize, self.dimLSTM)).cuda())
            else:
                return (Variable(torch.zeros(nlayer, batchSize, self.dimLSTM)),
                        Variable(torch.zeros(nlayer, batchSize, self.dimLSTM)))
        else:
            if self.flg_cuda:
                return Variable(torch.zeros(nlayer, batchSize, self.dimLSTM)).cuda()
            else:
                return Variable(torch.zeros(nlayer, batchSize, self.dimLSTM))

    def forward(self, Note, Num, Disease, Mask, Age, Demo):

        batchSize = Note.size()[0]
        self.x = Note.view(-1, self.doc_len)

        self.E = self.embed(self.x)  # Size: [batch * enc, doc_len, emb_dim]
        E = self.E.transpose(1, 2)

        if (self.randn_std is not None) & (self.training == True):
            if self.flg_cuda:
                noise = Variable(torch.randn(E.size()) * self.randn_std).cuda()
            else:
                noise = Variable(torch.randn(E.size()) * self.randn_std)
            E = E + noise

        # CNN on words
        h_CNN = [self.fc_layer(E, self.convs[i], self.bn_conv[i]) for i in range(len(self.convs))]
        h_CNN = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in h_CNN]
        h_CNN = torch.cat(h_CNN, 1)

        E2 = torch.chunk(h_CNN, batchSize, 0)
        E2 = torch.stack(E2)

        if self.bidir:
            h0 = self.init_hidden(batchSize=batchSize, nlayer=2)
            z = self.lstm(E2, h0)[1]
            z = torch.cat([z[0].squeeze(), z[1].squeeze()], 1)
        else:
            h0 = self.init_hidden(batchSize=batchSize, nlayer=1)
            if self.flg_AllLSTM:
                z = self.lstm(E2, h0)[0].contiguous().view(batchSize, -1)
            else:
                z = self.lstm(E2, h0)[0][:, -1, :]

        if len(self.lsDim) > 2:
            for i in range(len(self.lsDim) - 2):
                z = self.fc_layer(z, self.FCs[i], self.bns[i])

        self.y_hat = self.FCs[-1](z)

        if self.isViz:
            return self.y_hat
        else:
            return F.sigmoid(self.y_hat)


class DemoLab(nn.Module):
    def __init__(self, model_paras, embedding):
        """
        LSTM with lab and concatenate demographics
        model_paras:
        - enc_len: max number of encounters in each sample
        - flg_updateEmb: whether to train embeddings or not
        - batchSize: batch size
        - rnnType: 'GRU' or 'LSTM'
        - bidir: whether to train bi-directional RNN or not
        - p_dropOut: dropOut percentage
        - lsDim: dimensions of [hidden_state, multi-event dimension, 1]
        - flg_cuda: use GPU or not
        - emb_dim: embedding dimension, do not need if provide embedding
        - randn_std: std of random noise on embedding
        """

        super(DemoLab, self).__init__()

        self.enc_len = model_paras.get('enc_len', 30)
        flg_updateEmb = model_paras.get('flg_updateEmb', True) # Embedding for demographics
        self.model_paras = model_paras
        self.rnnType = model_paras.get('rnnType', 'GRU')
        self.dimLSTM = model_paras.get('dimLSTM', 128)  # LSTM hidden layer dimension
        self.bidir = model_paras.get('bidir', False)
        self.p_dropOut = model_paras.get('p_dropOut', 0.8)
        self.lsDim = model_paras.get('lsDim')
        self.flg_cuda = model_paras.get('flg_cuda', True)
        self.randn_std = model_paras.get('randn_std', None)
        self.lastRelu = model_paras.get('lastRelu', False)
        self.isViz = model_paras.get('isViz', False)
        self.flgBias = model_paras.get('flgBias', True)
        self.inSize = 151 + 56 + 1 # Dimension of input
        self.flg_AllLSTM = model_paras.get('flg_AllLSTM', False)

        #self.embed = nn.Embedding(self.n_words, self.emb_dim)

        self.lstm = getattr(nn, self.rnnType)(self.inSize, self.dimLSTM, 1, batch_first=True,
                                              bidirectional=self.bidir,
                                              dropout=self.p_dropOut, bias=self.flgBias)

        self.FCs = nn.ModuleList([nn.Linear(self.lsDim[i], self.lsDim[i + 1]) for i in range(len(self.lsDim) - 1)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.lsDim[i + 1]) for i in range(len(self.lsDim) - 2)])

        self.params = list(self.lstm.parameters())

        for fc in self.FCs:
            self.params += list(fc.parameters())
            # fc.weight.data.normal_(0.0, 0.02)

        for bn in self.bns:
            self.params += list(bn.parameters())

    def fc_layer(self, x, layer, bn=None):
        flg_bn = self.model_paras.get('flg_bn', True)
        p_dropOut = self.model_paras.get('p_dropOut', 0.5)

        x = layer(x)
        if flg_bn:
            x = F.relu(bn(x))
        else:
            x = F.relu(x)
        x = F.dropout(x, p_dropOut)
        return x

    def init_hidden(self, batchSize, nlayer=1):
        if self.rnnType == 'LSTM':
            if self.flg_cuda:
                return (Variable(torch.zeros(nlayer, batchSize, self.dimLSTM)).cuda(),
                        Variable(torch.zeros(nlayer, batchSize, self.dimLSTM)).cuda())
            else:
                return (Variable(torch.zeros(nlayer, batchSize, self.dimLSTM)),
                        Variable(torch.zeros(nlayer, batchSize, self.dimLSTM)))
        else:
            if self.flg_cuda:
                return Variable(torch.zeros(nlayer, batchSize, self.dimLSTM)).cuda()
            else:
                return Variable(torch.zeros(nlayer, batchSize, self.dimLSTM))

    def forward(self, Note, Num, Disease, Mask, Age, Demo):

        batchSize = Note.size()[0]
        demo = torch.cat([Age, Demo], dim = 2)
        self.x = torch.cat([Num, demo.repeat(1, self.enc_len, 1)], dim = 2)

        if self.bidir:
            h0 = self.init_hidden(batchSize=batchSize, nlayer=2)
            z = self.lstm(self.x, h0)[1]
            z = torch.cat([z[0].squeeze(), z[1].squeeze()], 1)
        else:
            h0 = self.init_hidden(batchSize=batchSize, nlayer=1)
            if self.flg_AllLSTM:
                z = self.lstm(self.x, h0)[0].contiguous().view(batchSize, -1)
            else:
                z = self.lstm(self.x, h0)[0][:, -1, :]

        if len(self.lsDim) > 2:
            for i in range(len(self.lsDim) - 2):
                z = self.fc_layer(z, self.FCs[i], self.bns[i])

        self.y_hat = self.FCs[-1](z)

        if self.isViz:
            return self.y_hat
        else:
            return F.sigmoid(self.y_hat)




class Enc_CNN_LSTM_DemoLab(nn.Module):

    def __init__(self, model_paras, embedding=None):
        """
        CNN on word, then LSTM with enc
        model_paras:
        - enc_len: max number of encounters in each sample
        - doc_len: max number of words in each enc, input dimension should be [batchSize, enc_len, doc_len]
        - flg_updateEmb: whether to train embeddings or not
        - batchSize: batch size
        - rnnType: 'GRU' or 'LSTM'
        - bidir: whether to train bi-directional RNN or not
        - p_dropOut: dropOut percentage
        - lsDim: dimensions of [hidden_state, multi-event dimension, 1]
        - flg_cuda: use GPU or not
        - emb_dim: embedding dimension, do not need if provide embedding
        - n_words: vocabulary size,  do not need if provide embedding
        - filters: dimension of CNN output
        - Ks: kernels
        - randn_std: std of random noise on embedding
        """

        super(Enc_CNN_LSTM_DemoLab, self).__init__()

        # self.enc_len = model_paras.get('enc_len', 30)
        self.doc_len = model_paras.get('doc_len', 800)
        flg_updateEmb = model_paras.get('flg_updateEmb', False)
        self.model_paras = model_paras
        self.rnnType = model_paras.get('rnnType', 'GRU')
        self.dimLSTM = model_paras.get('dimLSTM', 128)  # LSTM hidden layer dimension
        self.bidir = model_paras.get('bidir', False)
        self.p_dropOut = model_paras.get('p_dropOut', 0.8)
        self.lsDim = model_paras.get('lsDim')
        self.flg_cuda = model_paras.get('flg_cuda', True)
        self.filters = model_paras.get('filters', 128)
        self.Ks = model_paras.get('Ks', [1, 2])
        self.randn_std = model_paras.get('randn_std', None)
        self.lastRelu = model_paras.get('lastRelu', False)
        self.isViz = model_paras.get('isViz', False)
        self.flgBias = model_paras.get('flgBias', True)
        self.flg_AllLSTM = model_paras.get('flg_AllLSTM', False)
        self.flg_useNum = model_paras.get('flg_useNum', False)
        if self.flg_useNum:
            self.inSize = 151 + 56 + 1  # Use lab + demo
        else:
            self.inSize = 56 + 1  # Demo only

        if embedding is not None:
            self.n_words = embedding.size()[0]
            self.emb_dim = embedding.size()[1]
            self.embed = nn.Embedding(self.n_words, self.emb_dim)
            self.embed.weight = nn.Parameter(embedding, requires_grad=flg_updateEmb)
        else:
            self.n_words = model_paras.get('n_words', 20000)
            self.emb_dim = model_paras.get('emb_dim', 300)
            self.embed = nn.Embedding(self.n_words, self.emb_dim)

        self.convs = nn.ModuleList([nn.Conv1d(self.emb_dim, self.filters, K) for K in self.Ks])
        self.bn_conv = nn.ModuleList([nn.BatchNorm1d(self.filters) for K in self.Ks])

        self.lstm = getattr(nn, self.rnnType)(self.filters * len(self.Ks) + self.inSize, self.dimLSTM, 1,
                                              batch_first=True,
                                              bidirectional=self.bidir,
                                              dropout=self.p_dropOut, bias=self.flgBias)

        self.FCs = nn.ModuleList([nn.Linear(self.lsDim[i], self.lsDim[i + 1]) for i in range(len(self.lsDim) - 1)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.lsDim[i + 1]) for i in range(len(self.lsDim) - 2)])

        self.params = list(self.lstm.parameters())

        for c in self.convs:
            self.params += list(c.parameters())

        for b in self.bn_conv:
            self.params += list(b.parameters())

        for fc in self.FCs:
            self.params += list(fc.parameters())

        for bn in self.bns:
            self.params += list(bn.parameters())

        if flg_updateEmb:
            self.params += list(self.embed.parameters())

    def fc_layer(self, x, layer, bn=None):
        flg_bn = self.model_paras.get('flg_bn', True)
        p_dropOut = self.model_paras.get('p_dropOut', 0.5)

        x = layer(x)
        if flg_bn:
            x = F.relu(bn(x))
        else:
            x = F.relu(x)
        x = F.dropout(x, p_dropOut)
        return x

    def init_hidden(self, batchSize, nlayer=1):
        if self.rnnType == 'LSTM':
            if self.flg_cuda:
                return (Variable(torch.zeros(nlayer, batchSize, self.dimLSTM)).cuda(),
                        Variable(torch.zeros(nlayer, batchSize, self.dimLSTM)).cuda())
            else:
                return (Variable(torch.zeros(nlayer, batchSize, self.dimLSTM)),
                        Variable(torch.zeros(nlayer, batchSize, self.dimLSTM)))
        else:
            if self.flg_cuda:
                return Variable(torch.zeros(nlayer, batchSize, self.dimLSTM)).cuda()
            else:
                return Variable(torch.zeros(nlayer, batchSize, self.dimLSTM))

    def forward(self, Note, Num, Disease, Mask, Age, Demo):

        batchSize, enc_len = Note.size()[0], Note.size()[1]
        self.x = Note.view(-1, self.doc_len)

        self.E = self.embed(self.x)  # Size: [batch * enc, doc_len, emb_dim]
        E = self.E.transpose(1, 2)

        if (self.randn_std is not None) & (self.training == True):
            if self.flg_cuda:
                noise = Variable(torch.randn(E.size()) * self.randn_std).cuda()
            else:
                noise = Variable(torch.randn(E.size()) * self.randn_std)
            E = E + noise

        # CNN on words
        h_CNN = [self.fc_layer(E, self.convs[i], self.bn_conv[i]) for i in range(len(self.convs))]
        h_CNN = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in h_CNN]
        h_CNN = torch.cat(h_CNN, 1)

        E2 = torch.chunk(h_CNN, batchSize, 0)
        E2 = torch.stack(E2)

        demo = torch.cat([Age, Demo], dim=2)
        if self.flg_useNum:
            E2 = torch.cat([E2, Num, demo.repeat(1, enc_len, 1)], dim=2)
        else:
            E2 = torch.cat([E2, demo.repeat(1, enc_len, 1)], dim=2)

        if self.bidir:
            h0 = self.init_hidden(batchSize=batchSize, nlayer=2)
            z = self.lstm(E2, h0)[1]
            z = torch.cat([z[0].squeeze(), z[1].squeeze()], 1)
        else:
            h0 = self.init_hidden(batchSize=batchSize, nlayer=1)
            if self.flg_AllLSTM:
                z = self.lstm(E2, h0)[0].contiguous().view(batchSize, -1)
            else:
                z = self.lstm(E2, h0)[0][:, -1, :]

        if len(self.lsDim) > 2:
            for i in range(len(self.lsDim) - 2):
                z = self.fc_layer(z, self.FCs[i], self.bns[i])

        self.y_hat = self.FCs[-1](z)

        if self.isViz:
            return self.y_hat
        else:
            return F.sigmoid(self.y_hat)




# ============= Wrap training into a class ================
class trainModel(object):
    def __init__(self, train_paras, train_loader_pos, test_loader, model, optimizer, train_loader_neg=None):
        self.train_loader = train_loader_pos  # If no train_loader_neg, then pass the train_loader here; otherwise split _pos and _neg
        self.train_loader_neg = train_loader_neg
        self.test_loader = test_loader
        self.model = model
        self.optimizer = optimizer

        # self.train_paras = train_paras
        self.n_iter = train_paras.get('n_iter', 1)
        self.log_interval = train_paras.get('log_interval',
                                            1)  # list of two numbers: [log_per_n_epoch, log_per_n_batch]
        self.flg_cuda = train_paras.get('flg_cuda', False)
        # self.max_len = train_paras.get('max_len', 2000) # Max length of input
        self.lr_decay = train_paras.get('lr_decay',
                                        None)  # List of 4 numbers: [init_lr, lr_decay_rate, lr_decay_interval, min_lr]
        self.flgSave = train_paras.get('flgSave', False)  # Set to true if save model
        self.savePath = train_paras.get('savePath', './')
        self.posThres = train_paras.get('posThres', 0.5)
        self.alpha_L1 = train_paras.get('alpha_L1', 0.0)  # Regularization coefficient on fully connected weights
        self.flg_gradClip = train_paras.get('flg_gradClip', False)  # Whether to clip gradient or not

        if self.lr_decay:
            assert len(
                self.lr_decay) == 4  # Elements include: [starting_lr, decay_multiplier, decay_per_?_epoch, min_lr]
        self.criterion = torch.nn.BCELoss()
        # self.criterion = torch.nn.MultiLabelSoftMarginLoss() # For multi-class prediction
        self.cnt_iter = 0

    def run(self):
        lsTrainAccuracy = []
        lsTestAccuracy = []
        self.bestAccuracy = 0.0
        self.auc = 0.0

        for epoch in range(self.n_iter):
            self._train(epoch, lsTrainAccuracy)
            self._test(epoch, lsTestAccuracy)
            if self.auc > self.bestAccuracy:
                self.bestAccuracy = self.auc
                self._drawAUC(self.Y_multi, self.target_multi, self.mask)
                if self.flgSave:
                    self._saveModel()
        return self.model, lsTrainAccuracy, lsTestAccuracy

    def _train(self, epoch, lsTrainAccuracy):
        correct, train_loss = np.zeros(3), 0
        self.model.train()
        if self.lr_decay:
            lr = min(self.lr_decay[0] * (self.lr_decay[1] ** (epoch // self.lr_decay[2])), self.lr_decay[3])
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        j, nRec = 0, np.zeros(3)

        self.Y_multi_train = []
        self.target_multi_train = []
        self.mask_train = []
        
        if epoch == 0:
            self.train_iter = self.train_loader.__iter__()
            if self.train_loader_neg is not None:
                self.train_iter_neg = self.train_loader_neg.__iter__()

        while (j <= 1000):
            try:
                sample = self.train_iter.__next__()
            except StopIteration:
                self.train_iter = self.train_loader.__iter__()
                sample = self.train_iter.__next__()

            Note, Num, Disease, Mask, Age, Demo = sample['Note'], sample['Num'], sample['Disease'], sample['Mask'], \
                                                  sample['Age'], sample['Demo']
            if self.train_loader_neg is not None:
                try:
                    sample_neg = self.train_iter_neg.__next__()
                except StopIteration:
                    self.train_iter_neg = self.train_loader_neg.__iter__()
                    sample_neg = self.train_iter_neg.__next__()
                
                Note_neg, Num_neg, Disease_neg, Mask_neg, Age_neg, Demo_neg = sample_neg['Note'], sample_neg['Num'], \
                                                                              sample_neg['Disease'], sample_neg['Mask'], \
                                                                              sample_neg['Age'], sample_neg['Demo']

                Note = torch.cat([Note, Note_neg], 0)
                Num = torch.cat([Num, Num_neg], 0)
                Disease = torch.cat([Disease, Disease_neg], 0)
                Mask = torch.cat([Mask, Mask_neg], 0)
                Age = torch.cat([Age, Age_neg], 0)
                Demo = torch.cat([Demo, Demo_neg], 0)

            nRec += Note.size()[0] - np.sum(Mask.numpy(), axis=0)[0]
            Note, Num, Disease, Mask, Age, Demo = Variable(Note).long(), Variable(Num).float(), Variable(
                Disease).float(), Variable(Mask).float(), Variable(Age).float(), Variable(Demo).float()


            self.cnt_iter += 1

            if self.flg_cuda:
                Note, Num, Disease, Mask, Age, Demo = Note.cuda(), Num.cuda(), Disease.cuda(), Mask.cuda(), Age.cuda(), Demo.cuda()

            self.optimizer.zero_grad()
            Disease = Disease.squeeze(1)
            Mask = Mask.squeeze(1)
            output = self.model(Note, Num, Disease, Mask, Age, Demo)
            output = output * (1.0 - Mask)
            Disease = Disease * (1.0 - Mask)
            loss = self.criterion(output, Disease)

            if self.alpha_L1 > 0:
                l1_crit = nn.L1Loss(size_average=False)
                for fc in self.model.FCs:
                    if self.flg_cuda:
                        target_reg = Variable(torch.zeros(fc.weight.size())).cuda()
                    else:
                        target_reg = Variable(torch.zeros(fc.weight.size()))
                    loss += l1_crit(fc.weight, target_reg) * self.alpha_L1

            loss.backward()
            if self.flg_gradClip:
                torch.nn.utils.clip_grad_norm(self.model.params, 0.5)
            self.optimizer.step()

            self.Y_multi_train.append(output.data.cpu().numpy())
            self.target_multi_train.append(Disease.data.cpu().numpy())
            self.mask_train.append(Mask.data.cpu().numpy())

            correct += self._getAccuracy(output, Disease, Mask)
            train_loss += loss.data[0]
            j += 1
            if (j % self.log_interval[1] == 0):
                train_loss_temp = train_loss / np.sum(nRec)
                print('Train Epoch: {}, Batch: {}, Loss: {:.4f}'.format(epoch, j, train_loss_temp))

        if (epoch == 0) | (epoch % self.log_interval[0] == 0) | (epoch == self.n_iter - 1):
            trainAccuracy = 100. * correct / nRec
            train_loss /= np.sum(nRec)
            lsTrainAccuracy.append(trainAccuracy)

            # print('\nTrain Epoch: {} Loss: {:.4f}, Accuracy: {}, {}, {}/{} ({:.2f}%, {:.2f}%, {:.2f}%)'.format(epoch, train_loss, correct[0], correct[1], correct[2], nRec, trainAccuracy[0], trainAccuracy[1], trainAccuracy[2]))
            print('\nTrain Epoch: {} Loss: {:.4f}'.format(epoch, train_loss))

            auc, f1, tp, p, tn, n = self._aucAll(self.Y_multi_train, self.target_multi_train, self.mask_train)
            for i in range(len(auc)):
                print(str(i) + ' Train set: Accuracy: pos_{}/{}, neg_{}/{} ({:.2f}%, AUC: {:.4f}, F1: {:.4f}%)'.format(
                    tp[i], p[i], tn[i], n[i], trainAccuracy[i], auc[i], f1[i] * 100))

    def _test(self, epoch, lsTestAccuracy):
        if (epoch == 0) | (epoch % self.log_interval[0] == 0) | (epoch == self.n_iter - 1):

            self.model.eval()
            test_loss = 0
            correct, nRec = np.zeros(3), np.zeros(3)

            self.Y_multi = []
            self.target_multi = []
            self.mask = []

            for batch_idx, sample in enumerate(self.test_loader):
                Note, Num, Disease, Mask, Age, Demo = sample['Note'], sample['Num'], sample['Disease'], sample['Mask'], \
                                                      sample['Age'], sample['Demo']
                nRec += Note.size()[0] - np.sum(Mask.numpy(), axis=0)[0]

                Note, Num, Disease, Mask, Age, Demo = Variable(Note, volatile=True).long(), Variable(Num,
                                                                                                     volatile=True).float(), Variable(
                    Disease, volatile=True).float(), \
                                                      Variable(Mask, volatile=True).float(), Variable(Age,
                                                                                                      volatile=True).float(), Variable(
                    Demo, volatile=True).float()
                if self.flg_cuda:
                    Note, Num, Disease, Mask, Age, Demo = Note.cuda(), Num.cuda(), Disease.cuda(), Mask.cuda(), Age.cuda(), Demo.cuda()
                with torch.no_grad():
                    output = self.model(Note, Num, Disease, Mask, Age, Demo)
                    Disease = Disease.squeeze(1)
                    Mask = Mask.squeeze(1)
                    output = output * (1.0 - Mask)
                    Disease = Disease * (1.0 - Mask)

                    test_loss += (self.criterion(output, Disease)).data[0]
                    correct += self._getAccuracy(output, Disease, Mask)

                self.Y_multi.append(output.data.cpu().numpy())
                self.target_multi.append(Disease.data.cpu().numpy())
                self.mask.append(Mask.data.cpu().numpy())

            testAccuracy = 100. * correct / nRec
            test_loss /= np.sum(nRec)  # loss function already averages over batch size
            auc, f1, tp, p, tn, n = self._aucAll(self.Y_multi, self.target_multi, self.mask)
            self.auc = np.mean(auc)
            print('Test set: Average loss: {:.4f}'.format(test_loss))
            for i in range(len(auc)):
                print(str(i) + ' Test set: Accuracy: pos_{}/{}, neg_{}/{} ({:.2f}%, AUC: {:.4f}, F1: {:.4f}%)'.format(
                    tp[i], p[i], tn[i], n[i], testAccuracy[i], auc[i], f1[i] * 100))
            lsTestAccuracy.append([testAccuracy, auc, f1])

    def _getAccuracy(self, output, target, mask):
        pred = (output.data > 0.5).float()
        accuracy = (pred.eq(target.data).cpu().float() * (1.0 - mask.data.cpu())).numpy()
        accuracy = np.sum(accuracy, axis=0)
        return accuracy

    def _aucAll(self, Y_hat, Y, mask):
        # Multi-task, inputs are multiple dimensions
        Y_hat = np.vstack(Y_hat)
        Y = np.vstack(Y)
        mask = np.vstack(mask)
        auc, f1, tp, p, tn, n = [], [], [], [], [], []
        for i in range(Y_hat.shape[1]):
            result = self._getAucF1(Y_hat[:, i], Y[:, i], mask[:, i])
            for j in range(len(result)):
                [auc, f1, tp, p, tn, n][j].append(result[j])
        return auc, f1, tp, p, tn, n

    def _getAucF1(self, Y_hat, Y, mask):
        # Single task
        Y_hat = Y_hat[mask == 0]
        Y = Y[mask == 0]
        tp = sum((Y == 1) & (Y_hat >= self.posThres))
        tn = sum((Y == 0) & (Y_hat < self.posThres))
        p = sum(Y == 1)
        n = sum(Y == 0)
        if len(np.unique(Y)) == 1:
            auc = 0
            f1 = 0
        else:
            auc = metrics.roc_auc_score(Y, Y_hat)
            f1 = metrics.f1_score(Y, 1 * (Y_hat > self.posThres))
        return auc, f1, tp, p, tn, n

    def _saveModel(self):
        torch.save(self.model, self.savePath + '_model.pt')

    def _drawAUC(self, Y_multi, target_multi, mask):
        Y_multi = np.vstack(Y_multi)
        target_multi = np.vstack(target_multi)
        mask = np.vstack(mask)
        pickle.dump([Y_multi, target_multi, mask], open(self.savePath + '_pred.p', 'wb'))
        """
        # Uncomment if matplotlib is available
        fpr, tpr, thresholds = metrics.roc_curve(Y, Y_hat, pos_label=1)
        prec, recall, thresholds = metrics.precision_recall_curve(Y, Y_hat, pos_label= 1)

        # Print ROC curve
        pp = PdfPages(self.savePath + '_plots.pdf')

        figure1 = plt.figure(figsize=(16, 6))
        gs1 = gridspec.GridSpec(1, 2)

        ax1 = figure1.add_subplot(gs1[0])
        ax1.plot(fpr, tpr, 'g-')
        ax1.plot([0, 1], [0, 1], ls="--")
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.set_ylabel('True positive rate', fontsize=12)
        ax1.set_xlabel('False positive rate', fontsize=12)
        ax1.set_title('ROC curve', fontsize=16)

        ax2 = figure1.add_subplot(gs1[1])
        ax2.plot(prec, recall, 'b-')
        ax2.plot([0, 1], [0, 1], ls="--")
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1])
        ax2.set_ylabel('Recall', fontsize=12)
        ax2.set_xlabel('Precision', fontsize=12)
        ax2.set_title('Precision-recall curve', fontsize=16)

        pp.savefig(figure1)
        plt.close(figure1)
        pp.close()
        """



