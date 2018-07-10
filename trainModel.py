import torch
# import torch.nn as nn
# import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
# import torchwordemb
import torch.optim as optim

import sys
import time
import gc
import pickle
import os

import enc_model as m3
# import pandas as pd
# from util import *

import torch.utils.data

# from sklearn.metrics import auc
# from sklearn import metrics

'''
General Training Script for PyTorch Models
-- Modified to accommodate more flexible LSTM structure

'''


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--nClassGender", type = int, default=2) # Number of classes in gender variable
    parser.add_argument("--nClassRace", type=int, default=25)  # Number of classes in race variable
    parser.add_argument("--nClassEthnic", type=int, default=29)  # Number of classes in ethnic variable

    parser.add_argument("--modelName", default="Enc_CNN_LSTM")
    parser.add_argument("--dimLSTM", type=int, default=128)  # LSTM dimension
    parser.add_argument("--dimLSTM_num", type=int, default=128)  # LSTM dimension for numericals
    parser.add_argument("--p_dropOut", type=float, default=.5)
    parser.add_argument("--batch_norm", action='store_true')
    parser.add_argument("--bidir", action='store_true')
    parser.add_argument("--train_embed", action='store_true')
    parser.add_argument("--rnnType", default="GRU")
    parser.add_argument("--enc_len", type=int, default=20)
    parser.add_argument("--doc_len", type=int, default=1000)

    parser.add_argument("--n_iter", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_decay3", type=int, default=10)  # Decay learning rate every lr_decay3 epochs
    parser.add_argument("--i", type=int, default=1)  # Index of the element in the parameter set to be tuned
    parser.add_argument("--batchSizePos", type=int, default=16)
    parser.add_argument("--batchSizeNeg", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--flg_cuda", action='store_true')
    parser.add_argument("--emb_dim", type=int, default=300)  # Embedding dimension
    parser.add_argument("--logInterval", type=int, default=1)  # Print test accuracy every n epochs
    parser.add_argument("--flgSave", action='store_true')
    parser.add_argument("--savePath", default='./')
    parser.add_argument("--filters", type=int, default=128)
    parser.add_argument("--nK", type=int, default=3)  # Number of kernels
    parser.add_argument("--randSeed", type=int, default=42)
    parser.add_argument("--posThres", type=float, default=0.5)
    parser.add_argument("--inputPath", default="/ifs/data/razavianlab/encSeq_input/dim50/")
    parser.add_argument("--alpha_L1", type=float, default=0.0)
    parser.add_argument("--randn_std", type=float, default=None)
    parser.add_argument("--flgBias", action='store_true')
    parser.add_argument("--flg_gradClip", action='store_true')
    parser.add_argument("--flg_AllLSTM", action='store_true')
    parser.add_argument("--flg_useNum", action='store_true')

    args = parser.parse_args()

    torch.manual_seed(args.randSeed)  # For reproducible results
    if args.flgSave:
        if not os.path.isdir(args.savePath):
            os.mkdir(args.savePath)

    # args.d = ['chf', 'kf', 'str'][args.i -1]
    # lsAlpha = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    # args.alpha_L1 = lsAlpha[args.i -1]

    if args.flg_AllLSTM:
        dimLSTM = args.dimLSTM * args.enc_len
    else:
        dimLSTM = args.dimLSTM

    lsDim = [[dimLSTM, 256, 3], [dimLSTM, 512, 256, 3]][args.i-1]
    print('General parameters: ', args)

    unique = False

    print("Loading Data")
    # if args.modelName in ['Enc_SumLSTM', 'Enc_CNN_LSTM']:
    embedding = pickle.load(open(args.inputPath + 'embedding.p', 'rb'))
    embedding = torch.from_numpy(embedding).float()
    trainset_pos = m3.encDataset(args.inputPath, 'dfTrainPos.json', args.nClassGender, args.nClassRace, args.nClassEthnic,
                                 transform=m3.padOrTruncateToTensor(args.enc_len, args.doc_len))
    trainset_neg = m3.encDataset(args.inputPath, 'dfTrainNeg.json', args.nClassGender, args.nClassRace, args.nClassEthnic,
                                 transform=m3.padOrTruncateToTensor(args.enc_len, args.doc_len))
    testset = m3.encDataset(args.inputPath, 'dfDev.json', args.nClassGender, args.nClassRace, args.nClassEthnic,
                            transform=m3.padOrTruncateToTensor(args.enc_len, args.doc_len))

    print('To Loader')
    if args.flg_cuda:
        train_loader_pos = torch.utils.data.DataLoader(trainset_pos, batch_size=args.batchSizePos, shuffle=True,
                                                       pin_memory=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batchSizePos + args.batchSizeNeg,
                                                  shuffle=False, pin_memory=True)
        if trainset_neg is not None:
            train_loader_neg = torch.utils.data.DataLoader(trainset_neg, batch_size=args.batchSizeNeg, shuffle=True,
                                                           pin_memory=True)
    else:
        train_loader_pos = torch.utils.data.DataLoader(trainset_pos, batch_size=args.batchSizePos, shuffle=True,
                                                       pin_memory=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batchSizePos + args.batchSizeNeg,
                                                  shuffle=False, pin_memory=False)
        if trainset_neg is not None:
            train_loader_neg = torch.utils.data.DataLoader(trainset_neg, batch_size=args.batchSizeNeg, shuffle=True,
                                                           pin_memory=False)

    model_paras = {'enc_len': args.enc_len, 'doc_len': args.doc_len, 'flg_updateEmb': args.train_embed,
                   'flg_bn': args.batch_norm,
                   'rnnType': args.rnnType, 'bidir': args.bidir, 'p_dropOut': args.p_dropOut, 'lsDim': lsDim,
                   'dimLSTM': args.dimLSTM,
                   'flg_cuda': args.flg_cuda, 'filters': args.filters, 'Ks': [i + 1 for i in range(args.nK)],
                   'randn_std': args.randn_std, 'lastRelu': True, 'flgBias': args.flgBias,
                   'flg_AllLSTM': args.flg_AllLSTM,
                   'flg_useNum': args.flg_useNum, 'dimLSTM_num': args.dimLSTM_num}

    print('Model parameters: ', model_paras)

    model = getattr(m3, args.modelName)(model_paras, embedding)
    # model.apply(weights_init)

    if args.flg_cuda:
        model = model.cuda()

    print(model)

    opt = optim.Adam(model.params, lr=args.lr)

    print("Beginning Training")
    train_paras = {'n_iter': args.n_iter, 'log_interval': [args.logInterval, 1000], 'flg_cuda': args.flg_cuda,
                   'lr_decay': [args.lr, 0.9, args.lr_decay3, 1e-5],
                   'flgSave': args.flgSave, 'savePath': args.savePath, 'posThres': args.posThres,
                   'alpha_L1': args.alpha_L1, 'flg_gradClip': args.flg_gradClip}

    m = m3.trainModel(train_paras, train_loader_pos, test_loader, model, opt, train_loader_neg=train_loader_neg)

    _, lsTrainAccuracy, lsTestAccuracy = m.run()
    testAuc = [np.mean(x[1]) for x in lsTestAccuracy]
    print('Test AUC max: %.3f' % (max(testAuc)))
    print('Test AUC final: %.3f' % (testAuc[-1]))
    stopIdx = min(testAuc.index(max(testAuc)) * args.logInterval, args.n_iter)
    print('Stop at: %d' % (stopIdx))
