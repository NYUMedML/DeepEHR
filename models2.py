
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from sklearn import metrics
import pdb
import pickle

#import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec


class  CNN_Text(nn.Module):

    def __init__(self, embedding ,args):

        super(CNN_Text,self).__init__()

        self.args = args
        n_words = embedding.size()[0]
        emb_dim = embedding.size()[1]
        filters = args.h

        Ks = [ i for i in range(1, args.kernels + 1)]
        C = args.n_out

        self.embed = nn.Embedding( n_words, emb_dim)
        self.embed.weight = nn.Parameter(embedding,requires_grad=False)

        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, filters, K ) for K in Ks])

        self.bn = nn.ModuleList([nn.BatchNorm1d(filters) for K in Ks])

        self.fc = nn.Linear(len(Ks)* filters, C)

        self.params = list(self.fc.parameters())
        for c in self.convs:
            self.params += list(c.parameters())
        
        for b in self.bn:
            self.params += list(b.parameters())

        if args.train_embed:
            self.params += list(self.embed.parameters())


    def fc_layer( self, x, layer, bn = None):
        x = layer(x)

       
        if self.args.batch_norm:
            x = F.relu(bn(x))
        else:
            x = F.relu(x)
       
        x = F.dropout(x , self.args.dropout)
        return x

    def encoder(self,x):
        x = self.embed(x)
        x = x.transpose(1,2)

        h = [ self.fc_layer( x, self.convs[i], self.bn[i]) for i in range(len(self.convs) )]
        
        return h

    def forward(self, Note, Num, Disease, Mask, Age, Demo ):


        h = self.encoder(Note)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in h]

        x = torch.cat(x, 1)
        x = self.fc(x)
        
        return F.sigmoid(x)


class  CNN_Dense(nn.Module):

    def __init__(self, embedding ,args):

        super(CNN_Dense,self).__init__()

        self.args = args
        n_words = embedding.size()[0]
        emb_dim = embedding.size()[1]
        filters = args.h

        Ks = [ i for i in range(1, args.kernels + 1)]
        C = 3

        self.embed = nn.Embedding( n_words, emb_dim)
        self.embed.weight = nn.Parameter(embedding,requires_grad=False)

        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, filters, K ) for K in Ks])

        self.bn = nn.ModuleList([nn.BatchNorm1d(filters) for K in Ks])

        self.fc = nn.Linear(filters, C)
        
        self.dense = nn.Linear(len(Ks)*filters,   filters)
        self.dense2 = nn.Linear(len(Ks)*filters,  filters)

        self.params = list(self.fc.parameters())
        for c in self.convs:
            self.params += list(c.parameters())
        
        for b in self.bn:
            self.params += list(b.parameters())

        if args.train_embed:
            self.params += list(self.embed.parameters())

        self.params += list(self.dense.parameters())
        self.params += list(self.dense2.parameters())


    def fc_layer( self, x, layer, bn = None):
        x = layer(x)

       
        if self.args.batch_norm:
            x = F.relu(bn(x))
        else:
            x = F.relu(x)
       
        x = F.dropout(x , self.args.dropout)
        return x

    def encoder(self,x):
        x = self.embed(x)
        x = x.transpose(1,2)

        h = [ self.fc_layer( x, self.convs[i], self.bn[i]) for i in range(len(self.convs) )]
        
        return h

    def forward(self, x ):         


        h = self.encoder(x[:,:3000])
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in h]

        x = torch.cat(x, 1)
        x = F.relu(self.dense(x))
        #x = torch.cat([x,d],1)
        #x = self.dense2(x) + d
        #x = d
        #x = F.dropout(x , .3)

        x = self.fc(x)
        
        return F.sigmoid(x)




class  CNN_Text_Demo(nn.Module):

    def __init__(self, embedding ,args):

        super(CNN_Text_Demo,self).__init__()

        self.args = args
        n_words = embedding.size()[0]
        emb_dim = embedding.size()[1]
        filters = args.h
        self.n_demo_feat = args.n_demo_feat

        Ks = [ i for i in range(1, args.kernels+1)]
        C = 3

        self.embed = nn.Embedding( n_words, emb_dim)
        self.embed.weight = nn.Parameter(embedding,requires_grad=False)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, filters, K ) for K in Ks])

        self.bn = nn.ModuleList([nn.BatchNorm1d(filters) for K in Ks])
        self.dense = nn.Linear(len(Ks)*filters + self.n_demo_feat,   filters)
        self.fc = nn.Linear(filters, C)

        self.params = list(self.fc.parameters())
        for c in self.convs:
            self.params += list(c.parameters())
        
        for b in self.bn:
            self.params += list(b.parameters())

        if args.train_embed:
            self.params += list(self.embed.parameters())

        self.params += list(self.dense.parameters())

    def fc_layer( self, x, layer, bn = None):
        x = layer(x)
       
        if self.args.batch_norm:
            x = F.relu(bn(x))
        else:
            x = F.relu(x)
       
        x = F.dropout(x , self.args.dropout)
        return x

    def encoder(self,x):
        x = self.embed(x)
        x = x.transpose(1,2)

        h = [ self.fc_layer( x, self.convs[i], self.bn[i]) for i in range(len(self.convs) )]
        
        return h

    def forward(self, Note, Num, Disease, Mask, Age, Demo ):         

        Note, Num, Disease, Mask, Age, Demo
        text = Note
        demo = torch.cat([ Num, Age[:,0].float() , Demo[:,0].float() ] , 1)

        h = self.encoder(text)
        h = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in h]
       
        h.append(demo)
        h = torch.cat(h, 1)
       
        h = F.relu(self.dense(h))
        y_hat = self.fc(h)
        
        return F.sigmoid(y_hat)




class  CNN_H(nn.Module):

    def __init__(self, embedding ,args):

        super(CNN_H,self).__init__()

        self.args = args
        n_words = embedding.size()[0]
        emb_dim = embedding.size()[1]
        filters = args.h

        Ks = [ i for i in range(1, args.kernels)]
        C = 1

        self.embed = nn.Embedding( n_words, emb_dim)
        self.embed.weight = nn.Parameter(embedding,requires_grad=False)

        self.conv1 = nn.Conv1d(emb_dim, filters, 1 )
        self.bn1 = nn.BatchNorm1d(filters)
        
        self.conv2 = nn.Conv1d(filters, filters, 2 )
        self.bn2 = nn.BatchNorm1d(filters)

        self.conv3 = nn.Conv1d(filters, filters, 2 )
        self.bn3 = nn.BatchNorm1d(filters)

        #self.convs = nn.ModuleList([nn.Conv1d(filters, filters, 2 ) for K in Ks])
        #self.bn = nn.ModuleList([nn.BatchNorm1d(filters) for K in Ks])

        self.fc = nn.Linear( 3* filters, C)
        

        self.params = list(self.fc.parameters())
        
        for c in [ self.conv1,self.conv2,self.conv3,self.bn1,self.bn2,self.bn3]:
            self.params += list(c.parameters())

        #for c in self.convs:
        #    self.params += list(c.parameters())
        
        #for b in self.bn:
        #    self.params += list(b.parameters())

        if args.train_embed:
            self.params += list(self.embed.parameters())


    def fc_layer( self, x, layer, bn = None):
        x = layer(x)

       
        if self.args.batch_norm:
            x = F.relu(bn(x))
        else:
            x = F.relu(x)
       
        x = F.dropout(x , self.args.dropout)
        return x

    def encoder(self,x):
        x = self.embed(x)
        x = x.transpose(1,2)
        
        h1 = self.fc_layer( x, self.conv1, self.bn1)
        h2 = self.fc_layer( h1, self.conv2, self.bn2)
        h3 = self.fc_layer( h2, self.conv3, self.bn3)

        #h1 = F.max_pool1d(h1,h1.size(2)).squeeze(2)
        #h2 = F.max_pool1d(h2,h2.size(2)).squeeze(2)
        #h3 = F.max_pool1d(h3,h3.size(2)).squeeze(2)

        #h = [ self.fc_layer( x, self.convs[i], self.bn[i]) for i in range(len(self.convs) )]
        return [h1,h2,h3]

    def forward(self, x ):         


        h = self.encoder(x)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in h]

        x = torch.cat(x, 1)

        x = self.fc(x)
        
        return F.sigmoid(x)

class  CNN_NonNeg(nn.Module):

    def __init__(self, embedding ,args):

        super(CNN_NonNeg,self).__init__()

        self.args = args
        n_words = embedding.size()[0]
        emb_dim = embedding.size()[1]
        filters = args.h

        Ks = [ i for i in range(1, args.kernels+1)]
        C = 1

        self.embed = nn.Embedding( n_words, emb_dim)
        self.embed.weight = nn.Parameter(embedding,requires_grad=False)

        self.conv1 = nn.Conv1d(emb_dim, filters, 1 )
        self.convs = nn.ModuleList([nn.Conv1d(filters, filters, K ) for K in Ks])
        self.bn = nn.ModuleList([nn.BatchNorm1d(filters) for K in Ks])

        self.params = []
        
        self.W = nn.Parameter(.01*torch.randn(  len(Ks)* filters ,1),requires_grad=True)
        self.b = nn.Parameter( torch.zeros( 1 ,1),requires_grad=True)

        self.params += [self.W , self.b]
        self.params += list(self.conv1.parameters())

        for c in self.convs:
            self.params += list(c.parameters())
        
        for b in self.bn:
            self.params += list(b.parameters())

        if args.train_embed:
            self.params += list(self.embed.parameters())


    def fc_layer( self, x, layer, bn = None):
        x = layer(x)
       
        if self.args.batch_norm:
            x = F.relu(bn(x))
        else:
            x = F.relu(x)
       
        x = F.dropout(x , self.args.dropout)
 
        return x

    def encoder(self,x):
        x = self.embed(x)
        x = x.transpose(1,2)
        x = F.relu( self.conv1(x))

        h = [ self.fc_layer( x, self.convs[i], self.bn[i]) for i in range(len(self.convs) )]

        return h

    def forward(self, x ):         

        h = self.encoder(x)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in h]

        x = torch.cat(x, 1)

        x = torch.matmul( x,  torch.abs(self.W))
        x = x + self.b.expand_as(x)

        return F.sigmoid(x)

 


class DWAN(nn.Module):

    def __init__(self, embedding ,args):

        super(DWAN,self).__init__()

        n_words = embedding.size()[0]
        emb_dim = embedding.size()[1]
        
        self.args = args
        h = args.h
        C= 1

        self.h = h
        self.embed = nn.Embedding( n_words, emb_dim)
        self.embed.weight = nn.Parameter(embedding,requires_grad=False)

        self.fc1 = nn.Linear(emb_dim, h)
        self.fc2 = nn.Linear(h,h)
        self.fc3 = nn.Linear(h,h)
        self.fc4 = nn.Linear(h, C)


        self.bn0 = nn.BatchNorm1d(self.h)
        self.bn1 = nn.BatchNorm1d(self.h)
        self.bn2 = nn.BatchNorm1d(self.h)
        self.bn3 = nn.BatchNorm1d(self.h)


        self.params = list(self.fc1.parameters()) + list(self.fc2.parameters())  + list(self.fc4.parameters()) + list(self.bn1.parameters()) + list(self.bn2.parameters() ) + list(self.bn0.parameters() ) 



    def fc_layer( self, x, layer, bn = None):
        x = layer(x)

        if self.args.batch_norm:

            x = F.relu(bn(x))
        else:
            x = F.relu(x)

        x = F.dropout(x , self.args.dropout)
        return x


    def forward(self, x):
        x = self.embed(x)
        x = x.transpose(1,2)

        x = x.mean(2)

        x = self.fc_layer(x, self.fc1, self.bn1)
        x = self.fc_layer(x, self.fc2, self.bn2)
        x = self.fc_layer(x, self.fc3, self.bn3)

        x = self.fc4(x)

        return F.sigmoid(x)


class LSTM_Text(nn.Module):

    def __init__(self, embedding , args):

        super(LSTM_Text,self).__init__()
        n_words = embedding.size()[0]        
        emb_dim = embedding.size()[1]
        self.n_demo_feat = 61
        self.args = args

        C = 3
        h = args.h
        self.h = h
        self.embed = nn.Embedding( n_words, emb_dim)
        self.embed.weight = nn.Parameter(embedding,requires_grad=False)
        self.dense = nn.Linear(512 + self.n_demo_feat,   256)

        self.lstm = nn.GRU(emb_dim, h, 1, batch_first=True, bidirectional = args.bidir, dropout= args.dropout)

        if args.bidir:
            self.fc = nn.Linear(h, C)
        else:
            self.fc = nn.Linear(h, C)
        
        self.params = list(self.lstm.parameters()) + list(self.fc.parameters())
        self.params += list(self.dense.parameters())

    def forward(self,x):
                
        E = self.embed(x[:,:3000])
        demo = x[:,3000:3061].float()

        if self.args.bidir:
            h0 = Variable(torch.zeros(2, x.size()[0], self.h))
            if self.args.gpu:
                h0 = h0.cuda()

            z = self.lstm(E, h0)[0]
            z = z.max(1)[0]

        else:
            h0 = Variable(torch.zeros(1, x.size()[0], self.h))
            if self.args.gpu:
                h0 = h0.cuda()

            z = self.lstm(E, h0)[0][:, -1, :]

        z = torch.cat([z,demo],1)
        z = F.relu(self.dense(z))

        y_hat = self.fc(z)
        
        return F.sigmoid(y_hat)


class LSTM_TR(nn.Module):

    def __init__(self, embedding):

        super(LSTM_TR,self).__init__()
        n_words = embedding.size()[0]        
        emb_dim = embedding.size()[1]

        C = 1
        h = 128

        self.h = h
        self.C = C
        self.embed = nn.Embedding( n_words, emb_dim)
        self.embed.weight = nn.Parameter(embedding,requires_grad=False)
        
        self.lstm = nn.GRU(emb_dim, h, 1, batch_first=True)

        self.fc = nn.Conv1d(h, C , 1 ) 

        self.params = list(self.lstm.parameters()) + list(self.fc.parameters())

    def forward(self,x):
                
        h0 = Variable(torch.zeros(1, x.size()[0], self.h)).cuda()
        E = self.embed(x)
        z = self.lstm(E, h0)[0]
        z = z.transpose(1, 2)

        y_hat = self.fc(z)
        
        y_hat = y_hat.mean(2).squeeze()
    
        y_hat = F.sigmoid(y_hat)

        return y_hat


    def predict(self,x):

        y_hat = self.forward(x)

        return y_hat.mean(2).squeeze()

        

class SentCNN(nn.Module):

    def __init__(self, embedding,sent_len , doc_len , args):

        super(SentCNN,self).__init__()

        self.sent_len = sent_len
        self.doc_len = doc_len

        n_words = embedding.size()[0]
        emb_dim = embedding.size()[1]
        self.embed = nn.Embedding( n_words, emb_dim)
        self.embed.weight = nn.Parameter(embedding,requires_grad=False)
        self.emb_dim = emb_dim

        self.args = args
        filters = 128
        Ks = [1,2,3]
        #self.convs = nn.ModuleList([nn.Conv1d(emb_dim, filters, K ) for K in Ks])
        self.convs = nn.ModuleList([nn.Conv1d(filters, filters, K ) for K in Ks])
        self.h = filters
        self.bn = nn.ModuleList([nn.BatchNorm1d(filters) for K in Ks])

        self.fc = nn.Linear(len(Ks)* filters, 1)

        self.params = list(self.fc.parameters())
        for c in self.convs:
            self.params += list(c.parameters())

        for b in self.bn:
            self.params += list(b.parameters())



        self.gru = nn.GRU(emb_dim, filters , 1, batch_first=True)
        self.params += list(self.gru.parameters())

    def fc_layer( self, x, layer, bn = None):
        x = layer(x)

        if self.args.batch_norm:

            x = F.relu(bn(x))
        else:
            x = F.relu(x)

        x = F.dropout(x , self.args.dropout)
        return x


    def forward(self,x):
        
        batch_size = x.size()[0]
        x= x.view(-1,self.sent_len)

        E= self.embed(x)
        h0 = Variable(torch.zeros(1, E.size()[0], self.h)).cuda()
        E = self.gru(E, h0)[1]
        E = torch.squeeze(E)
        E = torch.chunk(E, batch_size , 0)
        E = torch.stack(E)

        x = E
        #x = E.sum(2)
        x = x.transpose(1,2)


        x = [ self.fc_layer( x, self.convs[i], self.bn[i]) for i in range(len(self.convs) )]

        #x = [F.relu(conv(x)) for conv in self.convs]
        #x = [ fc_layer( self, x,  self.convs[i], self.bn[i] for i in range(3)]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.fc(x)

        return F.sigmoid(x)



class SentLSTM(nn.Module):

    def __init__(self, embedding,sent_len , doc_len , args):

        super(SentLSTM,self).__init__()

        self.sent_len = sent_len
        self.doc_len = doc_len

        self.args = args
        n_words = embedding.size()[0]
        emb_dim = embedding.size()[1]
        self.embed = nn.Embedding( n_words, emb_dim)
        self.embed.weight = nn.Parameter(embedding,requires_grad=False)
        self.emb_dim = emb_dim
        
        h = 128
        self.h = h
        C = 1
        self.lstm = nn.GRU(emb_dim, h, 1, batch_first=True, bidirectional = args.bidir, dropout= args.dropout)

        if args.bidir:
            self.fc = nn.Linear(2*h, C)
        else:
            self.fc = nn.Linear(h, C)
        
        self.params = list(self.lstm.parameters()) + list(self.fc.parameters())


    def forward(self,x):
                
        batch_size = x.size()[0]
        x= x.view(-1,self.sent_len)

        E= self.embed(x)

        E = torch.chunk(E, batch_size , 0)
        E = torch.stack(E)
        
        E = E.sum(2)
        #x = x.transpose(1,2)

        if self.args.bidir:
            
            h0 = Variable(torch.zeros(2, E.size()[0], self.h))
            if self.args.gpu:
                h0 = h0.cuda()

            z = self.lstm(E, h0)[0]
            #z = F.max_pool1d(z, x.shape[1])
            z = torch.cat([ z[0].squeeze() , z[1].squeeze()] , 1)

        else:
            h0 = Variable(torch.zeros(1, x.size()[0], self.h))
            if self.args.gpu:
                h0 = h0.cuda()

            z = self.lstm(E, h0)[0][:, -1, :]

        y_hat = self.fc(z)
        
        return F.sigmoid(y_hat)

#============= Wrap training into a class ================
class trainModel(object):
    def __init__(self, train_paras, train_loader_pos, test_loader, model, optimizer, train_loader_neg = None):
        self.train_loader = train_loader_pos # If no train_loader_neg, then pass the train_loader here; otherwise split _pos and _neg
        self.train_loader_neg = train_loader_neg
        self.test_loader = test_loader
        self.model = model
        self.optimizer = optimizer
        
        #self.train_paras = train_paras
        self.n_iter = train_paras.get('n_iter', 1)
        self.log_interval = train_paras.get('log_interval', 5) # list of two numbers: [log_per_n_epoch, log_per_n_batch]
        self.flg_cuda = train_paras.get('flg_cuda', False)
        
        self.max_len = train_paras.get('max_len', 2000) # Max length of input
        self.lr_decay = train_paras.get('lr_decay', None) # List of 4 numbers: [init_lr, lr_decay_rate, lr_decay_interval, min_lr]
        self.flgSave =  train_paras.get('flgSave', False) # Set to true if save model
        self.savePath = train_paras.get('savePath', './')
        self.posThres = train_paras.get('posThres', 0.5)
        self.alpha_wneg = train_paras.get('alpha_wneg', 0.0) # Regularization coefficient on negative weights

        if self.lr_decay:
            assert len(self.lr_decay) == 4 # Elements include: [starting_lr, decay_multiplier, decay_per_?_epoch, min_lr]
        #self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion = torch.nn.BCELoss()
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
                self._drawAUC(self.Y_hat, self.Y)
                if self.flgSave:
                    self._saveModel()
            
        return self.model, lsTrainAccuracy, lsTestAccuracy

    def _train(self, epoch, lsTrainAccuracy):
        correct, train_loss = 0, 0
        self.model.train()
        if self.lr_decay:
            lr = min(self.lr_decay[0] * (self.lr_decay[1] ** (epoch // self.lr_decay[2])), self.lr_decay[3])
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        #pdb.set_trace()
        j,nRec = 0,0
        if epoch == 0:
            if self.train_loader_neg is not None:
                self.train_iter_neg = self.train_loader_neg.__iter__()
        
        for data, target in self.train_loader:
            if self.train_loader_neg is not None:
                try:
                    data_neg, target_neg = self.train_iter_neg.__next__()
                except StopIteration:
                    self.train_iter_neg = self.train_loader_neg.__iter__()
                    data_neg, target_neg = self.train_iter_neg.__next__()
                
                data = torch.cat([data, data_neg], 0)
                target = torch.cat([target, target_neg], 0)

            data = data[:,-self.max_len:]
            nRec += data.size()[0]
            data, target = Variable(data).long(), Variable(target.unsqueeze(1)).float()
            
            self.cnt_iter += 1

            if self.flg_cuda:
                data, target = data.cuda(), target.cuda()
            
            self.optimizer.zero_grad()
            
            output = self.model(data)
            loss = self.criterion(output, target) 
            if self.alpha_wneg > 0:
                l1_crit = nn.L1Loss(size_average=False)
                w_pos = self.model.fc.weight + torch.abs(self.model.fc.weight)
                if self.flg_cuda:
                    target_reg = Variable(torch.zeros(w_pos.size())).cuda()
                else:
                    target_reg = Variable(torch.zeros(w_pos.size()))
                loss += l1_crit(w_pos, target_reg) * self.alpha_wneg
            
            loss.backward()
            self.optimizer.step()
            correct += self._getAccuracy(output, target) 
            train_loss += loss.data[0] * data.size()[0]
            j += 1
            if (j % self.log_interval[1] == 0):
                print('Train Epoch: {}, Batch: {}, Loss: {:.4f}'.format(epoch, j, train_loss/nRec))
        
        if (epoch == 0) | (epoch % self.log_interval[0] == 0) | (epoch == self.n_iter - 1):
            trainAccuracy = 100. * correct / nRec
            train_loss /= nRec
            lsTrainAccuracy.append(trainAccuracy)

            print('Train Epoch: {} Loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                epoch, train_loss, correct, nRec, trainAccuracy))    
    
    def _test(self, epoch, lsTestAccuracy):
        if (epoch == 0) | (epoch % self.log_interval[0] == 0) | (epoch == self.n_iter - 1):

            self.model.eval()
            test_loss = 0
            correct = 0
            
            self.Y_hat = []
            self.Y = []
            
            for data, target in self.test_loader:
                data = data[:,-self.max_len:]
                data, target = Variable(data).long(), Variable(target.unsqueeze(1)).float()
                if self.flg_cuda:
                    data, target = data.cuda(), target.cuda()
                
                output = self.model(data)
                test_loss += self.criterion(output, target).data[0]
                correct += self._getAccuracy(output, target)
    
                self.Y_hat.append(output.cpu().data.numpy())
                self.Y.append( np.expand_dims(target.cpu().data.numpy(),1 ))
               
            testAccuracy = 100. * correct / len(self.test_loader.dataset)
            test_loss /= len(self.test_loader) # loss function already averages over batch size
            auc, f1, tp, p, tn, n = self._getAucF1(self.Y_hat, self.Y)
            self.auc = auc
            print('Test set: Average loss: {:.4f}, Accuracy: pos_{}/{}, neg_{}/{} ({:.2f}%, AUC: {:.4f}, F1: {:.4f}%)\n'.format(
                test_loss, tp, p, tn, n, testAccuracy, auc, f1 *100))
            lsTestAccuracy.append([testAccuracy, auc, f1])
    
    def _getAccuracy(self, output, target):
        pred = (output.data  > 0.5).float()
        accuracy = pred.eq(target.data).cpu().sum()
        return accuracy

    def _getAucF1(self, Y_hat, Y):
        Y_hat = np.vstack(Y_hat).reshape(-1)
        Y = np.vstack(Y).reshape(-1)
        tp = sum((Y == 1) & (Y_hat >= self.posThres))
        tn = sum((Y == 0) & (Y_hat < self.posThres))
        p = sum(Y == 1)
        n = sum(Y == 0)
        if len(np.unique(Y)) == 1:
            auc = 0
            f1 = 0
        else:
            auc = metrics.roc_auc_score(Y, Y_hat)
            f1 = metrics.f1_score(Y, 1*(Y_hat>self.posThres))
        return auc, f1, tp, p, tn, n

    def _saveModel(self):
        torch.save(self.model, self.savePath + '_model.pt')
    
    def _drawAUC(self, Y_hat, Y):
        Y_hat = np.vstack(Y_hat).reshape(-1)
        Y = np.vstack(Y).reshape(-1)
        pickle.dump([Y_hat, Y], open(self.savePath + '_pred.p', 'wb'))
        """
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
        





'''
sent_len = 20
doc_len = 100
#x_test = Variable(torch.ones(32,doc_len , sent_len)).long().cuda()
x_test = Variable(torch.from_numpy(np.random.randint(0,5,(32 ,doc_len ,  sent_len) ))).long().cuda()


#test_emb = torch.ones(64,200)
test_emb = torch.from_numpy(np.random.randint(0,5,(64,200) )).float()

model = SentCNN(test_emb,sent_len, doc_len)

model  = model.cuda()
print(model(x_test))


x_test = Variable(torch.ones(32,100)).long().cuda()
y_test = Variable(torch.ones(32)).long().cuda()

test_emb = torch.ones(100,200)

#model = LSTM_TR(test_emb)

model = DWAN(test_emb)
model = model.cuda()

#print(model(x_test).size())

print(model(x_test).size())

#criterion = torch.nn.CrossEntropyLoss()
#print(model.tr_cost(x_test, y_test, criterion))


'''
