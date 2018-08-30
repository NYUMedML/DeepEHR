# DeepEHR

## Synopsis
Pre-trained embeddings and scripts to run / evaluate model as shown in Paper: [link to Arxiv](https://arxiv.org/abs/1808.04928)
```
@article{liu2018deep,
  title={Deep EHR: Chronic Disease Prediction Using Medical Notes},
  author={Liu, Jingshu and Zhang, Zachariah and Razavian, Narges},
  journal={arXiv preprint arXiv:1808.04928},
  year={2018}
}

or 

@article{liu2018deep,
  title={Deep EHR: Chronic Disease Prediction Using Medical Notes},
  author={Liu, Jingshu and Zhang, Zachariah and Razavian, Narges},
  journal={Journal of Machine Learning Research (JMLR)},
  conference={Machine Learning in Healthcare},
  year={2018}
}

```

## Misc Details:
- [Detailed descriptions and illustrations of available models](https://github.com/NYUMedML/DeepEHR/wiki/Models)
- [Starspace embedding](https://github.com/NYUMedML/DeepEHR/wiki/Starspace-embedding)
- [Lab value extraction](https://github.com/NYUMedML/DeepEHR/wiki/Lab-value-extraction)

## Running scripts:

### System requirement
- python 3.6 (We do not guarantee but it may also work with 3.5)
- pytorch 0.4

### Generate synthetic data

By default, the following code will create a directory 'sampleData' under the root directory, and the generated  random synthetic data are saved in the directory.

```
python3 -u makeData.py
```

### Train models
The following shows the training statement of the encounter level CNN-LSTM model; training statement of other models are available in the run.sh file. You can find all potential input arguments and their definitions in the trainModel.py script.

```
python3 -u trainModel.py --randSeed 100 --i 1 --modelName Enc_CNN_LSTM --dimLSTM 256 --p_dropOut 0.5 --batch_norm --flg_cuda --logInterval 1 --n_iter 2 --batchSizePos 8 --batchSizeNeg 8 --nK 5 --filters 128 --enc_len 30 --doc_len 800 --lr 0.001 --lr_decay3 10 --inputPath sampleData/ --flgSave --savePath Enc_CNN_LSTM_Output/
```
If successful, you would see the following outputs:
```
General parameters:  Namespace(alpha_L1=0.0, batchSizeNeg=8, batchSizePos=8, batch_norm=True, bidir=False, dimLSTM=256, dimLSTM_num=128, doc_len=800, emb_dim=300, enc_len=30, filters=128, flgBias=False, flgSave=True, flg_AllLSTM=False, flg_cuda=True, flg_gradClip=False, flg_useNum=False, i=1, inputPath='sampleData/', logInterval=1, lr=0.001, lr_decay3=10, modelName='Enc_CNN_LSTM', nClassEthnic=29, nClassGender=2, nClassRace=25, nK=5, n_iter=2, num_workers=4, p_dropOut=0.5, posThres=0.5, randSeed=100, randn_std=None, rnnType='GRU', savePath='Enc_CNN_LSTM_Output/', train_embed=False)
Loading Data
Loaded:  dfTrainPos.json
Loaded:  dfTrainNeg.json
Loaded:  dfDev.json
To Loader
Model parameters:  {'enc_len': 30, 'doc_len': 800, 'flg_updateEmb': False, 'flg_bn': True, 'rnnType': 'GRU', 'bidir': False, 'p_dropOut': 0.5, 'lsDim': [256, 256, 3], 'dimLSTM': 256, 'flg_cuda': True, 'filters': 128, 'Ks': [1, 2, 3, 4, 5], 'randn_std': None, 'lastRelu': True, 'flgBias': False, 'flg_AllLSTM': False, 'flg_useNum': False, 'dimLSTM_num': 128}
Enc_CNN_LSTM(
  (embed): Embedding(1001, 300)
  (convs): ModuleList(
    (0): Conv1d(300, 128, kernel_size=(1,), stride=(1,))
    (1): Conv1d(300, 128, kernel_size=(2,), stride=(1,))
    (2): Conv1d(300, 128, kernel_size=(3,), stride=(1,))
    (3): Conv1d(300, 128, kernel_size=(4,), stride=(1,))
    (4): Conv1d(300, 128, kernel_size=(5,), stride=(1,))
  )
  (bn_conv): ModuleList(
    (0): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (4): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (lstm): GRU(640, 256, bias=False, batch_first=True, dropout=0.5)
  (FCs): ModuleList(
    (0): Linear(in_features=256, out_features=256, bias=True)
    (1): Linear(in_features=256, out_features=3, bias=True)
  )
  (bns): ModuleList(
    (0): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
)
Beginning Training

Train Epoch: 0, Batch: 1000, Loss: 0.0107

Train Epoch: 0 Loss: 0.0107
0 Train set: Accuracy: pos_2411/2558, neg_11638/12745 (91.81%, AUC: 0.9825, F1: 79.3614%)
1 Train set: Accuracy: pos_2880/3078, neg_11585/12496 (92.88%, AUC: 0.9844, F1: 83.8550%)
2 Train set: Accuracy: pos_2841/2893, neg_10455/12273 (87.67%, AUC: 0.9884, F1: 75.2383%)
Test set: Average loss: 0.0112
0 Test set: Accuracy: pos_1/37, neg_424/440 (89.10%, AUC: 0.4832, F1: 3.7037%)
1 Test set: Accuracy: pos_2/45, neg_416/423 (89.32%, AUC: 0.5076, F1: 7.4074%)
2 Test set: Accuracy: pos_0/44, neg_405/431 (85.26%, AUC: 0.5540, F1: 0.0000%)
Train Epoch: 1, Batch: 1000, Loss: 0.0058

Train Epoch: 1 Loss: 0.0058
0 Train set: Accuracy: pos_2576/2576, neg_12677/12677 (100.00%, AUC: 1.0000, F1: 100.0000%)
1 Train set: Accuracy: pos_2971/2971, neg_12606/12606 (100.00%, AUC: 1.0000, F1: 100.0000%)
2 Train set: Accuracy: pos_2925/2926, neg_12249/12250 (99.99%, AUC: 1.0000, F1: 99.9658%)
Test set: Average loss: 0.0095
0 Test set: Accuracy: pos_0/37, neg_438/440 (91.82%, AUC: 0.4670, F1: 0.0000%)
1 Test set: Accuracy: pos_1/45, neg_414/423 (88.68%, AUC: 0.5134, F1: 3.6364%)
2 Test set: Accuracy: pos_0/44, neg_428/431 (90.11%, AUC: 0.5754, F1: 0.0000%)
Test accuracy max: 0.519
Test accuracy final: 0.519
Stop at: 1
```
The 0, 1, 2 Training/Test set Accuracy/AUC/F1 entries are the training / test performance metrics for the three disease types, respectively. The order is the same as your input disease vector.

(Since we run the model on randomly generated synthetic data, the test AUC is around random)
