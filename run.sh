#!/bin/bash

# Encounter CNN-LSTM
python3 -u trainModel.py --randSeed 100 --i 1 --modelName Enc_CNN_LSTM --dimLSTM 256 --p_dropOut 0.5 --batch_norm --flg_cuda --logInterval 1 --n_iter 2 --batchSizePos 8 --batchSizeNeg 8 --nK 5 --filters 128 --enc_len 30 --doc_len 800 --lr 0.001 --lr_decay3 10 --inputPath sampleData/ --flgSave --savePath Enc_CNN_LSTM_Output/

# LSTM baseline with lab/demographics
python3 -u trainModel.py --i 2 --modelName DemoLab --dimLSTM 256 --p_dropOut 0.5 --batch_norm --flg_cuda --logInterval 1 --n_iter 2 --batchSizePos 8 --batchSizeNeg 8 --enc_len 30 --lr 0.001 --lr_decay3 10 --inputPath sampleData/ --flgSave --savePath DemoLab_Output/

# Encounter CNN-LSTM with lab and demographics
python3 -u trainModel.py --flg_useNum --i 1 --modelName Enc_CNN_LSTM_DemoLab --dimLSTM 512 --p_dropOut 0.5 --batch_norm --flg_cuda --logInterval 1 --n_iter 2 --batchSizePos 8 --batchSizeNeg 8 --nK 5 --filters 128 --enc_len 30 --doc_len 800 --lr 0.001 --lr_decay3 10 --inputPath sampleData/ --flgSave --savePath EncAll_Output/

