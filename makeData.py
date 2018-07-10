"""
Randomly generate synthetic data to run training / validation scripts

Output:
1. dfTrainPos.json (training set with positive examples)
2. dfTrainNeg.json (training set with negative examples)
3. dfDev.json (validation set)
4. embedding.p (embedding matrix with row ordered by the word index, first row all 0 as padding)
* In the synthetic data we simply take the first 1000 words in the embedding vector matrix. Please re-order the matrix based on the word index of your input data

Each file is formatted as a list of records. Each record is a patient's data containing encounters during the
12-month historical window and labels of the three target disease during the 6-month prediction window.
Each record has the following elements, an element can be a list of values or a value:
[Note, Num, Disease, Mask, Age, gender, race, ethnic]

1. Note: a list with each element an encounter, within each encounter the element are words (as converted to word index, starting from 1) in the note.
2. Num: a list with each element an encounter, within each encounter, the first, second, third 50 values are the min / median / max of the
    50 extracted, normalized labValues aggregated within this encounter for the corresponding patient, respectively. The last value is the days between
    the current encounter and the previous encounter. Thus there are 151 dimensions of numerical values at each encounter.
3. Disease: a list of 3 binary values, corresponding to whether the patient had CHF, KF, and stroke during the prediction window. (1 = yes, 0 = no)
4. Mask: a list of 3 binary values, value 1 indicates the corresponding record shouldn't be considered for the corresponding disease prediction.
5. Age: normalized value of age
6. Gender: index of gender, 0 is missing
7. Race: index of race, 0 is missing
8. Ethnic: index of ethnic, 0 is missing
"""

import numpy as np
import argparse
import json
import os
import util
import pickle
#import pdb

def makeNotes(maxIdx = 1000, maxDocLen = 800):
    '''
    Generate synthetic data of word indexes in one encounter's note.
    :param maxIdx: maximum number of word index
    :param maxDocLen: maximum note length
    '''
    docLen = np.random.choice(maxDocLen-1) + 1
    note = list(np.random.choice(maxIdx, size = docLen))
    note = [int(x) for x in note]
    return note


def makeNum(maxDays = 50):
    '''
    Generate synthetic data of the numerical values of one encounter
    :param maxDays: maximum days between two encounters
    '''
    labs = list(np.random.normal(0, 1, size = 150))
    days = np.random.choice(maxDays)
    labs.append(days)
    return labs

def makeBinary(ndim, lsPosPct):
    '''
    Generate a list of binary values
    :param ndim: number of binary values to generate
    :param lsPosPct: positive percentage of each value
    '''
    out = []
    for i in range(ndim):
        out.append(np.random.choice(2, p = [1-lsPosPct[i], lsPosPct[i]]))
    return out

def makeRecord(maxWordIdx, maxEncounter = 30):
    '''
    Generate synthetic data of each record
    :param maxEncounter: maximum number of encounters
    '''
    nEncounter = np.random.choice(maxEncounter-1) + 1 # minimum 1 encounter
    Notes, Num = [], []
    for i in range(nEncounter):
        Notes.append(makeNotes(maxIdx = maxWordIdx))
        Num.append(makeNum())

    Disease = makeBinary(ndim = 3, lsPosPct=[0.1, 0.1, 0.1])
    Mask = makeBinary(ndim = 3, lsPosPct=[0.05, 0.05, 0.05])

    Age = np.random.normal()
    Gender = np.random.choice(2)
    Race = np.random.choice(25)
    Ethnic = np.random.choice(29)
    return [Notes, Num, Disease, Mask, Age, Gender, Race, Ethnic]

def splitPos(dfTrain):
    dfTrainPos, dfTrainNeg = [], []
    for row in dfTrain:
        if max(row[2]) > 0:
            dfTrainPos.append(row)
        else:
            dfTrainNeg.append(row)
    return dfTrainPos, dfTrainNeg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nRec", type=int, default=1000) # Number of records to generate in train and dev, first half goes into training and second into dev
    parser.add_argument("--outPath", default = 'sampleData/')
    parser.add_argument("--randSeed", type=int, default=42)
    parser.add_argument("--embFile", default ='sspEmbedding_dim300.tsv') # Name of the embedding vector file
    args = parser.parse_args()

    if not os.path.isdir(args.outPath):
        os.mkdir(args.outPath)

    np.random.seed(args.randSeed)
    maxWordIdx = 1000

    df= []

    for i in range(args.nRec):
        df.append(makeRecord(maxWordIdx))

    n = int(args.nRec/2)
    dfTrain, dfDev = df[0:n], df[n:]
    dfTrainPos, dfTrainNeg = splitPos(dfTrain)

    json.dump(dfTrainPos, open(args.outPath + 'dfTrainPos.json','w'))
    json.dump(dfTrainNeg, open(args.outPath + 'dfTrainNeg.json','w'))
    json.dump(dfDev, open(args.outPath + 'dfDev.json','w'))

    w2v_vocab, vec = util.load_star_space(args.embFile, torch = False)
    embedding = np.zeros(shape=(maxWordIdx + 1, vec.shape[1]))
    embedding[1:] = vec[0:maxWordIdx]
    pickle.dump(embedding, open(args.outPath + 'embedding.p', 'wb'))