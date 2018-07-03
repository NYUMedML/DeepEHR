#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 07:47:34 2017

@author: jshliu

Util scripts
"""

import os
import pandas as pd
import numpy as np
import pprint
import tarfile
import copy
import re
import torch
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import pdb
from datetime import datetime as dt


def readTarFile(f_tar, file, toString = True):
    """
    Given key = [NoteID, NoteCSNID, LineID], output string
    Assume key is unique
    f_tar, dictNotes are output from the tarFile2Dict function
    """
    f = f_tar.extractfile(file)
    if f:
        content = f.read()
        if toString:
            content = content.decode("utf-8") 
    else:
        content = ''
    return content

def load_data(path, isCompress = True):
    if isCompress:
        X = pd.read_csv(path,chunksize = 10000, compression = "gzip", error_bad_lines=False)
    else:
        X = pd.read_csv(path,chunksize = 10000, error_bad_lines=False)
    df = []
    for x in X:
        df.append(x)
    return pd.concat(df, axis= 0)

def splitSentence(content):
    """
    Given block of text, split into sentence
    Output: list of sentences
    """
    # Multiple space to single space, remove separators like - and _
    if pd.notnull(content):
        content = re.sub('\s*\t\t\t', ' ', content)
        content = re.sub('--+|==+|__+', ' ', content)
        content = re.sub('\.\s+', '. ',content)
        content = re.sub(':\s+', ': ',content)
        content = re.sub('\s+\[\*', ' [*', content)
        content = re.sub(' \s+', '. ',content)
        lsS = content.split('. ')
    else:
        lsS = []
    return lsS


def update(s):
    """
    #- replace number to <num> (keep number right after text, as typically are certain clinical names)
    #- replace time to <time>
    - replace digits to <N> token
    - add space before/after non-character
    """
    s = re.sub('\d', 'N', s)
    #s = re.sub('\d+:\d+(:\d+)?\s*((a|A)|(p|P))(m|M)(\s*est|EST)?', ' <time> ', s)
    #s = re.sub('( |^|\(|:|\+|-|\?|\.|/)\d+((,\d+)*|(\.\d+)?|(/\d+)?)', ' <num> ', s) # cases like: 12,23,345; 12.12; .23, 12/12;
    s = re.sub(r'([a-zA-Z->])([<\),!:;\+\?\"])', r'\1 \2 ', s)
    s = re.sub(r'([\(,!:;\+>\?\"])([a-zA-Z<-])', r' \1 \2', s)
    s = re.sub('\s+', ' ', s)
    return s


def replcDeid(s):
    """
    replace de-identified elements in the sentence (date, name, address, hospital, phone)
    """
    s = re.sub('\[\*\*\d{4}-\d{2}-\d{2}\*\*\]', '<date>', s)
    s = re.sub('\[\*\*.*?Name.*?\*\*\]', '<name>', s)
    s = re.sub('\[\*\*.*?(phone|number).*?\*\*\]', '<phone>', s)
    s = re.sub('\[\*\*.*?(Hospital|Location|State|Address|Country|Wardname|PO|Company).*?\*\*\]', '<loc>', s)
    s = re.sub('\[\*\*.*?\*\*\]', '<deidother>', s)
    return s

def tag_negation( doc ):

    from nltk.sentiment.util import mark_negation
    return ' '.join( mark_negation(doc.split()) )

def cleanString(s, lower = True):
    s = replcDeid(s)
    s = update(s)
    if lower:
        s = s.lower()
    return s


def replaceContractions(s):
    contractions = ["don't","wouldn't","couldn't","shouldn't", "weren't", "hadn't" , "wasn't", "didn't" , "doesn't","haven't" , "isn't","hasn't"]
    for c in contractions:
        s = s.replace( c, c[:-3] +' not')
    return s

def preprocess_string(s):
    s = cleanString(s, True)
    s = replaceContractions(s)
    return s


def cleanNotes(content):
    """
    Process a chunk of text 
    """
    lsOut = []
    content = str(content)
    if len(content) > 0:
        lsS = splitSentence(content)
        for s in lsS:
            if len(s) > 0:
                s = cleanString(s, lower = True)
                s = replaceContractions(s)
                lsOut.append(s)
        out = ' '.join(lsOut)
    else:
        out = ''
    return out


def load_star_space(fn, torch = True):
    #ss = pd.read_csv(fn,sep='\t')
    ss =  pd.read_csv(fn,sep='\t', quoting=3, header= None)

    keys= list(ss.iloc[:,0])
    keys= dict([ (k,i) for i,k in enumerate(keys)])
    params = np.array(ss.iloc[:,1:])
    if torch:
        params = torch.from_numpy(params)
    return keys, params

def stopwords():
    
    return pickle.load(open('./data/stop_words.p','rb'))

def stopwords2(fileName):
    lsW = []
    with open(fileName) as f:
        for line in f:
            data = line.split()
            lsW.extend(data)   
    return lsW

#====== Other transformation functions ======
def difDays(d1, d2, dateFormat1, dateFormat2):
    if (pd.notnull(d1) & pd.notnull(d2)):
        d1 = dt.strptime(d1, dateFormat1).date()
        d2 = dt.strptime(d2, dateFormat2).date()
        return (d1 - d2).days
    return None

def normalize(df, isTrain, meanValue = None, stdValue = None):
    if isTrain:
        meanValue = df.mean(axis = 0)
        stdValue = df.std(axis = 0)
        stdValue[pd.isnull(stdValue)] = 1.0
    df = (df - meanValue) / stdValue
    return df, meanValue, stdValue
#===== Helper functions to extract labs ======
def getNumPct(content):
    # Compute percentage of numerical values in a string
    out, nNum = 0, 0
    if pd.notnull(content):
        content2 = str(content)
        content2 = re.sub('[^0-9A-Za-z\s]', '', content2)
        nNum = len(re.sub('[^0-9\s]','', content2).split())
        nText = len(re.sub('[^A-Za-z\s]','', content2).split())
        if (nNum + nText) > 0:
            out = nNum / (nNum + nText)
    return out, nNum


def getNumBlock(content, dictKeyWords, numPct, numCount, split = '     '):

    """
    Split paragraph by multiple space into blocks, keep blocks with keywords and number
    :param content: input string
    :param lsKeyWords:
    :return: lsOut
    """
    lsOut = []
    if pd.notnull(content):
        content = str(content)
        content = re.sub('\t', split, content)
        content = re.sub('--+|==+|__+', split, content)
        content = re.sub(r'([^0-9])(\s+)([><=0-9])', r'\1 \3', content) # Remove multiple space before numbers
        content2 = content.split(split)

        for s in content2:
            s = str(s).strip()
            s = re.sub(r'\s*\[\*\*.*?\*\*\]\s*', ' ', s)
            s = re.sub('\d+:\d+(:\d+)?\s*((a|A)|(p|P))(m|M)(\s*est|EST)?', '', s)
            s2 = s.lower()
            flgNum = 0
            n_pos_start = len(s)

            for w in dictKeyWords: # Look for the first starting position of lab values
                n_pos = re.search(w, s2)
                if n_pos is not None:
                    if n_pos.start() < n_pos_start:
                        pctNum, nNum = getNumPct(s2[n_pos.end():])
                        n_pos_start = n_pos.start()
                        type = dictKeyWords[w]
                        flgNum = 1

            if flgNum == 1:
                if (pctNum > numPct[0]) & (nNum > numCount[0]):
                    lsOut.append((type, s[n_pos_start:]))
                    if n_pos_start != 0:
                        pctNum, nNum = getNumPct(s[0:n_pos_start])
                        if (pctNum > numPct[1]) & (nNum > numCount[1]):
                            lsOut.append(('numOther', s[0:n_pos_start]))
                        else:
                            lsOut.append(('num_prev', s[0:n_pos_start]))
                else:
                    lsOut.append(('text', s))

            elif flgNum == 0:
                pctNum, nNum = getNumPct(s)
                if (pctNum >= numPct[1]) & (nNum >= numCount[1]) & (pctNum < 1):
                    lsOut.append(('numOther', s))
                else:
                    lsOut.append(('text', s))

    #pdb.set_trace()
    return lsOut


def build_vocab(text, negation = False, max_df = .7, max_features = 20000, vecPath = '/ifs/data/razavianlab/ehr_ssp_embedding/word2CurDiag_ge5_5.tsv',
                stopWordPath = '/ifs/data/razavianlab/stop_words.txt', torch = True):
    '''
    Fit vocabulary and create PubMed w2v matrix
    :param text: list of documents for creating vocabulary
    :return: embedding matrix and vectorizer
    '''
    #import torchwordemb

    #load w2v
    #w2v_vocab, vec = torchwordemb.load_word2vec_bin("./data/PubMed-and-PMC-w2v.bin")
    w2v_vocab , vec = load_star_space(vecPath, torch)

    #vect = CountVectorizer(stop_words = 'english',max_df = max_df,  max_features = max_features)
    stopWords = stopwords2(stopWordPath)
    vect = CountVectorizer(stop_words = stopWords, max_df = max_df, max_features = max_features)

    vect.fit(text)

    no_embedding = [ k for k in vect.vocabulary_.keys() if k not in w2v_vocab ]
    print("No Embeddings for: ")
    print(len(no_embedding))

    vocab = dict([ (k, w2v_vocab[k])  for k in vect.vocabulary_.keys() if k in w2v_vocab])
    new_vocab = dict([ (k,i+1) for i,k in enumerate(vocab.keys()) ]) # Set 0 to be the padding index

    if torch:
        embedding = torch.zeros(len(new_vocab)+1, vec.shape[1])
    else:
        embedding = np.zeros(shape = (len(new_vocab) + 1, vec.shape[1]))

    for k,i in new_vocab.items():
        embedding[i] = vec[vocab[k]]

    if negation:
        n_emb = embedding.size()[0] - 1
        neg_emb = -1 * embedding
        if torch:
            embedding = torch.cat( [embedding, neg_emb],0)
        else:
            embedding = np.concatenate([embedding, neg_emb], 0)
        
        for k,v in new_vocab.items():
            new_vocab[k +'_NEG'] = v +n_emb 

    vect.vocabulary_ = new_vocab

    return embedding, vect


def pad_doc(seq, max_len, n):
    padded_seq = torch.zeros(n, max_len)

    start = 0 if len(seq) >= n else n - len(seq)

    for i, s in enumerate(seq):

        if len(s) > max_len:
            padded_seq[start + i] = torch.Tensor(s[:max_len]).long()
        else:
            if len(s) == 0:
                continue

            padded_seq[start + i, -len(s):] = torch.Tensor(s).long()

    return padded_seq


def prepare( text, vectorizer , max_len ,unique = False ):

    vocab = vectorizer.vocabulary_
    tokenizer = vectorizer.build_tokenizer()

    if unique:
        seq = [ list(set( [  vocab[y] for y in tokenizer(x) if y in vocab])) for x in text ]
    else:
        seq = [ [  vocab[y] for y in tokenizer(x) if y in vocab] for x in text]
    

    lengths = np.array([ len(s) for s in seq])

    print("Average Sequnce Length: " , lengths.mean())
    print("90% Length: " , np.percentile(lengths, 90))

    padded_seq = pad_doc(seq, max_len, len(seq))

    return padded_seq


def sentence_prepare( text, vectorizer , sent_len , doc_len ,unique = False):

    #from nltk.tokenize import sent_tokenize
    from segtok.segmenter import split_multi

    vocab = vectorizer.vocabulary_
    tokenizer = vectorizer.build_tokenizer()

    #text = [sent_tokenize(doc) for doc in text  ]
    text = [list(split_multi(doc)) for doc in text]

    seq = []
    sent_l = []
    doc_l = []
    for doc in text:
        doc_tok = []
        for sent in doc:
            sent_toks = [vocab[y] for y in tokenizer(sent) if y in vocab]             
            doc_tok.append(sent_toks)
            sent_l.append(len(sent_toks))

        seq.append(doc_tok)
        doc_l.append(len(doc_tok))

    sent_l = np.array(sent_l)
    doc_l = np.array(doc_l)

    print("Average Sent Length: " , sent_l.mean())
    print("90% Length: " , np.percentile(sent_l, 90))

    print("Average Doc Length: " , doc_l.mean())
    print("90% Length: " , np.percentile(doc_l, 90))

    #sent_len = np.percentile(sent_l, 90)
    #doc_len = np.percentile(doc_l, 90)

    padded_docs = torch.zeros(len(seq) , doc_len , sent_len)

    for i, _doc in enumerate(seq):

        if len(_doc) > doc_len:
            _doc = _doc[:doc_len]
            padded_seq = pad_doc(_doc, sent_len, len(_doc))
        else:
            if len(_doc) ==0:
                continue

            padded_seq = pad_doc(_doc, sent_len, doc_len)

        padded_docs[i] = padded_seq

    return padded_docs


#w2v_vocab , vec = load_star_space('/ifs/data/razavianlab/ehr_ssp_embedding/word2CurDiag_ge3.tsv')

#================= Math functions ====================
def softmax(x):

    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.expand_dims(np.max(x,axis=1),1))
    return e_x / np.expand_dims(e_x.sum(axis=1),1)
