import numpy as np
import pickle
import sys
import time
import pandas as pd
from multiprocessing import Pool
from itertools import product
import time
from sklearn.metrics import auc
from sklearn import metrics
import ast
from sklearn.feature_extraction.text import CountVectorizer
from negex import *

def parse_sent(text,nlp, tok, neg_words):

    #tokens = word_tokenize(text)
    tokens = tok(text)

    if any([ w in neg_words for w in tokens ]):
        tags = ast.literal_eval(nlp.parse(text))['sentences'][0]['dependencies']

        for tag in tags:
            if tag[0] =='neg':
                text = text.replace( tag[1], tag[1] + '_NEG' )

        return text
    else:
        return text

def split_sent(text):

    return re.split(r'[:?.]+',str(text))

def negate(text, irules , conditions):

    #cond = [c for c in conditions if c in text]
    
    sentences = list(split_sent(text))

    tagged = []
    filter_conds = 0
    tag = 0

    for s in sentences:
        cond = [c for c in conditions if c in s]

        t = negTagger(sentence = s, phrases = cond, rules = irules, negP=False).getNegTaggedSentence()
        
        tagged.append(t)

    return ' '.join(tagged)    

    
if __name__ =='__main__':

    rules=  pd.read_csv('./negex_triggers.txt' ,sep='\t',header=None)
    rules = list(rules[0] + '\t\t' + rules[2])
    irules = sortRules(rules)
    
    conditions = ['cough','headache']

    sentence = 'the patient is negative for cough, and headache.'

    print(negate(sentence , irules , conditions))
