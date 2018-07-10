"""
# Generate word count and subset embedding files with word > N occurrences
"""
import sys
sys.path.append('/ifs/home/liuj24/EHR/code')


dataDir = '/ifs/data/razavianlab/ssp_input/'
embFile = '/ifs/data/razavianlab/ehr_ssp_embedding/word2CurDiag_ge5_dim300.tsv'
outputFile = '/ifs/data/razavianlab/ehr_ssp_embedding/word2CurDiag_ge5_dim300_ge1k.tsv'


import util_noTorch as util
from collections import Counter
import pickle
#import pandas as pd

word_cnt = Counter([])
i = 0
with open(dataDir + 'input_model0_curICD_train.txt') as f:
    for line in f:
        text = line.split('\t<diag>')[0].strip()
        word_cnt += Counter(text.split(' '))
        i += 1
        if (i % 2000) == 0:
            print(i)
pickle.dump(word_cnt, open('/ifs/data/razavianlab/ssp_input/word_cnt.p', 'wb'))

word_list = [k for k, c in word_cnt.items() if c >= 1000]
print('Total words: ', len(word_list))

w2v_vocab, vec = util.load_star_space(embFile, torch = False)

with open(outputFile, 'a') as f:
    for w in word_list:
        if w in w2v_vocab:
            f.write(w +'\t' + '\t'.join([str(x) for x in list(vec[w2v_vocab[w]])]) + '\n')

f.close()



