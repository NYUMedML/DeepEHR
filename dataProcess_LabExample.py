#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  21 2018

@author: jshliu

Extract lab values from notes

Libs to install:
    import nltk
"""

import sys
srcDir = '.'
sys.path.append(srcDir + 'Valx_modified')

import Valx_core as v
import W_utility.file as ufile
import pandas as pd
import util_noTorch as util
import re
import numpy as np
import time

import argparse

def getNum(s, feature_dict_dk, feature_dict_umls):
    """
    Clean text input and run valx
    """
    all_exps = []
    s = re.sub(r'(\d+)(,)(\d+)', r'\1\2', s)
    s = re.sub(r'(\()(.*(m|g|n)(/|\\|\^)*\d.*)(\))', r'\2', s) # keep units in parenthesis
    s = re.sub(r'\([^\)]*?\)', '', s) # remove parenthesis
    s = re.sub(' ', '  ', s) # for next step
    s = re.sub(r'(^|\s)(((0|1)\d{3})|(2(0|1|2|3|4)\d{2}))(\s|$)', '', s) # remove time stamp
    s = re.sub('(^|\s)\d+((E|e)ncounter(s?)|hrs|HRS|(R|r)eading(s?))\s', ' ', s) # remove encounter counts
    s = re.sub(r'(\d+)(\s*)\'(\s*)((\d+)(\s*)\")?', r'\1.\5 feet', s)
    s = re.sub(r'\?C', 'Celsius', s)
    s = re.sub(r'\?F', 'Fahrenheit', s)
    s = re.sub('[:,]', ' ', s)
    s = re.sub(r'([^0-9])(\s*)(\.)(\s*)([^0-9])', r'\1 \5', s) # Remove . not between digits
    s = re.sub(r'\s+', ' ', s)
    t = v.preprocessing(s)
    sections_num, t2 = v.extract_candidates_numeric(t)
    if len(t2) > 0:
        for t3 in t2:
            flg_longStr = False
            t3 = v.formalize_expressions(t3)
            exp_text, key_ngrams = v.identify_variable(t3, feature_dict_dk, feature_dict_umls)
            for ngram in key_ngrams:  # Remove cases like 'BP Temp Temp src Pulse Resp SpO2 Height Weight 2.1 ?C (98.7 ?F) Oral 97 17 97 %'
                ngram = re.sub('[^a-zA-Z0-9\\/]', ' ', ngram)
                ngram = re.sub('\s+', ' ', ngram)
                lsNgram = ngram.split()
                if len(lsNgram) > 2:
                    cntMatch = 0
                    for gram in lsNgram:
                        if gram in feature_dict_dk:
                            cntMatch += 1
                    if cntMatch >= 2:
                        flg_longStr = True
                        break

            if not flg_longStr:
                variables, vars_values = v.associate_variable_values(exp_text)

                for k in range(len(variables)):
                    curr_var = variables[k]
                    curr_exps = vars_values[k]
                    if curr_var in features:
                        # pdb.set_trace()
                        fea_list = features[curr_var]
                        curr_exps = v.context_validation(curr_exps, fea_list[1], fea_list[2])
                        curr_exps = v.normalization(fea_list[3], curr_exps)  # unit conversion and value normalization
                        if (fea_list[4] != '') & (fea_list[5] != ''):
                            curr_exps = v.hr_validation(curr_exps, float(fea_list[4]),
                                                        float(fea_list[5]))  # heuristic rule-based validation
                    if len(curr_exps) > 0:
                        all_exps += curr_exps

    return all_exps

def exp2DF(exp, ID):
    """
    Convert expression to data frame
    :param exp:
    :return:
    """
    if len(exp) == 0:
        dfOut = pd.DataFrame({'ItemName': '', 'Relationship': '', 'Value': None, 'Unit': '', 'PAT_ENC_CSN_ID': ID, 'Sequence': 0}, index=[0])
    else:
        dfOut = pd.DataFrame.from_records(exp, columns = ['ItemName', 'Relationship', 'Value', 'Unit'])
        dfOut['PAT_ENC_CSN_ID'] = [ID] * len(dfOut)
        dfOut['Sequence'] = np.arange(len(dfOut))
    return dfOut




def main(dfNote):
    #===== Split note types =======

    dictKeyWords = {'lab results': 'labs', 'labs': 'labs', 'blood': 'labs',
                    'vital signs': 'vitals', 'vitals': 'vitals', 'wt reading': 'vitals', 'temp': 'vitals',
                    'value ref range': 'value with ref', 'value date/time': 'value with date/time', 'value date': 'value with date'}
    # Lab format remove reference; vital format remove date time

    dfNote3 = None
    for row in dfNote.iterrows():
        row = row[1]
        out = util.getNumBlock(row['NOTE_TEXT'], dictKeyWords, [0.1, 0.5], [1, 3])
        dfOut = pd.DataFrame.from_records(out, columns = ['Type', 'Text'])
        #dfOut['NOTE_ID'] = row['NOTE_ID']
        dfOut['PAT_ENC_CSN_ID'] = row['PAT_ENC_CSN_ID']
        if dfNote3 is None:
            dfNote3 = dfOut
        else:
            dfNote3 = pd.concat([dfNote3, dfOut])
        #pdb.set_trace()

    dfNote3['Type'].value_counts()
    dfNote4 = dfNote3[~dfNote3['Type'].isin(['text', 'num_prev'])]

    def f_temp(group):
        i = group.index[0]
        return exp2DF(getNum(group.loc[i,'Text'], feature_dict_dk, feature_dict_umls), group.loc[i,'PAT_ENC_CSN_ID'])

    dfNote4 = dfNote4.reset_index(drop = True)
    dfLab = dfNote4.groupby(dfNote4.index, group_keys=False).apply(f_temp).reset_index(drop = True)
    dfLab = dfLab[dfLab['ItemName'] != '']

    return dfLab, dfNote4



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--inputFile")
    parser.add_argument("--outputFile")
    parser.add_argument("--num_splits", type=int, default=4)
    parser.add_argument("--split_index", type=int, default=1)
    args = parser.parse_args()

    np.random.seed(42)

    fea_dict_umls = ufile.read_csv_as_dict_with_multiple_items(srcDir + 'Valx/data/variable_features_umls.csv')
    feature_dict_umls = {}
    for key, value in fea_dict_umls.items():
        feature_dict_umls[key] = value[0]


    fea_dict_dk = ufile.read_csv_as_dict_with_multiple_items(srcDir + 'Valx/data/variable_features_dkV2.csv')

    features, feature_dict_dk = {}, {}
    features = fea_dict_dk
    del features["Variable name"]
    for key, value in fea_dict_dk.items():
        names = value[0].lower().split('|')
        for name in names:
            if name.strip() != '': feature_dict_dk[name.strip()] =key

    v.init_features()


    #=============== Sample notes =================

    dfNote = util.load_data(args.inputFile, False)
    dfNote = dfNote[dfNote['NOTE_TYPE'] != 'Patient Instructions']
    dfNote = dfNote.drop_duplicates(['PAT_ENC_CSN_ID', 'NOTE_TEXT'])

    dfNote = np.array_split(dfNote, args.num_splits)[args.split_index - 1]

    print('starting')
    start = time.time()

    print('Number of encounters: ', len(dfNote['PAT_ENC_CSN_ID'].unique()))

    dfLab, dfNote4 = main(dfNote)
    dfLab.to_csv(args.outputFile + '_' + str(args.split_index) +'.csv', index = False)
    dfNote4.to_csv(args.outputFile + '_noteText' + str(args.split_index) +'.csv', index = False)
    print('Number of encounters with lab: ', len(dfLab['PAT_ENC_CSN_ID'].unique()))

    agg_f = {'ItemName': lambda x: len(x.unique()), 'Sequence': 'max'}
    cntLab = dfLab.groupby('PAT_ENC_CSN_ID').agg(agg_f)
    print(cntLab[['ItemName', 'Sequence']].describe())

    print('Process Time: ')
    print(time.time() - start)

    