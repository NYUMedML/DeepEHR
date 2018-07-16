# Valx: A system for extracting and structuring numeric lab test comparison statements from text
# Created by Tony HAO, th2510@columbia.edu
# Please kindly cite the paper: Tianyong Hao, Hongfang Liu, Chunhua Weng. Valx: A system for extracting and structuring numeric lab test comparison statements from text. Methods of Information in Medicine. Vol. 55: Issue 3, pp. 266-275, 2016

import W_utility.file as ufile
from W_utility.log import ext_print
import os,sys,re
import Valx_core


def extract_variables (fdin, ffea, ffea2, var):
    # read input data
    if fdin is None or fdin =="": return False
    trials = ufile.read_csv (fdin)
    if trials is None or len(trials) <= 0:
        print(ext_print)
        print('input data error, please check either no such file or no data --- interrupting')
        return False
    print(ext_print)
    print('found a total of %d data items' % len(trials))
    
    # read feature list - domain knowledge
    if ffea is None or ffea =="": return False
    fea_dict_dk = ufile.read_csv_as_dict_with_multiple_items (ffea)
    if fea_dict_dk is None or len(fea_dict_dk) <= 0:
        print(ext_print)
        print('no feature data available --- interrupting')
        return False

    # get feature info
    features, feature_dict_dk = {}, {}
    if var == "All":
        features = fea_dict_dk
        del features["Variable name"]
    elif var in fea_dict_dk:
        features = {var:fea_dict_dk[var]}
    for key, value in fea_dict_dk.iteritems():
        names = value[0].lower().split('|')
        for name in names:
            if name.strip() != '': feature_dict_dk[name.strip()] =key

    # read feature list - UMLS (can be replaced by full UMLS)
    if ffea2 is None or ffea2 =="": return False
    fea_dict_umls = ufile.read_csv_as_dict (ffea2)
    if fea_dict_umls is None or len(fea_dict_umls) <= 0:
        print(ext_print)
        print('no feature data available --- interrupting')
        return False

    #load numeric feature list
    Valx_core.init_features()

    output = []
    for i in range(len(trials)):
        if i%1000 == 0:
            print ('processing %d' % i)
        # pre-processing eligibility criteria text
        text = Valx_core.preprocessing(trials[i][1]) # trials[i][1] is the eligibility criteria text
        (sections_num, candidates_num) = Valx_core.extract_candidates_numeric(text) # extract candidates containing numeric features
        for j in range(len(candidates_num)): # for each candidate
            exp_text = Valx_core.formalize_expressions(candidates_num[j]) # identify and formalize values
            (exp_text, key_ngrams) = Valx_core.identify_variable(exp_text, feature_dict_dk, fea_dict_umls) # identify variable mentions and map them to names
            (variables, vars_values) = Valx_core.associate_variable_values(exp_text)
            all_exps = []
            for k in range(len(variables)):
                curr_var = variables[k]
                curr_exps = vars_values[k]
                if curr_var in features:
                    fea_list = features[curr_var]
                    curr_exps = Valx_core.context_validation(curr_exps, fea_list[1], fea_list[2])                           
                    curr_exps = Valx_core.normalization(fea_list[3], curr_exps) # unit conversion and value normalization
                    curr_exps = Valx_core.hr_validation (curr_exps, float(fea_list[4]), float(fea_list[5])) # heuristic rule-based validation
                if len(curr_exps) > 0:
                    if var == "All" or var.lower() == curr_var.lower() or var.lower() in curr_var.lower(): all_exps += curr_exps                     
                 
            if len(all_exps) > 0: output.append((trials[i][0], sections_num[j], candidates_num[j], exp_text, str(all_exps).replace("u'", "'"))) # output result

    # output result
    fout = os.path.splitext(fdin)[0] + "_exp_%s.csv" % var
    ufile.write_csv (fout, output)
    print(ext_print)
    print('saved processed results into: %s' % fout)
    return True


# processing the command line options
import argparse
def _process_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', default=r"D:\_My_programs\_CUMC\Extract_Variables\_GitHub\data\example data diabetes_Type 1.csv", help='input: a specific disease')
    parser.add_argument('-f1', default=r"D:\_My_programs\_CUMC\Extract_Variables\_GitHub\data\variable_features_dk.csv", help='input: a feature list')
    parser.add_argument('-f2', default=r"D:\_My_programs\_CUMC\Extract_Variables\_GitHub\data\variable_features_umls.csv", help='input: a feature list')
    parser.add_argument('-v', default="HBA1C", help='Variable name: All, HBA1C, BMI, Glucose, Creatinine, BP-Systolic, BP-Diastolic') # 'All' means to detect all variables
    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__' :
    print('')
    args = _process_args()
    extract_variables (args.i, args.f1, args.f2, args.v)
    print('')
