# -*- encoding: utf-8 -*-
# Valx: A system for extracting and structuring numeric lab test comparison statements from text
# Created by Tony HAO, th2510@columbia.edu
# Please kindly cite the paper: Tianyong Hao, Hongfang Liu, Chunhua Weng. Valx: A system for extracting and structuring numeric lab test comparison statements from text. Methods of Information in Medicine. Vol. 55: Issue 3, pp. 266-275, 2016

"""
To-do:
1. relation like increased, change, etc
2. fractions: 170 lb 3.2 oz
3. Deal with vitals separately,
- Lab Results
- Vitals:  Vital Signs:
- Component Value Date/Time
- Result Value Ref Range

- remove dates, remove time


- ?C / ?F: Celsuis, F
- section space: 19:24 / 7
- item space: 6?


"""


import re, math, csv, os
import W_utility.file as ufile
from NLP import sentence
from NLP import sentence_keywords
import pdb

srcDir = os.path.dirname(__file__)

#--------------------------Define representative logics and their candidate representations 

greater, greater_equal, greater_equal2, lower, lower_equal, lower_equal2, equal, between, selects, connect, features, temporal, temporal_con, error1, error2, symbols, numbers, unit_special, unit_ori, unit_ori_s, unit_exp, negation = "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""

def init_features ():
    feature_set = ufile.read_csv_as_dict(srcDir + '/data/numeric_features.csv', 0, 1, True)
    global greater, greater_equal, greater_equal2, lower, lower_equal, lower_equal2, equal, between, selects, connect, features, temporal, temporal_con, error1, error2, symbols, numbers, unit_special, unit_ori, unit_ori_s, unit_exp, negation
    greater, greater_equal, greater_equal2, lower, lower_equal, lower_equal2, equal, between, selects, connect, features, temporal, temporal_con, error1, error2, symbols, numbers, unit_special, unit_ori, unit_ori_s, unit_exp, negation = \
    feature_set["greater"], feature_set["greater_equal"], feature_set["greater_equal2"], feature_set["lower"], feature_set["lower_equal"], feature_set["lower_equal2"], feature_set["equal"], feature_set["between"], feature_set["selects"], feature_set["connect"], feature_set["features"], feature_set["temporal"], feature_set["temporal_con"], feature_set["error1"], feature_set["error2"], feature_set["symbols"], feature_set["numbers"], feature_set["unit_special"], feature_set["unit_ori"], feature_set["unit_ori_s"], feature_set["unit_exp"], feature_set["negation"]
    temporal = temporal + '|' + temporal.replace('|', 's|') + 's'
    unit = (unit_ori + "|" + unit_ori_s.replace("|", "s|") + "s|" + unit_ori_s + "|" + temporal)
    #return ""

def preprocessing (text):
    # handle special characters
    #text = text.decode('ascii', 'ignore')

    text = text.strip().replace('\n\n', '#')
    text = text.replace ('\n', '')
    text = text.replace(u'＝','=').replace(u'＞', '>').replace(u'＜','<').replace(u'≤','<=').replace (u'≥','>=').replace(u'≦','<=').replace(u'≧','>=').replace(u'mm³','mm^3').replace(u'µl','ul').replace(u'µL','ul').replace(u'·','').replace(u'‐','-').replace(u'—','-')

    text = text.replace('((', '(').replace('))', ')')
    text = re.sub('(\d+)( |)(~|/|&|\|)( |)(\d+)',r'\1 - \5',text) # e.g., '10~20' to '10 ~ 20'
    text = re.sub(r"(\d+),(\d{3})", r'\1\2', text) # 10,123 to 10123
    text = re.sub(r"(\d+),(\d{1,2})", r'\1.\2', text) # 10,1 to 10.1
    text = re.sub(r"between (\d+), (\d{1,2}) (and|or) ", r'between \1.\2 \3 ', text) # 'between 7, 5 and ' to 'between 7.5 and '
    text = re.sub(r"(\d+) (y(\.|/)?o)", r'age is \1 years', text) # Process age
    while '  ' in text:
        text = text.replace('  ',' ')
    # avoid connected values separated by splitting, e.g., ", but below 10%"
    text = re.sub(", ("+connect+") ", r' \1 ', text) # 

    return text.lower()


def split_text_inclusion_exclusion(text):
    in_fea = 'inclusion criteria:|key inclusion criteria|inclusion criteria [^:#;\.]+:|inclusion:|(?<!(\w| ))inclusion criteria\W\W|inclusion for'
    ex_fea = 'exclusion criteria:|key exclusion criteria|exclusion criteria [^:#;\.]+:|exclusion:|(?<!(\w| ))exclusion criteria\W\W|exclusion for'
   
    in_text, ex_text = '', ''
    in_bool = True

    text = text.lower()
    while text != '':
        if in_bool:
            n_pos = re.search('('+ex_fea+')',text)
            if n_pos is not None:
                in_text += text[0:n_pos.start()]
                text = text[n_pos.start():]
            else:
                in_text += text[0:]
                text = ''
        else:
            n_pos = re.search('('+in_fea+')',text)
            if n_pos is not None:
                ex_text += text[0:n_pos.start()]
                text = text[n_pos.start():]
            else:
                ex_text += text[0:]
                text = ''
        in_bool = False if in_bool else True
    
    sections_text =[]
    if in_text !='': sections_text.append(["Inclusion", in_text])
    if ex_text !='': sections_text.append(["Exclusion", ex_text])    
    return sections_text


#====find expression candidates according to pre-defined feature list
def extract_candidates_numeric (text):
    # process text
    sections_text = split_text_inclusion_exclusion(text)
    
    sections_num = []
    candidates_num = []
      
    for section_text in sections_text:        
        sentences = sentence.sentence_splitting_symbols(section_text[1], "[#!?.;]\s", 1)
        for sent in sentences:
            sent = sent.strip().strip('- ')
            if sent == '':
                continue
                
            digit = re.search("(?<!(\w))\d+", sent)
            if digit:
                sections_num.append(section_text[0])
                candidates_num.append(sent)

    return (sections_num, candidates_num)


def extract_candidates_name (sections_num, candidates_num, name_list):
    sections = []
    candidates = []
    names = name_list.split('|')
    for i in range(len(candidates_num)):            
        for name in names:
           if name in candidates_num[i]:
                sections.append(sections_num[i])
                candidates.append(candidates_num[i])
                break

    return (sections, candidates)


#====identify expressions and formalize them into labels "<VML(tag) L(logic, e.g., greater_equal)=X U(unit)=X>value</VML>"
def formalize_expressions (candidate):
    text = candidate
    #csvfile = open(srcDir + '/data/rules.csv', 'rb')
    #srcDir = '/Users/jshliu/Google Drive/NYUProjects/CapStone/code/Valx'
    csvfile = open(srcDir + '/data/rules.csv', 'r')
    reader = csv.reader(csvfile)
    now_pattern = "preprocessing"

    for i,pattern in enumerate(reader):
        source_pattern = pattern[0]
        target_pattern = pattern[1]
        pattern_function = pattern[2]
        #pdb.set_trace()
        if(pattern_function == "process_numerical_values" and pattern_function != now_pattern):
            matchs = re.findall('<Unit>([^<>]+)</Unit>', text)
            for match in matchs: text = text.replace(match, match.replace(' / ', '/').replace(' - ','-'))

        if(pattern_function == "process_special_logics" and pattern_function != now_pattern):
            # process 'select' expression, use the first one
            global selects
            aselect = selects.split('|')
            for selec in aselect:
                selec = selec.replace('X', '<VML Unit([^<>]+)>([^<>]+)</VML>')
                #pdb.set_trace()
                text = re.sub(selec, r'<VML Unit\1>\2</VML>', text) #

            #  process 'between' expressions
            global between
            betweens = between.split('|')
            for betw in betweens:
                betw = betw.replace('X', '<VML Unit([^<>]+)>([^<>]+)</VML>')
                text = re.sub(betw, r'<VML Logic=greater_equal Unit\1>\2</VML> - <VML Logic=lower_equal Unit\3>\4</VML>', text) #
        text = re.sub(source_pattern, target_pattern, text)
        now_pattern = pattern_function

    csvfile.close()
    return text


add_mentions_front = 'total|absolute|mean|average|abnormal|gross'
add_mentions_back = 'test results|test result|test scores|test score|tests|test|scores|score|results|result|values|value|levels|level|ratios|ratio|counts|count|volume'
def identify_variable (exp_text, feature_dict_dk, feature_dict_umls):
    #exp_text = 'pulse: <VML Logic=equal Unit=>88</VML> '
    # find candidate string
    if exp_text.find('<VML') == -1:
        return (exp_text, [])
    #exp_text = re.sub(r'(\\|/)hr', 'hour', exp_text) # Typically means heart rate in EHR
    can_texts = re.findall('(\A|VML>)(.+?)(<VML|\Z)',exp_text) 

    # generate n-grams
    first_ngram, key_ngrams = '', [] # first ngram; key ngrams are the ngrams except the ngrams match with domain knowledge and umls
    match = False
    for cantext in can_texts:
        if '<VL Label' in cantext[1]: 
            ngrams = re.findall('<VL Label=([^<>]+) Source', cantext[1])
            for ngram in ngrams:# judge if they are potential variables
                if ngram in feature_dict_dk:
                    exp_text = re.sub(r'<VL Label='+ngram+' Source=', r"<VL Label=%s Source=" % feature_dict_dk[ngram], exp_text)
                elif ngram in feature_dict_umls:
                    exp_text = re.sub(r'<VL Label='+ngram+' Source=', r"<VL Label=%s Source=" % feature_dict_umls[ngram], exp_text)
            match = True
        else:
            if len(cantext[1].split()) == 1: # Add to avoid removal single character by nltk
                ngrams = cantext[1].split()
            else:
                #ngrams = sentence_keywords.keywords_ngrams_reverse(cantext[1].replace(' - ', '-').strip())
                ngrams = sentence_keywords.keywords_ngrams_reverse(cantext[1].strip())
            if len(ngrams) > 0:
                ngrams = [x.replace(' - ', '-') for x in ngrams]
                longest_str = max(ngrams, key=len)
                key_ngrams.append(longest_str)
                if first_ngram == '': first_ngram = longest_str
            for ngram in ngrams:# judge if they are potential variables
                if ngram in feature_dict_dk:
                    if ngram in key_ngrams: key_ngrams.remove(ngram)
                    exp_text = re.sub(r'(?<!(\w|<|>))'+ngram+'(?!(\w|<|>))', r"<VL Label=%s Source=DK>%s</VL>" % (feature_dict_dk[ngram], ngram), exp_text, 1)
                    match = True
                    break
                elif ngram in feature_dict_umls:
                    if ngram in key_ngrams: key_ngrams.remove(ngram)
                    exp_text = re.sub(r'(?<!(\w|<|>))'+ngram+'(?!(\w|<|>))', r"<VL Label=%s Source=UMLS>%s</VL>" % (feature_dict_umls[ngram], ngram), exp_text, 1)
                    match = True
                    break

    exp_text = re.sub(r'<VL ([^>]+)<VL Label=[^<>]+>([^<>]+)</VL>',r'<VL \1\2', exp_text)
    exp_text = re.sub(r'(?<!(\w|<|>|=))('+add_mentions_front+') <VL Label=([^<>]+) Source=([^<>]+)>([^<>]+)</VL>', r"<VL Label=\2 \3 Source=\4>\2 \5</VL>", exp_text)
    exp_text = re.sub(r'</VL>'+' ('+add_mentions_back+r')(?!(\w|<|>))', r" \1</VL>", exp_text)

    # Remove guesses for now
    #if len(can_texts)>0 and not match and first_ngram.strip() != '': #guess variable
    #    exp_text = exp_text.replace(first_ngram, "<VL Label=%s Source=ngram>%s</VL>" % (first_ngram, first_ngram), 1)
#     marks =re.findall(r'<VL Label=([^<>]+)>[^<>]+</VL>', exp_text)

    return (exp_text, key_ngrams)


def associate_variable_values(exp_text):
    # reorder exp_text to arrange variable values in order
    can_str = exp_text
    can_str = re.sub(r'<VL ([^<>]+)>([^<>]+)</VL> <VML ([^<>]+)>([^<>]+)</VML> <VL ([^<>]+)>([^<>]+)</VL>', r'<VL \1>\2</VL> <VML \3>\4</VML>; <VL \5>\6</VL>', can_str) 
    can_str = re.sub(r'<VML ([^<>]+)>([^<>]+)</VML> (-|to|and) <VML ([^<>]+)>([^<>]+)</VML>( of| for) <VL ([^<>]+)>([^<>]+)</VL>', r'<VL \7>\8</VL> <VML \1>\2</VML> \3 <VML \4>\5</VML>', can_str) 
    can_str = re.sub(r'<VML ([^<>]+)>([^<>]+)</VML>( of| for) <VL ([^<>]+)>([^<>]+)</VL>', r'<VL \4>\5</VL> <VML \1>\2</VML>', can_str) 
    
    # find association    
    variables, vars_values = [], []
    start = 0
    while can_str.find('<VL') >-1 and can_str.find('<VML') >-1:
        con1 = can_str.find('<VL')
        start = con1 # In EHR only map to items on dictionary
        #start = 0 if start == 0 else con1
        end = can_str.find('<VL' , con1+1)
        if end > -1:
            text = can_str[start:end] # pos could be -1 so curr_str always ends with a space
            can_str = can_str[end:]
        else:
            text = can_str[start:] # pos could be -1 so curr_str always ends with a space
            can_str = ''
        # get all values in the range
        var =re.findall(r'<VL Label=([^<>]+) Source=([^<>]+)>([^<>]+)</VL>', text) # get last VL label as variable
        values =re.findall(r'<VML Logic=([^<>]+) Unit=([^<>]*)>([^<>]+)</VML>', text)
        if len(var) > 0 and len(values) > 0:
            variables.append(var[0][0])
            var_values = []
            for value in values: 
                logic_for_view = value[0].replace('greater', '>').replace('lower', '<').replace('equal', '=').replace('_', '')
                var_values.append([var[0][0], logic_for_view, value[2], value[1].strip()])
            vars_values.append(var_values)

    return (variables, vars_values)


def context_validation (var_values, allow_units, error_units):

    # unit based validation
    curr_exps = []
    allow_units = (str(allow_units).replace("TEMPORAL", temporal)).split('|')
    error_units = (str(error_units).replace("TEMPORAL", temporal)).split('|')
    for exp in var_values:
        if exp[3].startswith('x ') or exp[3].startswith('times'):
            condition = True
        elif error_units == ['ALL_OTHER']:
            condition = (exp[3]=='' or exp[3] in allow_units)
        else:
            condition = (exp[3]=='' or exp[3] in allow_units or exp[3] not in error_units)
        if condition:
            curr_exps.append(exp)

    return curr_exps

       
       
#====================normalize the unit and their corresponding values
def normalization (nor_unit, exps):
#     for i in xrange(len(exps)):
    exp_temp = []
    for exp in exps:
        if ' x ' in exp[2]: 
            temp = exp[2].strip().split(' x ')
            exp[2] = 1
            for tem in temp:
                exp[2] = exp[2] * float(tem)
        elif '^' in exp[2]:
            temp = exp[2].split('^')
            x,y = float(temp[0].strip()),float(temp[1].strip())
            exp[2] = math.pow(x, y)
        else:
            exp[2] = float(exp[2])
        # start define unit conversion
        if nor_unit == '%':
            if exp[3] == '' and exp[2] < 1:
                exp[2], exp[3] = exp[2]*100.0, nor_unit
            elif exp[3].startswith('percent'):
                exp[3] = nor_unit
            elif exp[3].startswith('mmol/mol'):
                exp[2], exp[3] = exp[2]/10.0, nor_unit
            elif exp[3] =='':
                exp[3] = nor_unit
        elif nor_unit == 'mmol/l':
            if exp[3] == '' and exp[2] >= 60:
                exp[3] = 'mg'
            if exp[3].startswith('mg'):
                exp[2], exp[3] = exp[2]/18.0, nor_unit
            elif exp[3].startswith('g/l'):
                exp[2], exp[3] = exp[2]*7.745, nor_unit
        elif nor_unit == 'kg/m2':            
            if exp[3] != '' and exp[3] != 'kg/m2':
                exp[3] = nor_unit
            elif exp[3] == '':
                exp[3] = nor_unit
        elif nor_unit == 'mg/dl':
            if exp[3] == '' and exp[2] >= 100:
                exp[3] = 'mol'
            if exp[3].startswith('umol') or exp[3].startswith('mol') or exp[3].startswith('micromol'):
                exp[2], exp[3] = exp[2]/88.4, nor_unit
            elif exp[3] == 'mmol/l':
                exp[2], exp[3] = exp[2]*18.0, nor_unit
            elif exp[3].startswith('mg/g'):
                exp[2], exp[3] = exp[2]/1000.0, nor_unit
        elif nor_unit == 'kg':
            if exp[3] == 'grams':
                exp[2], exp[3] = exp[2]/1000.0, nor_unit
            elif exp[3] == 'lb':
                exp[2], exp[3] = 0.453592 * exp[2], nor_unit
        elif nor_unit == 'm':
            if exp[3] == 'cm':
                exp[2], exp[3] = exp[2]/100.0, nor_unit
            elif exp[3] == 'feet':
                temp = str(exp[2]).split('.')
                exp[2], exp[3] = (int(temp[0]) + int(temp[1]) / 12 )* 0.3048 , nor_unit
        elif nor_unit == 'celsius':
            if (exp[3] in ['celsius', 'degrees c']) | (exp[2] < 50):
                exp[3] = nor_unit
            elif (exp[3] in ['fahrenheit', 'degrees f']) | (exp[2] >= 50):
                exp[2], exp[3] = (exp[2] - 32 ) /9 * 5, nor_unit
        elif nor_unit == 'seconds':
            if (exp[3] in ['sec', 's', 'second', 'seconds']):
                exp[3] = nor_unit
            elif (exp[3] in ['m', 'min', 'minute', 'minutes']):
                exp[2], exp[3] = exp[2] * 60, nor_unit
            
        elif exp[3] == '' and nor_unit != "":
            exp[3] = nor_unit
        exp[2] = round(exp[2], 2)
        exp_temp.append(exp)
#         exps[i] = exp_temp
    return exp_temp
        
   
# heuristic rule-based validation     
def hr_validation(exps_temp, min_value, max_value):
    # ------------------ judge an exp by its value comparing with average value. 100 mg/dl, 1 (day), in this case, 1 (day) will be removed
    exps = []
    tagg_temp = []
    # validation by comparing with average value step1. This has been tested to be not as valid as the previous validation method
#     total, num = 0.0, 0.0
#     for exp in exps_temp:
#         if exp[3] <> '':
#            total += float(exp[2])
#            num += 1
           
    thre1, thre2 = 2.0, 8.0 
    for exp in exps_temp:
        if exp[3].startswith('x ') or exp[3].startswith('times'):
            tagg_temp.append(exp)
            continue
        # validation by heuristic rules
        if float(exp[2]) < min_value/thre1 or float(exp[2]) > max_value*thre1: 
            continue

        # validation by comparing with average value step2. This has been tested to be not as valid as the previous validation method            
#         if exp[3] == '' and num > 0 and (total/num >= thre2*float(exp[2]) or float(exp[2]) >= thre2*total/num):
#             continue
        
        tagg_temp.append(exp)
    return tagg_temp

