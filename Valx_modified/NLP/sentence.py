# -*- encoding: utf-8 -*-
# a set of NLP functions that can be used for text processing
# Created by Tony HAO, th2510@columbia.edu

from nltk import sent_tokenize
import NLP.word as NLP_word
import re, string
import NLP.porter2

# splitting text into Sentences using NLTK tokenization
def sentence_splitting (texts, slen = 1):
	if len(texts) <= 0:
		return []
	
	# splitting
	sentences = []
	text_sents = sent_tokenize(texts)
	if (text_sents != [''] and len(text_sents) >  0):
		for sent in text_sents:
			sent = sent.strip().split('\r') # split strings that contains "\r"
			for sen in sent:
				se = sen.split('. ')
				for s in se: 
					if (NLP_word.words_counting(s) >= slen):
						sentences.append(s)

	return sentences


# splitting text into Sentences using NLTK tokenization
def sentence_splitting_symbols (texts, splitter = None, slen = 1):
	if len(texts) <= 0:
		return []
	
	if splitter is None:
		splitter = "[#!?.]\s"
	# splitting
	sentences = []
	text_sents = re.split(splitter, texts)
	if (text_sents != [''] and len(text_sents) >  0):
		for sent in text_sents:
			sent = sent.strip().split('\r') # split strings that contains "\r"
			for sen in sent:
				if (NLP_word.words_counting(sen) >= slen):
					sentences.append(sen.strip('-').strip())

	return sentences


# separate a sentence into phrases (avoiding sth like "AA (BB" since "(" should be used to separate them)
def phrase_splitting(sentence):
	phrases = []
	splitter=re.compile('[\(\){}\[\]?!,:;]')
	can_phrs = splitter.split(sentence)
	for can_phr in can_phrs:
		can_phr = phrase_cleaning(can_phr)
		if len(can_phr) > 0:
			phrases.append(can_phr)
	return phrases


def phrase_cleaning(txt):
	if len(txt.strip()) >0:
		txt = txt.replace('\n','')
		remove = '"“”#$&|*'
		for r in remove:
			if r in txt:
				txt = txt.replace (r, '')
		
	# 	remove = '\/'
	# 	for r in remove:
	# 		if r in txt:
	# 			txt = txt.replace (r, ' '+r+' ')
			
		while '  ' in txt:
			txt = txt.replace('  ',' ')
		
		txt = txt.strip()
		if len(txt) > 0 and txt[0] in string.punctuation:
			txt = txt[1:]
		if len(txt) > 0 and txt[-1] in string.punctuation:
			txt = txt[0:-1]
		
		txts = txt.split(' ')
		if txts[0].startswith('\\'):
			txt = ' '.join(txts[1:])
	return txt.strip()

def stem_phrase(phrase):
	words = phrase.split()
	for i in range(0, len(words)):
		words [i] = porter2.stem(words[i])
		
	return ' '.join(words)
		