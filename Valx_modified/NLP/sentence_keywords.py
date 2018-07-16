# -*- encoding: utf-8 -*-
# Extract quantitative variables and their values from free eligibility criteria text
# Created by Tony HAO, th2510@columbia.edu

import os,sys,re,string
import NLP.sentence as NLP_sent
import NLP.word as NLP_word
import nltk
				
#=========================================syntax
def keywords_syntax_nltk(sentence):
	global text_terms
	terms = []
	phrases = NLP_sent.phrase_splitting(sentence)		
	for phrase in phrases:
		if len(phrase) <= 2: # e.g.'ii'
			continue
		if phrase in text_terms:
			phrase_terms = text_terms[phrase]
		else:
			#-------------------POS tagging output
			words = NLP_word.word_splitting(phrase.lower())
			pos_tags = NLP_word.word_pos_tagging(words)
	
			#-------------------parsed tree
			grammar = r"""
				NBAR:
					# Nouns and Adjectives, terminated with Nouns
					{<NN.*|JJ>*<NN.*>}
			
				NP:
					{<NBAR>}
					# Above, connected with in/of/etc...
					{<NBAR><IN><NBAR>}
			"""
		
			cp = nltk.RegexpParser(grammar, loop=2)
			cp_tree = cp.parse(pos_tags)
			phrase_terms = get_terms(cp_tree)
			text_terms[phrase] = phrase_terms

		terms += phrase_terms 

	keywords = []
	for term in terms:
		if len(term) > 0:
			keywords.append(' '.join(term))
	return keywords


# Ref to https://gist.github.com/879414
#from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = nltk.WordNetLemmatizer()
#stemmer = nltk.stem.porter.PorterStemmer()
from nltk.corpus import stopwords
stopwords = stopwords.words('english')

def normalise(word):
    """Normalises words to lowercase and stems and lemmatizes it."""
    word = word.lower()
#    word = stemmer.stem_word(word)
    word = lemmatizer.lemmatize(word)
    return word

def acceptable_word(word):
    """Checks conditions for acceptable word: length, stopword."""
    accepted = bool(2 <= len(word) <= 40
        and word.lower() not in stopwords)
    return accepted

def leaves(tree):
    """Finds NP (nounphrase) leaf nodes of a chunk tree."""
    for subtree in tree.subtrees(filter = lambda t: t.node=='NP'):
        yield subtree.leaves()

def get_terms(tree):
    for leaf in leaves(tree):
        term = [ normalise(word) for word, tag in leaf
            if acceptable_word(word) ]
        yield term
 
 
# =====================================ngrams
def keywords_ngrams(sentence):
	ngrams = []
	phrases = NLP_sent.phrase_splitting(sentence)		
	for phrase in phrases:
		if len(phrase) <= 2: # e.g.'ii'
			continue
		words = NLP_word.word_splitting(phrase.lower())
		stop_pos = [] # record all positions of stop  or non-preferred (POS) words in the phrase to increase efficiency
		for i in range(len(words)):
			type = word_checking_stop(words[i])
			stop_pos.append(type)
		
		# Generate n-gram
		for i in range(len(words)):
			if 0 < stop_pos[i]:
				continue
			for j in reversed(range(i+1, min(len(words), i+4)+1)): # the maximum length of a ngram is 5
				if 0 < stop_pos[j-1]:# check validity
					continue
				ngram = ' '.join(words[i:j])
				if len(ngram)>2: # at least two characters
					ngrams.append(ngram)

	return ngrams


def keywords_ngrams_reverse(sentence):
	ngrams = []
	splitter=re.compile('[\(\){}\[\]?!,:;]')
	phrases = splitter.split(sentence)
	for phrase in reversed(phrases):
		if len(phrase) <= 1: # e.g.'ii'
			continue
		# 	words = NLP_word.word_splitting(sentence.lower()) # method 1: NLP
		splitter=re.compile('[^a-zA-Z0-9_-]') # method 2: splitters
		words = splitter.split(phrase)
			
		stop_pos = [] # record all positions of stop  or non-preferred (POS) words in the phrase to increase efficiency
		for i in range(len(words)):
			type = word_checking_stop(words[i])
			stop_pos.append(type)
		
		# Generate n-gram
		for i in reversed(range(len(words))):
			if 0 < stop_pos[i]:
				continue
			for j in range(max(0, i-98), i+1): # the maximum length of a ngram is 10
				if 0 < stop_pos[j]:# check validity
					continue
				ngram = ' '.join(words[j:i+1])
				if len(ngram)>1: # at least two characters
					ngrams.append(ngram)

	return ngrams

add_stopwords = ["must","within", "every", "each", "based"]
# check if a word is a stop word
def word_checking_stop(word):
	if len(word) < 1:
		return 1
	elif word[0] in string.punctuation:
		return 2
	elif word[0].isdigit():
		return 3
	elif word in stopwords: 
		return 4
	elif word in add_stopwords:
		return 5
	else:
		return 0
	