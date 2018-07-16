# a set of NLP functions that can be used for text processing
# Created by Tony HAO, th2510@columbia.edu

from nltk import word_tokenize
from nltk import pos_tag
import re


# separate a phrase into words
def word_splitting(phrase):
	words = word_tokenize(phrase)
	return words


# POS tagging
def word_pos_tagging(words):
	pos = pos_tag (words)
	return pos


# counting the words number for a sentence
def words_counting(sentence):
	return len(sentence.split())


# counting the words number for a sentence
def words_counting2(sentence):
	splitter=re.compile('[^a-zA-Z0-9\\+\\-]')
	words = []
	for singleWord in splitter.split(sentence):
		currWord = singleWord.strip()
		if len(currWord) > 1: # at least 2 character
			words.append(currWord)	
	
	return len(words)