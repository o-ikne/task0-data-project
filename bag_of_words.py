#!/usr/bin/env python
#
# File Name : bag_of_words.py
#
# Description : BAG OF WORDS (BOW)
#
# Creation Date : 02-09-2022
# Authors : Omar Ikne & Zakaria Boulkhir


import numpy as np
import pandas as pd
import os
import sys
from utils.utils import *
from sklearn.ensemble import RandomForestClassifier
from utils.eval import *


def create_trainset(lemmas, forms, attributes, max_lemma, max_form, max_attrs, char_dict, attr_dict):
	"""create a bag of words training set"""
	## create X and y train
	X_train, y_train = [], []
	for lemma, form, set_attrs in zip(lemmas, forms, attributes):
		x, y = [], []
		
		l = []
		for char in lemma:
			l.append(char_dict[char[0]])
			
		for char in form:
			y.append(char_dict[char[0]])

		at = []
		for attr in set_attrs:
			at.append(attr_dict[attr[0]])


		x = np.append(pad(l, max_lemma, 0), pad(at, max_attrs, 0))
		X_train.append(x)
		y_train.append(pad(y, max_form, 0))
		
	return X_train, y_train


def bag_of_words(train_file, test_file, verbose=1):
	## dataframe
	df_train = read_file(train_file)
	df_test = read_file(test_file)

	## number of training & testing samples
	n_train = df_train.shape[0]
	n_test = df_test.shape[0]
	
	## get the number of unique characters
	text = ''.join(df_train[['lemma', 'form']].to_numpy().flatten())

	## get (number of) unique characters
	unique_chars = set(text)
	n_chars = len(unique_chars)
	
	## get unique morphological attributes
	morph_attrs = ';'.join(df_train['attributes'].to_list()).split(';')
	morph_attrs = np.asarray(morph_attrs)
	unique_attrs = set(morph_attrs)
	n_attrs = len(unique_attrs)
	
	if verbose >= 1:
		print(f'- Number of training samples: {n_train}')
		print(f'- Number of testing samples : {n_test}')
		print(f"- Number of unique characters: {n_chars}")
		print(f"- Number of unique morphological attributes: {n_attrs}")
	if verbose == 2:
		print(f"- Morphological attributes: {', '.join(unique_attrs)}")
		print(f"- Characters: {', '.join(unique_chars)}")
	
	## bag of words
	char_dict = dict(zip(unique_chars, range(1, len(unique_chars)+1)))
	inv_char_dict = {n:char for char, n in char_dict.items()}

	attr_dict = {attr:i for i, attr in enumerate(unique_attrs, start=1)}
	
	## get all characters in each lemma
	train_lemmas = np.array([[[char] for char in lemma] for lemma in df_train['lemma'].to_list()], dtype='object')
	test_lemmas = np.array([[[char] for char in lemma] for lemma in df_test['lemma'].to_list()], dtype='object')

	## get each attribute
	train_attrs = df_train['attributes'].apply(lambda x: x.split(';'))
	train_attrs = np.array([[[attr] for attr in set_attrs] for set_attrs in train_attrs], dtype='object')
	test_attrs = df_test['attributes'].apply(lambda x: x.split(';'))
	test_attrs = np.array([[[attr] for attr in set_attrs] for set_attrs in test_attrs], dtype='object')

	## get forms
	train_forms = np.array([[[char] for char in form] for form in df_train['form'].to_list()], dtype='object')
	test_forms = np.array([[[char] for char in form] for form in df_test['form'].to_list()], dtype='object')

	## compute maximum possible characters in a lemma and a form
	max_lemma_length = df_train['lemma'].apply(list).apply(len).max()
	max_form_length = df_train['form'].apply(lambda x: len(list(x))).max()

	## compute maximum possible number of attributes
	max_n_attrs = df_train['attributes'].apply(lambda x: len(x.split(';'))).max()
	
	## get training & test set
	X_train, y_train = create_trainset(train_lemmas, train_forms, train_attrs, max_lemma_length, max_form_length, max_n_attrs, char_dict, attr_dict)
	X_test, y_test = create_trainset(test_lemmas, test_forms, test_attrs, max_lemma_length, max_form_length, max_n_attrs, char_dict, attr_dict)
	
	## create & fit an RF model
	clf = RandomForestClassifier(random_state=0)
	clf.fit(X_train, y_train)

	## make predictions
	y_pred = clf.predict(X_test)

	## vects to words
	words_prediction = [vect2word(vect, inv_char_dict) for vect in y_pred]
	words_test = [vect2word(vect, inv_char_dict) for vect in y_test]
	
	print(f'- The word by word accuracy          : {word_accuracy(words_prediction, words_test)}')
	print(f'- The character by character accuracy: {character_accuracy(words_prediction, words_test)}')
	
if __name__ == "__main__":
	train_file = sys.argv[1]
	test_file = sys.argv[2]
	verbose = 1
	if len(sys.argv) > 3:
		verbose = int(sys.argv[3])
		
	bag_of_words(train_file, test_file, verbose)
