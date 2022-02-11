#!/usr/bin/env python
#
# File Name : root_prefix_suffix.py
#
# Description : ROOT PREFIX SUFFIX (RPS) APPROACH
#
# Creation Date : 02-09-2022
# Authors : Omar Ikne & Zakaria Boulkhir


## libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from tqdm import tqdm
import codecs
import os
import sys
from utils.utils import *
from utils.eval import *
import unidecode
from sklearn.ensemble import RandomForestClassifier




def get_prefix(form, lemma):
    """return the prefix from the given form and lemma"""
    if lemma in form:
        idx = form.index(lemma)
        return form[:idx]
    return ''

def get_suffix(form, lemma):
    """return the suffix from the given lemma and form"""
    if lemma in form:
        idx = form.index(lemma)
        return form[idx + len(lemma):]
    return ''

def remove_prefix(form, prefix):
    """remove the prefix from the form"""
    if form.startswith(prefix):
        return form[len(prefix):]
    return form

def remove_suffix(form, suffix):
    """remove the suffix from the form"""
    if suffix and form.endswith(suffix):
        return form[:-len(suffix)]
    return form

def get_lemma(form, prefix, suffix):
    """return the lemma from the form given the prefix and the suffix"""
    lemma = remove_suffix(form, suffix)
    lemma = remove_prefix(lemma, prefix)
    return lemma
    
def get_root(string_1, string_2):
    """return the root intersection of two strings"""
    
    if len(string_1) > len(string_2):
        larger_s = string_1 
        smaller_s = string_2
    else:
        larger_s = string_2
        smaller_s = string_1
        
    inter = ''
    for i in range(len(larger_s)):
        for j in range(i, len(larger_s)+1):
            if j - i < len(inter):
                continue
            part = larger_s[i:j]
            
            if part in smaller_s and len(part) > len(inter):
                inter = part
        
    return inter
    
def bag_of_chars(lemmas, max_lemma, char_dict):
    """create a bag of words training set"""
    ## create X and y train
    X_train = []
    for lemma in lemmas:
        
        x = []
        for char in lemma:
            x.append(char_dict[char])
            
        X_train.append(pad(x, max_lemma, 0))
        
    return np.array(X_train)
    
def get_predictions(X_test, y_pred, inv_char_dict, max_root_length, inv_form_pref_dict, inv_form_suff_dict):
    predictions = []
    for x, y in zip(X_test, y_pred):
        root, lemma_pref, form_pref, lemma_suff, form_suff = y[:max_root_length], *y[max_root_length:]
        pref = inv_form_pref_dict[form_pref] 
        suff = inv_form_suff_dict[form_suff]
        form = pref + vect2word(root, inv_char_dict) + suff
        predictions.append(form)
    return predictions
    
    
def root_prefix_suffix(train_file, test_file, verbose=1):
	""""""
	
	## dataframe
	df_train = read_file(train_file)
	df_test = read_file(test_file)

	## number of training & testing samples
	n_train = df_train.shape[0]
	n_test = df_test.shape[0]
	 

	## get the number of unique characters
	text = ''.join(df_train[['lemma', 'form']].to_numpy().flatten())

	## get (number of) unique characters
	unique_chars = sorted(set(text))
	n_chars = len(unique_chars)
	

	## get unique morphological attributes
	morph_attrs = ';'.join(df_train['attributes'].to_list()).split(';')
	morph_attrs = np.asarray(morph_attrs)
	unique_attrs = sorted(set(morph_attrs)) ## sort to keep same order
	n_attrs = len(unique_attrs)
	
	## bag of words for characters
	char_dict = dict(zip(unique_chars, range(1, len(unique_chars)+1)))
	inv_char_dict = {n:char for char, n in char_dict.items()}


	if verbose >= 1:
		print(f'- Number of training samples: {n_train}')
		print(f'- Number of testing samples : {n_test}')
		print(f"- Number of unique characters: {n_chars}")
		print(f"- Number of unique morphological attributes: {n_attrs}")
	if verbose == 2:
		print(f"- Morphological attributes: {', '.join(unique_attrs)}")
		print(f"- Characters: {', '.join(unique_chars)}")
	
		
	## extract root
	df_train['root'] = df_train.apply(lambda col: get_root(col.lemma, col.form), axis=1)

	## extract prefixes
	df_train['lemma_prefix'] = df_train.apply(lambda col: get_prefix(col.lemma, col.root), axis=1)
	df_train['form_prefix'] = df_train.apply(lambda col: get_prefix(col.form, col.root), axis=1)

	## extract suffixes
	df_train['lemma_suffix'] = df_train.apply(lambda col: get_suffix(col.lemma, col.root), axis=1)
	df_train['form_suffix'] = df_train.apply(lambda col: get_suffix(col.form, col.root), axis=1)


	## compute maximum possible characters in a lemma and a form
	max_lemma_length = df_train['lemma'].apply(list).apply(len).max()
	max_form_length = df_train['form'].apply(lambda x: len(list(x))).max()

	## compute maximum possible number of attributes
	max_n_attrs = df_train['attributes'].apply(lambda x: len(x.split(';'))).max()

	## get training & test set
	X_train = bag_of_chars(df_train['lemma'].to_numpy(), max_lemma_length, char_dict)
	X_test  = bag_of_chars(df_test['lemma'].to_numpy(), max_lemma_length, char_dict)


	## bag of words
	lemma_prefix_dict = dict(zip(df_train['lemma_prefix'].unique(), range(1, len(df_train['lemma_prefix'].unique())+1)))
	lemma_suffix_dict = dict(zip(df_train['lemma_suffix'].unique(), range(1, len(df_train['lemma_suffix'].unique())+1)))
	form_prefix_dict  = dict(zip(df_train['form_prefix'].unique(), range(1, len(df_train['form_prefix'].unique())+1)))
	form_suffix_dict  = dict(zip(df_train['form_suffix'].unique(), range(1, len(df_train['form_suffix'].unique())+1)))

	inv_form_pref_dict = {n:pref for pref, n in form_prefix_dict.items()}
	inv_form_suff_dict = {n:suff for suff, n in form_suffix_dict.items()}

	## encode roots, suffixes & prefixes
	max_length_root = df_train['root'].apply(len).max()
	encoded_roots = bag_of_chars(df_train['root'].to_list(), max_length_root, char_dict)
	encoded_lemma_suffix = df_train['lemma_suffix'].apply(lemma_suffix_dict.get).to_numpy().reshape(-1, 1)
	encoded_form_suffix  = df_train['form_suffix'].apply(form_suffix_dict.get).to_numpy().reshape(-1, 1)
	encoded_lemma_prefix = df_train['lemma_prefix'].apply(lemma_prefix_dict.get).to_numpy().reshape(-1, 1)
	encoded_form_prefix  = df_train['form_prefix'].apply(form_prefix_dict.get).to_numpy().reshape(-1, 1)

	## create target labels
	y_train = np.concatenate((encoded_roots, encoded_lemma_prefix, encoded_form_prefix, encoded_lemma_suffix, encoded_form_suffix), axis=1)


	## create & fit an RF model
	clf = RandomForestClassifier(random_state=0)
	clf.fit(X_train, y_train)

	## make predictions
	y_pred = clf.predict(X_test)
	
	## vector to word
	words_test = df_test['form']
	words_prediction = get_predictions(df_test['lemma'].to_numpy(), y_pred, inv_char_dict, max_length_root,
						inv_form_pref_dict, inv_form_suff_dict)

	print(f'- The word by word accuracy          : {word_accuracy(words_prediction, words_test)}')
	print(f'- The character by character accuracy: {character_accuracy(words_prediction, words_test)}')
	
	
if __name__ == "__main__":
	train_file = sys.argv[1]
	test_file = sys.argv[2]
	verbose = 1
	if len(sys.argv) > 3:
		verbose = int(sys.argv[3])
		
	root_prefix_suffix(train_file, test_file, verbose)
