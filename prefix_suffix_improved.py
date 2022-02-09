#!/usr/bin/env python
#
# File Name : prefix_suffix_improved.py
#
# Description : IMPORVED PREFIX SUFFIX APPROACH (IPS)
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
    
def get_predictions(X_test, y_pred):
    pred_forms = []
    for lemma, y in zip(X_test, y_pred):
        prefix, suffix = y
        form = prefix + lemma + suffix
        pred_forms.append(form)
    return pred_forms
    
    
def encode_attributes(attributes, max_n_attrs, attributes_dict):
    """(multi-)onehot encoding for attributes"""
    encoded = np.zeros(len(attributes_dict))
    encoded[[attributes_dict[attr] for attr in attributes]] = 1
    return encoded
    
def create_trainset(lemmas, forms, attributes, max_lemma, max_form, max_attrs, char_dict, attr_dict):
    """create a bag of words training set"""
    ## create X and y train
    X_train, y_train = [], []
    for lemma, form, set_attrs in zip(lemmas, forms, attributes):
        x, y = [], []
        
        l = []
        for char in lemma:
            l.append(char_dict[char])
            
        for char in form:
            y.append(char_dict[char])

        at = []
        for attr in set_attrs:
            at.append(attr_dict[attr])


        x = np.append(pad(l, max_lemma, 0), pad(at, max_attrs, 0))
        X_train.append(x)
        y_train.append(pad(y, max_form, 0))
        
    return np.array(X_train), np.array(y_train)
    
    
def prefix_suffix(train_file, test_file, verbose=1):
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

	if verbose >= 1:
		print(f'- Number of training samples: {n_train}')
		print(f'- Number of testing samples : {n_test}')
		print(f"- Number of unique characters: {n_chars}")
		print(f"- Number of unique morphological attributes: {n_attrs}")
	if verbose == 2:
		print(f"- Morphological attributes: {', '.join(unique_attrs)}")
		print(f"- Characters: {', '.join(unique_chars)}")
	
		
	df_train['prefix'] = df_train.apply(lambda col: get_prefix(col.form, col.lemma), axis=1)
	df_train['suffix'] = df_train.apply(lambda col: get_suffix(col.form, col.lemma), axis=1)
	
			
	## bag of words for characters
	char_dict = dict(zip(unique_chars, range(1, len(unique_chars)+1)))
	inv_char_dict = {n:char for char, n in char_dict.items()}

	## bag of words for attributes
	attr_dict = {attr:i for i, attr in enumerate(unique_attrs, start=1)}

	## compute maximum possible characters in a lemma and a form
	max_lemma_length = df_train['lemma'].apply(list).apply(len).max()
	max_form_length = df_train['form'].apply(lambda x: len(list(x))).max()

	## compute maximum possible number of attributes
	max_n_attrs = df_train['attributes'].apply(lambda x: len(x.split(';'))).max()

	## get training & test set
	X_train, y_train = create_trainset(df_train['lemma'].values,
		                               df_train['form'].values, 
		                               df_train['attributes'].apply(lambda x:x.split(';')).values,
		                               max_lemma_length, max_form_length, max_n_attrs, 
		                               char_dict, attr_dict)

	X_test, y_test = create_trainset(df_test['lemma'].values, 
		                             df_test['form'].values,
		                             df_test['attributes'].apply(lambda x:x.split(';')).values, 
		                             max_lemma_length, max_form_length, max_n_attrs, 
		                             char_dict, attr_dict)
                                 
	## create target array
	y_train = df_train[['prefix', 'suffix']].to_numpy()


	## bag of words for attributes
	attributes_dict = dict(zip(unique_attrs, range(n_attrs)))

	## encode attributes   
	encoded_attributes = np.array([encode_attributes(attrs.split(';'), max_n_attrs, attributes_dict) for attrs in df_train['attributes'].to_list()])
	encoded_test_attributes = np.array([encode_attributes(attrs.split(';'), max_n_attrs, attributes_dict) for attrs in df_test['attributes'].to_list()])

	## new training data
	new_X_train = np.concatenate((X_train[:, :max_lemma_length], encoded_attributes), axis=1)
	new_X_test = np.concatenate((X_test[:, :max_lemma_length], encoded_test_attributes), axis=1)

	## create & fit an RF model
	clf = RandomForestClassifier(random_state=0)
	clf.fit(new_X_train, y_train)

	## make predictions
	y_pred = clf.predict(new_X_test)
	
	## vector to word
	words_test = [vect2word(vect, inv_char_dict) for vect in y_test]
	words_prediction = get_predictions(df_test['lemma'].to_numpy(), y_pred)

	print(f'- The word by word accuracy          : {word_accuracy(words_prediction, words_test)}')
	print(f'- The character by character accuracy: {character_accuracy(words_prediction, words_test)}')
	
	
if __name__ == "__main__":
	train_file = sys.argv[1]
	test_file = sys.argv[2]
	verbose = 1
	if len(sys.argv) > 3:
		verbose = int(sys.argv[3])
		
	prefix_suffix(train_file, test_file, verbose)
