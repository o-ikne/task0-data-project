# -*- coding: utf-8 -*-
"""A nae version of the probabilistic approach. This approach consists on:
* Computing the probability of changes (in the from of Prefix-Suffix) to be representing a certain number of Morphological properties
* Selecting the most probable change for a new sample of the testing dataset 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def suffix_prefix_detection(s, c):
  """
  Looks up the prefix and suffix of a lemma c. The maximal substring in s that correspond to lemma is extracted
  :params: s for the form and c for the lemma
  :return: the prefix+"-"+suffix, where "-" replaces the inducted substring of the lemma c
  """
  # Perte d'information, quoi faire pour rattaraper ? Seek the upgraded approach
  for i in range(len(c),0, -1):
    np = c[:i]
    ns = s.replace(np, '-')
    if ns != s:
      # The substring found, and is replaced
      break
  return ns

def best_candidates_from_probabilities_naive(morphs, probs):
  """
  An approach that searches for potential candidates for the changes.
  It simply regroupes the different changes that have been recorded to have had *the exact same* morphological attributes
  :params:  * morphs: (String) morphological attributes, mainly on the test data set
            * Probs: A dictionary that regroupe changes to their morphological attributes and there occurences on training dataset
  :returns: A dictionnary with potential changes to make to a lemma, and the corresponding occurence within the training dataset
  """
  possible_predictions = dict()
  for c in probs.keys():
    dict_probs = probs[c]
    if set(morphs.split(";")) == set(dict_probs.keys()):
      if c in possible_predictions.keys():
        possible_predictions[c] += 1
      else:
        possible_predictions[c] = 1
    # If an example is not met on the training set, we abandon the cause. Hence, naïve approach.
  return possible_predictions


def train(fd):
  """
    Simple training function creating the tuple (lemma, form, M.A) also the tuple corresp.
    :return: corresp (array) The changes "prefix-suffix", the attributes and occurences that correspond to them.
    :params: fd is a file descriptor
  """
  # Create words, not strictly speaking included in the process of training
  l = fd.readline()
  ws, fs, rs = list(), list(), list()
  while l:
      w, f, r = list( map(lambda a : a.strip(), l.split('\t')) )
      ws.append(w); fs.append(f); rs.append(r)
      l = fd.readline()
  # Create a dictionnary with rule (pre-suf) as keys and have the corresponding morph attributes 
  corresp = dict()
  for i in range(len(ws)):
    rule = suffix_prefix_detection(fs[i], ws[i])
    if rule in corresp.keys():
      for k in rs[i].split(";"):
        if k in corresp[rule]:
          corresp[rule][k] += 1
        else:
          corresp[rule][k] = 1
    else:
        corresp[rule] =  {key: 1 for key in rs[i].split(";")}
  return (ws, fs, rs), corresp


def generate_test(fd):
  """
    In the case of test datasets, the tuple (lemma, M.A) only is returned.
  """
  i = 0
  l = fd.readline()
  ws, rs = list(), list()
  while l:
      w, r = list( map(lambda a : a.strip(), l.split('\t')) )
      ws.append(w); rs.append(r)
      l = fd.readline()
      i += 1
  return (ws, rs)

def change_learned_accuracies_into_probabilities(corresp):
  """
    A function that changes occurences in corresp to probabilities
    :Example:
    > corresp = {"pre-é": {"V":7, "Noun":3}, "dé-é" : {"V":8, "Adj":2}}
    > probs   = {"pre-é": {"V":0.7, "Noun":0.3}, "dé-é" : {"V":0.8, "Adj":0.2}}
  """
  probs = corresp.copy()
  for key1 in probs.keys():
    sumValues = sum(probs[key1].values())
    for key2 in probs[key1].keys():
      probs[key1][key2] = probs[key1][key2] / sumValues
  return probs

def selectMaxFromDict(dictio):
  """
    Selects the key that correspond to the highest value in a dictionary
  """
  ks = np.array(list(dictio.keys()))
  vs = np.array(list(dictio.values()))
  ord = np.argsort(vs)
  ks_n = ks[ord]
  return ks_n[0] if not ks_n.shape[0] == 0 else '-'

def selectFromDict(dictio, threshold):
  """
    Selects cases (keys) from dictionary where the values are >= threshold. The returned are ordered from highest to lowest.
  """
  ks = np.array(list(dictio.keys()))
  vs = np.array(list(dictio.values()))
  ks = ks[vs>=threshold]
  vs = vs[vs>=threshold]
  ord = np.argsort(vs)
  ks_n = ks[ord]
  return ks_n
  
def CharByCharAccuracy(lemma1, lemma2):
  """
    An example to defining the distance between two lemmas
  """
  d = suffix_prefix_detection(lemma1, lemma2)
  d = d.replace('-', '')
  return np.mean(list(map(lambda a:1,d))+[0])/max(len(lemma1), len(lemma2))

def printAccuracies(fdTrain, fdTest):
  """
	Prints accuracies in the case where the test set disposes actually of forms that are to be predicted
  """
  # Training data and dependencies
  (ws, fs, rs), corresp = train(fdTrain)
  print("Number of the training examples are ", len(ws))
  probs = change_learned_accuracies_into_probabilities(corresp)
  # Test data
  (ws_t, fs_t, rs_t), _ = train(fdTest)
  print("Number of the testing examples are =", len(ws_t))
  acc, acc2 = list(), list()
  # For each test sample
  for i in range(len(rs_t)):
    c = selectMaxFromDict(best_candidates_from_probabilities_naive(rs_t[i], probs))
    # Without taking the lemma to account
    cbc = CharByCharAccuracy(c, fs_t[i])
    c = c.replace("-", ws_t[i])
    acc.append(cbc)
    acc2.append(fs_t[i] == c)
    
  print("Char-by-char accuracy is:", 1-np.mean(acc), ". Word-to-word on the other hand is:", np.mean(acc2))

##################################
##    UNUSED & PROBS USELESS	##
##################################

def look_changes(morphs, probs, threshold):
  """
  A very naïve approach that searches for potential candidates for the changes.
  It simply regroupes the different changes that have been recorded to have had one amongst morphological attributes
  :params:  * morphs: (array) morphological attributes, mainly on the test data set
            * Probs: A dictionary that regroupe changes to their morphological attributes and there occurences on training dataset
            * Threshold: Unused, but serves as an indication to ignore underrated examples
  :returns: A dictionnary with potential changes to make to a lemma, and the corresponding accumulative probabilities
  within the training dataset
  """
  possible_predictions = dict()
  for c in probs.keys():
    dict_probs = probs[c]
    for morph in morphs:
      if morph in dict_probs.keys() and dict_probs[morph] >= threshold:
        if c in possible_predictions.keys():
          possible_predictions[c] += dict_probs[morph]
        else:
          possible_predictions[c] = dict_probs[morph]
  return possible_predictions

def lemmasAndChanges(ws, fs, rs):
  """
    From (lemma, form, M.A.) return (lemma, changes, M.A.)
  """
  cs = list(map(lambda f, w: suffix_prefix_detection(f, w), zip(fs, ws)))
  return ws, cs, rs

