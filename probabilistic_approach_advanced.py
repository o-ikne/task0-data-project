# -*- coding: utf-8 -*-
"""A nae version of the probabilistic approach. This approach consists on:
* Computing the probability of changes (in the from of Prefix-Suffix) to be representing a certain number of Morphological properties
* Selecting the most probable change for a new sample of the testing dataset 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from probabilistic_approach_advanced import *

def best_candidates_from_probabilities(morphs, probs):
  """Same as before, but additionally records the cases where only one of the morphological attributes is present"""
  possible_predictions = dict()
  for c in probs.keys():
    dict_probs = probs[c]
    if np.all(list(map(lambda a : a in dict_probs.keys(), morphs.split(";")))):
      # This is the case where the training set has a sample that had the exacte Morph. Att. and is amplified (+2)
      if c in possible_predictions.keys():
        possible_predictions[c] += 2
      else:
        possible_predictions[c] = 2
    # The case where at least one morph. att. is present. This case is less representable (+1) 
    elif np.any(list(map(lambda a : a in dict_probs.keys(), morphs.split(";")))):
      if c in possible_predictions.keys():
        possible_predictions[c] += 1
      else:
        possible_predictions[c] = 1
    # +0 for the case of no presence. This occurences and amplification serve for the probabilistic model later on.
  return possible_predictions

def best_candidates_from_probabilities_advanced(lemma, morphs, corresp, corresp2, corresp3, dist):
  """
    This is an advanced method. It takes to account some learned information from the training set to 
    select a potential candidates. It, all the same, is still a naïve approach.
    :params:  * corresp: Contains, as probs before, the changes "prefix-suffix", the attributes and occurences that correspond to them.
              * corresp2: Contains changes and the corresponding lemmas.
              * corresp3: Contains the changes and their occurences within the training data set (the world)
              * dist: Distance to defines as one pleases between two lemmas
    :return: The most probable changes taking to account:
    * Their M.A. and those of the test set
    * The probability that these changes occur (if they accur rarely in the training set, so they should in the test set)
    * The distance of the new lemma from the lemmas in the training set (if this lemma is so close to another one, so the changes are 
    most probably the same as those on the training set)
  """
  pp = best_candidates_from_probabilities(morphs, fromCorrespToProbs(corresp))
  new_preds = dict.fromkeys(pp.keys())
  for k in pp.keys():
    dists = list(map(lambda a: dist(a, lemma),corresp3[k]))
    new_preds[k] = pp[k] * corresp2[k] * (np.max(dists) - np.mean(dists))
  minIdx = np.argmax(list(new_preds.values()))
  return np.array(list(new_preds.keys()))[minIdx]


def train(fd):
  """
    Simple training function creating the tuple (lemma, form, M.A) also the tuple (corresp, corresp2, corresp3) as:
              * corresp: The changes "prefix-suffix", the attributes and occurences that correspond to them.
              * corresp2: Contains changes and the corresponding lemmas.
              * corresp3: Contains the changes and their occurences within the training data set (the world)
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
  corresp2 = dict()
  corresp3 = dict()
  for i in range(len(ws)):
    rule = suffix_prefix_detection(fs[i], ws[i])
    if rule in corresp.keys():
      corresp2[rule] += 1
      corresp3[rule].append(ws[i])
      for k in rs[i].split(";"):
        if k in corresp[rule]:
          corresp[rule][k] += 1
        else:
          corresp[rule][k] = 1
    else:
        corresp[rule] =  {key: 1 for key in rs[i].split(";")}
        corresp2[rule] = 1
        corresp3[rule] = [ws[i]]
  return (ws, fs, rs), (corresp, corresp2, corresp3)

def distanceFromLemma(lemma1, lemma2):
  """
    An example to defining the distance between two lemmas
  """
  d = 0
  # Distance is the sum of the character-wise differences
  for i in range(min(len(lemma1), len(lemma2))):
    d += abs(ord(lemma1[i]) - ord(lemma2[i]))
  lemma = lemma1 if len(lemma1) > len(lemma2) else lemma2
  # Adding the difference
  reste = sum(list(map(lambda a : ord(a),lemma[i+1:])))
  return d + reste


def distanceFromLemma_(lemma1, lemma2):
  """
    Another example where first and last character are more put to interest than the reste. And no character-wise distance.
  """
  return 0.5 * (lemma1 == lemma2) + 0.25 * ((lemma1[:2] == lemma2[:2]) + (lemma1[-2:] == lemma2[-2:])) if min(len(lemma1), len(lemma2)) >= 2 else int(lemma1 == lemma2)


def printAccuracies(fdTrain, fdTest):
  """
	Prints accuracies in the case where the test set disposes actually of forms that are to be predicted
  """
  # Training data and dependencies
  (ws, fs, rs), (corresp, corresp2, corresp3) = train(fdTrain)
  print("Number of the training examples are ", len(ws))
  probs = change_learned_accuracies_into_probabilities(corresp)
  # Test data
  (ws_t, fs_t, rs_t), _ = train(fdTest)
  print("Number of the testing examples are =", len(ws_t))
  acc, acc2 = list(), list()
  # Defining a distance that is a sum of the above mentionned ones
  dist = lambda a, b : distanceFromLemma(a, b) - distanceFromLemma_(a, b)
  # For each test sample
  for i in range(len(rs_t)):
    c = best_candidates_from_probabilities_advanced(ws_t[i], rs_t[i], corresp, corresp2, corresp3, dist)
    # Without taking the lemma to account
    cbc = CharByCharAccuracy(lemma1, lemma2)
    c = c.replace("-", ws_t[i])
    acc.append(cbc[0]/cbc[1])
    acc2.append(fs_t[i] == c2)
    
  print("Char-by-char accuracy is:", np.mean(acc), ". Word-to-word on the other hand is:", np.mean(acc2))

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

