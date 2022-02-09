
import numpy as np


def word_accuracy(y_true, y_pred):
	"""accuracy word by word"""
	return np.sum([accuracy_metric(s1, s2) for s1, s2 in zip(y_true, y_pred)]) / len(y_true)


def accuracy_metric(string_1, string_2):
	"""return 1 if string_1 = string_2, 0 otherwise"""
	return int(string_1 == string_2)
	
	
def Levenshtein_distance(y_true, y_pred):
	""""""
	dist = 0
	for word1, word2 in zip(y_true, y_pred):
		dist += distance(word1, word2)
		
	return dist
	

def character_accuracy(y_true, y_pred):
	"""accuracy character by character"""
	## character accuracy
	char_accuracy = 0
	total_chars = 0
	for word1, word2 in zip(y_true, y_pred):
		char_accuracy += max(len(word1), len(word2)) - distance(word1, word2)
		total_chars += max(len(word1), len(word2)) 
		
	return char_accuracy / total_chars

def eval_acc(gold, sys):
    return np.sum([g == s for g, s in zip(gold, sys)]) / float(len(gold))

def distance(str1, str2):
    """Simple Levenshtein implementation for eval."""
    m = np.zeros([len(str2)+1, len(str1)+1])
    for x in range(1, len(str2) + 1):
        m[x][0] = m[x-1][0] + 1
    for y in range(1, len(str1) + 1):
        m[0][y] = m[0][y-1] + 1
    for x in range(1, len(str2) + 1):
        for y in range(1, len(str1) + 1):
            if str1[y-1] == str2[x-1]:
                dg = 0
            else:
                dg = 1
            m[x][y] = min(m[x-1][y] + 1, m[x][y-1] + 1, m[x-1][y-1] + dg)
    return int(m[len(str2)][len(str1)])
