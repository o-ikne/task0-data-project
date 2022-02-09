## numpy to handle arrays & matices
import numpy as np

## pandas to handle dataframes
import pandas as pd
import codecs

def read_file(fname):
	
	"""return a dataframe given a text file
	
	Parameters
	----------
	fname: string,
		File path.
		
	Returns
	-------
	df: DataFrame of shape (nlines, ncols),
		The dataframe containing the (lemma, form, morphological attributes).
	"""
	
	## instantiate the dataframe
	df = []
	
	## read file
	with codecs.open(fname, 'rb', encoding='utf-8') as f:
		## extract words from lines
		for line in f:
			row = line.split('\t')
			row[-1] = row[-1].strip('\n').strip('\r')
			df.append(row)
	
	## convert list lo dataframe
	if len(df[0]) == 3:
		columns = ['lemma', 'form', 'attributes']
	else:
		columns = ['lemma', 'attributes']
	df = pd.DataFrame(df, columns=columns).dropna()
	
	return df
	
	
def pad(array, n, val):
	"""to pad a given vector to size n with value val"""

	return np.append(array, np.full(n - len(array), val))


def vect2word(vect, char_dict):
	"""transform vect of integers to word 
	given the character dictionnary that maps each number to a character"""
	
	word = ''.join([char_dict[i] for i in vect if i])
	return word
