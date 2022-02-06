## numpy to handle arrays & matices
import numpy as np

## pandas to handle dataframes
import pandas as pd
import codecs

def read_file(fname, train=True):
    
    """return a dataframe given a text file
    
    Parameters
    ----------
    file: string,
        File path.
    
    ncols: int,
        Number of colmns in the file.
        
    Returns
    -------
    df: DataFrame of shape (nlines, ncols),
        The dataframe containing the text file.
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
    if train:
        columns = ['lemma', 'form', 'attributes']
    else:
        columns = ['lemma', 'attributes']
    df = pd.DataFrame(df, columns=columns).dropna()
    
    return df
    
    
def eval_acc(gold, sys):
    return sum([1 if g == s else 0 for g, s in zip(gold, sys)]) / float(len(gold))

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
