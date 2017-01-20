''' several useful functions '''
import numpy as np
from numpy.random import random_sample
import time
import sys as Sys

np.random.seed(123)


def log_normalize(v):
    ''' return log(sum(exp(v)))'''

    log_max = 100.0
    if len(v.shape) == 1:
        max_val = np.max(v)
        log_shift = log_max - np.log(len(v)+1.0) - max_val
        tot = np.sum(np.exp(v + log_shift))
        log_norm = np.log(tot) - log_shift
        v = v - log_norm
    else:
        max_val = np.max(v, 1)
        log_shift = log_max - np.log(v.shape[1]+1.0) - max_val
        tot = np.sum(np.exp(v + log_shift[:,np.newaxis]), 1)

        log_norm = np.log(tot) - log_shift
        v = v - log_norm[:,np.newaxis]

    return (v, log_norm)

def log_sum(log_a, log_b):
	''' we know log(a) and log(b), compute log(a+b) '''
	v = 0.0;
	if (log_a < log_b):
		v = log_b+np.log(1 + np.exp(log_a-log_b))
	else:
		v = log_a+np.log(1 + np.exp(log_b-log_a))
	return v


def argmax(x):
	''' find the index of maximum value '''
	n = len(x)
	val_max = x[0]
	idx_max = 0

	for i in range(1, n):
		if x[i]>val_max:
			val_max = x[i]
			idx_max = i		

	return idx_max			


def discrete_sample(values, weights, size):
    bins = np.add.accumulate(weights/np.sum(weights))
    samples = values[np.digitize(random_sample(size), bins)]
    return samples


def data_format_regularization():
	pass


def printProgress (iteration, total, prefix = '', suffix = '', decimals = 2, barLength = 100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : number of decimals in percent complete (Int) 
        barLength   - Optional  : character length of bar (Int) 
    """
    filledLength    = int(round(barLength * iteration / float(total)))
    percents        = round(100.00 * (iteration / float(total)), decimals)
    bar             = '#' * filledLength + '-' * (barLength - filledLength)
    Sys.stdout.write('%s [%s] %s%s %s\r' % (prefix, bar, percents, '%', suffix)),
    Sys.stdout.flush()