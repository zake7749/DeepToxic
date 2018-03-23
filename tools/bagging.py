import sys
import numpy as np
import pandas as pd
from sklearn import *
from scipy.special import expit,logit

ensembeled = sys.argv[1:]
print("Going ensemble on",)
subs = []
for e in ensembeled:
	print(e)
	subs.append(pd.read_csv(e))

classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
for sub in subs[1:]:
	for c in classes:
	    subs[0][c] += sub[c]
for c in classes:
	subs[0][c] /= len(subs)

subs[0].to_csv('Bagging.csv', index=False)