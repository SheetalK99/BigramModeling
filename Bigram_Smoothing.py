
#Author:Sheetal Kadam
#Date: 09/22/2019
#Bigram Probabilities (40 points):

import sys

filename=sys.argv[1]
with open(filename, 'r') as file:
    data = file.read()
	


#calcualte unuigrams
unigrams = dict()
for i in data.split():
    
    word=i.split('_')[0].lower()
    unigrams[word] = unigrams.get(word, 0) + 1
    
    
#split data into sentences

sentences=data.split('\n')


#calcualte bigrams

from nltk import FreqDist
from nltk.util import ngrams  
import string

def calculate_freq(sentences):
        
    bigramfdist = FreqDist()

    for line in sentences:
        
        if len(line) > 1:
            tokens = line.strip().lower().split()
            tokens_wo_tag=[word.split('_')[0] for word in tokens]
          #  x = [''.join(c for c in s if c not in string.punctuation) for s in tokens_wo_tag]
           # x = [s for s in x if s]

            bigrams = ngrams(tokens_wo_tag, 2)
            bigramfdist.update(bigrams)
    return bigramfdist
    
    
bigramfdist=calculate_freq(sentences)


# calculate co-occurence for all possible token pairs

bigram_corpus={}
for t1 in unigrams:
    for t2 in unigrams:
        if (t1,t2) in bigramfdist.keys():
            bigram_corpus[(t1,t2)]=bigramfdist[(t1,t2)]
        else:
            bigram_corpus[(t1,t2)]=0



#######Bigram Probability


No_smoothing_prob={}
Add_one_prob={}
good_turing_prob={}
N={}
c_star={}
#  non smoothing 

for gram in bigram_corpus.keys():
    No_smoothing_prob[gram]=(bigram_corpus[gram]/unigrams[gram[0]])
    
    
##  Add One Smoothing

count_Add={}
total_tokens=len(unigrams)
for keys, values in bigram_corpus.items():
    word=keys[0]
    den=unigrams[word]+total_tokens
    Add_one_prob[keys]= (values+1)/(den)
    count_Add[keys]=((values+1)*unigrams[word])/(den)

## Good turing

for gram in bigram_corpus.keys():
    if bigram_corpus[gram] not in N:
        N[bigram_corpus[gram]]=1

    else:
        N[bigram_corpus[gram]]= N[bigram_corpus[gram]]+1

#for keys, values in count_Add.items():
for key,val in bigram_corpus.items():
    if val==0: #c* for N0=N1
         c_star[key]=(N[1])
        
    elif val+1 in N:
        c_star[key]=((val+1)*N[val+1])/N[val]
    else:    # if #set the Good Turing smoothed count C*_(c)=0 if N_(c+1)=0
        c_star[key]=0
        
        
total_bigrams=sum(bigram_corpus.values())


for key, val in c_star.items():
    good_turing_prob[key]=val/total_bigrams



#######Input sentence


input='the standard turbo engine is hard to work'
import nltk
import numpy as np
bigrm = list(nltk.bigrams(input.split()))

print('No smoothing:')
for big in bigrm:
    print(big,No_smoothing_prob[big])
    
print('Add one smoothing:')

sum_prob=0
for big in bigrm:
    print(big,(Add_one_prob[big]))
    sum_prob=sum_prob+np.log(Add_one_prob[big])
print((sum_prob))
print(np.exp(sum_prob))

print('Good turing:')

sum_prob=0
for big in bigrm:
    print(big,(good_turing_prob[big]))
    sum_prob=sum_prob+np.log(good_turing_prob[big])
print(np.exp(sum_prob))