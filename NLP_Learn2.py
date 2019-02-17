# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 20:15:26 2019

@author: Hillel Awaskar
"""
from __future__ import division
import nltk



## the text is scrapped and collected from WikiPedia 
Mauryan = open('D:\Data\CodePython\DataAI\Config\\Mauryanempire.txt', 'r',encoding="utf-8")
Mau =  Mauryan.read()
tokens = nltk.word_tokenize(Mau)
Mauryan_text = nltk.Text(tokens)



### finding text 
Mauryan_text.concordance("history") 
Mauryan_text.concordance("chanakya") 

Mauryan_text.dispersion_plot(["history","chanakya"])

Mauryan_text.similar("history")

Mauryan_text.common_contexts(["history"])

len(Mauryan_text)

sorted(set(Mauryan_text))

len(set(Mauryan_text))


len(Mauryan_text) / len(set(Mauryan_text))


def lexical_diversity(text): 
    return len(text) / len(set(text))

def percentage(count, total): 
    return 100 * count / total


lexical_diversity(Mauryan_text) 
percentage(Mauryan_text.count('history'), len(Mauryan_text)) 

Mauryan_text[2000:2005]

WordFreq = FreqDist(Mauryan_text) 

WordFreq
Vocabolary = WordFreq.keys()
WordFreq.plot(30, cumulative = True)

#############Learn 2

WordFreq.hapaxes()
Mauryan_text.collocations()


[len(word) for word in Mauryan_text]
   
 
freq_Dist = FreqDist([len(word) for word in Mauryan_text])
freq_Dist.keys()
freq_Dist.items()   


# most frequent word length 
freq_Dist.max()

freq_Dist[3]


# Conditionals
[w for w in Mauryan_text if len(w) > 4]
sorted([w for w in set(Mauryan_text) if w.endswith('yan')])
sorted([term for term in set(Mauryan_text) if 'ury' in term])
sorted([item for item in set(Mauryan_text) if item.istitle()])
sorted([item for item in set(Mauryan_text) if item.isdigit()])



## Cleaning the text for alpha words , generally done for unique words ; but words like "why?" will be ignored :)
len(Mauryan_text)
len(set(Mauryan_text))
len(set([word.lower() for word in Mauryan_text]))
len(set([word.lower() for word in Mauryan_text if word.isalpha()])) 





for token in Mauryan_text:
    if token.islower():
        print (token, 'is a lowercase word')
    elif token.istitle():
        print (token, 'is a titlecase word')
    else:
        print (token, 'is punctuations')










