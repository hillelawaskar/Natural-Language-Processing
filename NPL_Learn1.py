# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 00:37:51 2019

@author: Hillel Awaskar
"""
import nltk
from __future__ import division


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




