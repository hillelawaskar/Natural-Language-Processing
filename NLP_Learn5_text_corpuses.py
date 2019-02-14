# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 08:01:42 2019

@author: Irene
C:\Users\Irene\AppData\Roaming\nltk_data\corpora\reuters.zip\reuters

"""

import nltk

nltk.corpus.gutenberg.fileids()

guttxt = nltk.corpus.gutenberg.words('whitman-leaves.txt') 
len(guttxt)



# orher way to import 
from nltk.corpus import gutenberg
gutenberg.fileids()
guttxt=gutenberg.words('whitman-leaves.txt')


for fileid in gutenberg.fileids():
    num_chars = len(gutenberg.raw("austen-emma.txt"))
    num_words = len(gutenberg.words("austen-emma.txt")) 
    num_sents = len(gutenberg.sents("austen-emma.txt"))
    
    num_vocab = len(set([w.lower for w in gutenberg.words("austen-emma.txt")]))
    print ( int(num_chars/num_words ), int(num_words/num_sents), int(num_words/num_vocab), "austen-emma.txt")
    
    


macbeth_sentences = gutenberg.sents('shakespeare-macbeth.txt')
macbeth_sentences
macbeth_sentences[1037]
longest_len = max([len(s) for s in macbeth_sentences])
[s for s in macbeth_sentences if len(s) == longest_len]



from nltk.corpus import webtext
for fileid in webtext.fileids():
    print (fileid)
    
    
from nltk.corpus import nps_chat
for fileid in nps_chat.fileids():
    print (fileid)


from nltk.corpus import brown
for fileid in brown.fileids():
    print (fileid)

############tabululating modals
brown.categories()
brown.words(categories='news')

news_text = brown.words(categories='news') 
fdist = nltk.FreqDist([w.lower() for w in news_text]) 
modals = ['can', 'could', 'may', 'might', 'must', 'will'] 
for m in modals: 
    print (m + ':', fdist[m],)

############tabululating modals / genres

cfd = nltk.ConditionalFreqDist(
        (genre, word)
        for genre in brown.categories()
        for word in brown.words(categories=genre)) 
genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor'] 
modals = ['can', 'could', 'may', 'might', 'must', 'will'] 
cfd.tabulate(conditions=genres, samples=modals)

cfd.items()

#                  can could   may might  must  will 
#           news    93    86    66    38    50   389 
#       religion    82    59    78    12    54    71 
#        hobbies   268    58   131    22    83   264 
#science_fiction    16    49     4    12     8    16 
#        romance    74   193    11    51    45    43 
#          humor    16    30     8     8     9    13 


from nltk.corpus import reuters 

reuters.fileids()

reuters.categories()


reuters.categories('training/9865')
reuters.fileids('barley')
reuters.fileids(['barley', 'corn'])
reuters.words('training/9865')[:14]
reuters.words(['training/9865', 'training/9880'])
reuters.words(categories='barley')
reuters.words(categories=['barley', 'corn'])


### Inaugral Corpous
from nltk.corpus import inaugural
inaugural.fileids()
[fileid[:4] for fileid in inaugural.fileids()]

cfd = nltk.ConditionalFreqDist(
        (target, fileid[:4])
        for fileid in inaugural.fileids()
        for w in inaugural.words(fileid)
        for target in ['america', 'citizen']
        if w.lower().startswith(target))
    
cfd.plot()

