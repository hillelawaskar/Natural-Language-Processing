# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 18:01:20 2019

@author: Irene
"""

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
  

hotel_rev = ['Great place to be when you are in Bangalore.',
             'The place was being renovated when I visited so the seating was limited.',
             'Loved the ambience, loved the food',
             'The food is delicious but not over the top.',
             'Service - Little slow, probably because too many people.',
             'The place is not easy to locate',
             'Mushroom fried rice was tasty']
  
sid = SentimentIntensityAnalyzer()
for sentence in hotel_rev:
     print(sentence)
     ss = sid.polarity_scores(sentence)
     for k in ss:
         print('{0}: {1}, '.format(k, ss[k]), end='')
     print()
     
     
#########  Using Naive Bayes Classifier  
     
import nltk
from nltk.tokenize import word_tokenize
  
# Step 1 – Training data
train = [("Great place to be when you are in Bangalore.", "pos"),
  ("The place was being renovated when I visited so the seating was limited.", "neg"),
  ("Loved the ambience, loved the food", "pos"),
  ("The food is delicious but not over the top.", "neg"),
  ("Service - Little slow, probably because too many people.", "neg"),
  ("The place is not easy to locate", "neg"),
  ("Mushroom fried rice was spicy", "pos"),
]
  
# Step 2 

# loop for creating dictionary
#for passage in train :
#    print (passage[0])
#    
#    for word in word_tokenize(passage[0]):
#        a = word.lower()
#        print (a)
dictionary = set(word.lower() for passage in train for word in word_tokenize(passage[0]))
  


# Step 3
t = [({word: (word in word_tokenize(x[0].lower())) for word in dictionary}, x[1]) for x in train]



  
# Step 4 – the classifier is trained with sample data
classifier = nltk.NaiveBayesClassifier.train(t)
  
test_data = "The fried food is very good"
test_data_features = {word.lower(): (word in word_tokenize(test_data.lower())) for word in dictionary}
print (classifier.classify(test_data_features))










#######From Tweeter txt files for Positive and Negative tweets

import nltk

def format_sentence(sent):

    Text = nltk.word_tokenize(sent)
    Text = set([word.lower() for word in Text if word.isalpha()])
    print (Text)
    return({word: True for word in Text})




print(format_sentence("The cat is very cute"))


# Wchih produces
# {'The': True, 'cat': True, 'is': True, 'very': True, 'cute': True}

#######

pos = []
with open("D:\\Data\\CodePython\\ML Learning\\data\\pos_tweets.txt") as f:
    for i in f: 
        pos.append([format_sentence(i), 'pos'])

neg = []
with open("D:\\Data\\CodePython\\ML Learning\\data\\neg_tweets.txt") as f:
    for i in f: 
        neg.append([format_sentence(i), 'neg'])

# next, split labeled data into the training and test data
training = pos[:int((.8)*len(pos))] + neg[:int((.8)*len(neg))]
test = pos[int((.8)*len(pos)):] + neg[int((.8)*len(neg)):]

###########


from nltk.classify import NaiveBayesClassifier

classifier = NaiveBayesClassifier.train(training)

###########


classifier.show_most_informative_features()


##########

example1 = "Cats are awesome!"

print(classifier.classify(format_sentence(example1)))