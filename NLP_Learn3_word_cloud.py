# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 23:29:15 2019

@author: Hillel Awaskar
"""

import wikipedia, os
from wordcloud import WordCloud, STOPWORDS
curr_path = "C:\\Users\\Irene\\Natural-Language-Processing\\"

stopwords = set(STOPWORDS)

def get_wiki_content(query):
    title = wikipedia.search(query)[0]
    page = wikipedia.page(title)
    return page.content

def beautiful_wordcloud(text):
    sw = set(STOPWORDS)
    wc = WordCloud(background_color="white", 
                   max_words = 200,
                   max_font_size=40, 
                   scale=3,
                   stopwords = sw)
    WordCloud()
    wc.generate(text)
    wc.to_file(os.path.join(curr_path, "word_cld.png"))
    

beautiful_wordcloud(get_wiki_content("Mauryan Empire"))

