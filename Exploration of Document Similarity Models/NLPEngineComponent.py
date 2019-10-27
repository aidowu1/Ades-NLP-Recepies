#!/usr/bin/env python
# coding: utf-8

# ## Text Cleaning using NLP techniques
# Documents used for ML decision making need to be pre-processed i.e. cleaned. These steps include:
#     - Tokenization    
#     - Lowercasing
#     - Noise removal 
#     - Removing stop words 
#     - Text normalization
#     - Stemming
#     - Lemmatization    

# ### Tokenization
#     - Tokenization is the process of breaking down a corpus into individual tokens.

# ### Lowercasing
#     - Converts all the existing uppercase characters into lowercase ones so that the entire corpus is in lowercase

# ### Noise Removal
#     - Removing noise in the corpus such as puntuations, hashtags, html markup, non-ASCII characters 
#     - and other unwanted characters

# ### Removing Stop Words
#     - Removing standard/typical stop words from the corpus (i.e. such as those define in NLTK library)
#     - Removing user specified domain specific stop words

# ### Text Normalization
#     - This is the process of converting a raw corpus into a canonical and standard form, which is basically to ensure that 
#     the textual input is guaranteed to be consistent before it is analyzed, processed, and operated upon.

# ### Stemming
#     - Stemming is performed on a corpus to reduce words to their stem or root form. The reason for saying "stem or root form" is that the process of stemming doesn't always reduce the word to its root but sometimes just to its canonical form.

# ### Lemmatizer
#     - Lemmatization is a process that is like stemming – its purpose is to reduce a word to its root form. What makes it different is that it doesn't just chop the ends of words off to obtain this root form, but instead follows a process, abides by rules, and often uses WordNet for mappings to return words to their root forms.

# ## Imports

# In[62]:


from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import re
import string


# In[66]:


class NLPEngine(object):
        """
        Natural Language Processing engine used to pre-process a corpus
        """
        def __init__(self,docs, is_stemmimg=False, is_lemmentization=False):
            self.__docs = docs
            self.__is_stemming = is_stemmimg
            self.__is_lemmentization = is_lemmentization
            self.__stop_words = self.getStopWords()
            if self.__is_stemming: 
                self.__stemmer = PorterStemmer()
            else:
                self.__stemmer = None
            if self.__is_lemmentization: 
                self.__lemmatizer = WordNetLemmatizer()
            else:
                self.__lemmatizer = None
           
            
        def removeNoise(self, text):
            #remove html markup
            text = re.sub("(<.*?>)","",text)
            #remove non-ascii and digits
            text=re.sub("(\W|\d+)"," ",text)
            # prepare regex to filter punctuations
            re_punc = re.compile('[%s]' % re.escape(string.punctuation))
            text=re_punc.sub('', text)
            #remove whitespace
            text=text.strip()
            return text
        
        def normalize(self,text):
            normalize_dict = {
                brb: "be right back"
            }
            new_text = normalize_dict.get(text)
            if new_text is None:
                return text
            else:
                return new_text
            
        def getStopWords(self):
            stop_words = set(stopwords.words('english'))
            user_defined_stop_words = []
            all_stop_words = []
            all_stop_words.extend(stop_words)
            all_stop_words.extend(user_defined_stop_words)
            all_stop_words = list(set(all_stop_words))
            return all_stop_words
            
            
            
        def preprocessDoc(self, doc):
            # Tokenization
            tokens_0 = word_tokenize(doc)
            # Lowercasing/uppercasing
            tokens_1 = [w.lower() for w in tokens_0]
            # remove punctuation from each word
            tokens_2 = [self.removeNoise(w) for w in tokens_1]
            # remove remaining tokens that are not alphabetic
            tokens_3 = [word for word in tokens_2 if word.isalpha()]
            # Removing stop words
            tokens_4 = [w for w in tokens_3 if not w in self.__stop_words]
            if self.__is_stemming:
                tokens_5 =[self.__stemmer.stem(word = word) for word in tokens_4]
            else:
                tokens_5 = tokens_4
            if self.__is_lemmentization:
                tokens_6 =[self.__lemmatizer.lemmatize(word = word, pos = 'v') for word in tokens_5]
            else:
                tokens_6 = tokens_5
            return tokens_6
        
        def preprocessDocs(self):
            preprocess_docs = [self.preprocessDoc(doc) for doc in self.__docs]
            return preprocess_docs
                
            
            


# ### Test NLP Preprocessing Engine

# ### Define the test data:
#     - Test data was was used for a Kaggle competition it is the Quora pair dataset Quora dataset [further details here](https://www.kaggle.com/currie32/predicting-similarity-tfidfvectorizer-doc2vec/data)
#     

# In[67]:


df_corpus = pd.read_csv(r"Data\questions.csv")
df_sample_corpus = df_corpus[:100]
df_sample_corpus.head()


# In[68]:


def testNlpEngine():
    docs = df_sample_corpus.question1.tolist()
    nlp_engine = NLPEngine(docs)
    clean_docs = nlp_engine.preprocessDocs()
   
    
testNlpEngine()


# In[ ]:




