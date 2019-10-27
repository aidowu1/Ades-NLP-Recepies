# Ades-NLP-Recepies
__Exploration of Document Similarity Models

_Abstract_: 
The need to develop robust document/text similarity measure solutions is an essential step for building applications such as Recommendation Systems, Search Engines, Information Retrieval Systems including other ML/AI applications such as News Aggregators  (that require document clustering) or automated recruitment systems used to match CVs to job specification and so on. In general, text similarity is the measure of how words/tokens, tweets, phrases, sentences, paragraphs and entire documents are lexically and semantically close to each other. Texts/words are lexically similar if they have similar character sequence or structure and, are semantically similar if they have the same meaning, describe similar concepts and they are used in the same context.  
This work will demonstrate a number of strategies for feature extraction i.e. transforming documents to numeric feature vectors. This transformation step is a prerequisite for computing the similarity between documents. Typically each strategy will involve 3 steps, namely: 1) the use of standard natural language pre-processing techniques to prepare/clean the documents, 2) the transformation of the document text into a vector of vectors. 3), calculation of document feature vectors using the Cosine Similarity metric.

Strategies and associated ML/NLP libraries that will be presented are:
  - document pre-processing using NLTK library
  - feature extraction using Term frequency – Inverse Document Frequency (TF-IDF) with the aid of the Scikit-Learn library
  - feature extraction using a pre-trained GloVe embedding with the aid of the Spacy NLP Library
  - feature extraction using a trained Word2Vec embedding from scratch with the aid of the Gensim NLP library
  - feature extraction using a trained Doc2Vec embedding from scratch also with the aid of the Gensim library

