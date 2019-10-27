from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import copy

import DocumentFeatureVectorExtractionInterface as fe


class TFIDFVectorExtraction(fe.IFeatureVectorExtraction):
    def __init__(self, is_reshape_corpus=True):
        self.__is_reshape_coprpus = is_reshape_corpus
        self.__reshaped_corpus = None
        
    def createFeatureMatrix(self, corpus):
        vectorizer = TfidfVectorizer(min_df=0.0, max_df=1.0, ngram_range=(1,1))
        if self.__is_reshape_coprpus:
            self.__reshaped_corpus = TFIDFVectorExtraction.preprocessData(corpus)
        else:
            self.__reshaped_corpus = copy.deepcopy(corpus)
        #print(self.__reshaped_corpus[:3])
        feature_matrix = vectorizer.fit_transform(self.__reshaped_corpus).astype(float)
        dense_feature_matrix = feature_matrix.toarray()
        return (feature_matrix, dense_feature_matrix)
    
    def measureSimilarity(self, doc_vec_1, doc_vec_2):
        cosine_measure = np.dot(doc_vec_1, doc_vec_2)
        return cosine_measure
    
    @staticmethod
    def preprocessData(corpus):
        return [' '.join(x) for x in corpus]