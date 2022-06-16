
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List
from scipy.sparse import csr_matrix

import FeatureExtraction.FeatureExtractionInterface as ie
import FeatureExtraction.Constants as c

class TfidfFeatureVectorizer(ie.VectorizerInterface):
    """
    Term Frequency Inverse Document Frequency feature vectorization model
    It is used to convert the text (tokens) of each document into a vector of floats
    """
    def __init__(self):
        """
        Constructor
        """
        self.__vectorizer = None
        self.__vectors = None

    def __call__(self, docs: List[str]) -> csr_matrix:
        """
        Learns vocabulary and inverse document frequency and converts input documents into a list of vectors.
        Each vector per document
        The returned result is also known as a document-term matrix
        :param docs: Input list of documents that are going to be transformed into a vector of vector of floats
        :return: Compressed Sparse Row (CSR) representing vectors per document
        """
        self.__vectorizer = TfidfVectorizer(
            max_features=c.TFIDF_CONSTANTS.n_features,
            stop_words=c.TFIDF_CONSTANTS.stop_words,
            use_idf=c.TFIDF_CONSTANTS.use_idf
        )
        self.__vectors = self.__vectorizer.fit_transform(docs)
        return self.__vectors

    def transform(self, docs: List[str]) -> csr_matrix:
        """
        Transforms the document to a list of vectors which represent the document encoding
        :param docs: Input list of documents which are going to be transformed into a vector array of floats
        :return: Compressed Sparse Row CSR) matrix, representing the vectors per matrix
        """
        vectors = self.__vectorizer.transform(docs)
        return vectors

    @property
    def vectorizer(self) -> object:
        """
        Getter for the TF-IDF vectorizer
        :return: TF-IDF vectorizer
        """
        return self.__vectorizer

    @property
    def vectors(self) -> csr_matrix:
        """
        Getter for the 2-D array of vectors
        :return: Array of vectors
        """
        return self.__vectors
