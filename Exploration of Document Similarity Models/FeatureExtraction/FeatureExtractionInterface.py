import abc
from abc import ABCMeta, abstractmethod
from typing import List, Any
from scipy.sparse import csr_matrix

class VectorizerInterface(abc.ABC):
    """
    Interface specification for feature extraction
    """

    @abstractmethod
    def transform(self, docs: List[str]) -> csr_matrix:
        """
        Transforms the document to a list of vectors which represent the document encoding
        :param docs: Input list of documents which are going to be transformed into a vector array of floats
        :return: Compressed Sparse Row CSR) matrix, representing the vectors per matrix
        """
        pass

    @property
    @abstractmethod
    def vectorizer(self) -> object:
        """
        Getter property of the vectorizer
        :return: vectorizer object
        """

    @property
    @abstractmethod
    def vectors(selfs) -> csr_matrix:
        """
        Getter for the 2-D array of vectors
        :return: Array of vectors
        """
        pass