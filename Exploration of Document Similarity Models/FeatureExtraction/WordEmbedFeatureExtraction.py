import numpy as np
from typing import List, Dict, Tuple
from scipy.sparse import csr_matrix
import tqdm

import FeatureExtraction.FeatureExtractionInterface as ie
import FeatureExtraction.FeatureVectorizerEnums as fe
import FeatureExtraction.Constants as c

class WordEmbeddingVector(ie.VectorizerInterface):
    """
    Word embedding computation of document tokens
    The supported word embedding strategies are:
        - GLoVe (Stanford University)
        - Fasttext (Facebook)
        - Word2Vec (Google)
    """
    def __init__(self, embedding_type: fe.FeatureVectorizerType = fe.FeatureVectorizerType.glove):
        """
        Constructor
        :param embedding_type: Embedding type
        """
        self.__embedding_type = embedding_type
        self.__embedding_map, self.__embedding_dim = self.__loadEmbeddingVectors()
        self.__vectors = None
        print(f"Vectorization approach is: {self.__embedding_type}")

    def __call__(self, docs: List[str]) -> csr_matrix:
        """
        Converts the words (tokens) into numeric vectors
        For a collection of words (tokens) in a sentence it will aggregate (i.e. compute the mean) value of each
        word vector per sentence
        :param docs: Input list of documents that are going to be transformed into a vector of vector of floats
        :return: Compressed Sparse Row (CSR) representing vectors per document
        """
        self.__vectors = self.transform(docs)
        return self.__vectors

    def transform(self, docs: List[str]) -> csr_matrix:
        """
        Transforms the document to a list of vectors per document
        :param docs: Input list of documents that are going to be transformed into a vector of vector of floats
        :return: Compressed Sparse Row (CSR) representing vectors per document
        """
        embedding_vectors = csr_matrix(np.array([self.__sentence2vector(s) for s in docs]))
        return embedding_vectors

    @property
    def vectorizer(self) -> object:
        """
        Getter for the vectorizer
        :return: TF-IDF vectorizer
        """
        return self

    @property
    def vectors(self) -> csr_matrix:
        """
        Getter for the 2-D array of vectors
        :return: Array of vectors
        """
        return self.__vectors

    def __sentence2vector(self, sentence: str) -> np.ndarray:
        """
        Converts the sentence (collection of tokens) to a vector of floats.
        the vector is an aggregate (mean) of all the individual token vectors that make-up the sentence
        :param sentence: Input sentence
        :return: Numeric vector representing the sentence
        """
        embedding_vector_list = []
        for w in sentence:
            if w in self.__embedding_map:
                embedding_vector_list.append(self.__embedding_map[w])
        if len(embedding_vector_list) == 0:
            return np.zeros(self.__embedding_dim)
        embedding_matrix = np.array(embedding_vector_list)
        embedding_aggregate_vector = embedding_matrix.sum(axis=0)
        embedding_aggregate_vector = embedding_aggregate_vector / np.sqrt((embedding_aggregate_vector ** 2).sum())
        return embedding_aggregate_vector


    def __loadEmbeddingVectors(self) -> Tuple[Dict[str, List[float]], int]:
        """
        Loads the pre-trained embedding vectors from disk as a dictionary of words (keys) and vectors (value)
        Reference URL: https://en.wikipedia.org/wiki/Word2vec
        :return: Word embedding object which is a tuple of embedding dictionary and its dimension
        """
        embedding_map = {}
        embedding_file_path = c.WORD_EMBEDDING_TYPE[self.__embedding_type]
        with open(
            embedding_file_path,
            mode='r',
            encoding=c.EMBEDDING_ENCODING_TYPE,
            newline="\n",
            errors="ignore"
        ) as fin:
            if embedding_file_path.endswith(".vec"):
                n_samples, dim = map(int, fin.readline().split())
            for line in tqdm.tqdm(fin):
                tokens = line.rstrip().split(" ")
                embedding_map[tokens[0]] = list(map(float, tokens[1:]))
            n_samples = len(embedding_map.values())
            dim = len(list(embedding_map.values())[0])
            print(f"The {self.__embedding_type} embedding with {n_samples} word/token samples "
                  f"and dimension: {dim} has been loaded..")
            return embedding_map, dim