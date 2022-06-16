from sentence_transformers import SentenceTransformer, util
import logging
import torch
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import pandas as pd
import math
from typing import List, Dict, Tuple, Optional

import FeatureExtraction.SentenceEmbeddingTypes as se
import NLPProcessing.NLPEngineComponent as nlp
import Constants as c
import Utils as u

logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)

class SentenceEmbeddingUSE(object):
    """
    Component used to create the sentence embedding using Google's Universal Sentence Encoder (USE)
    """
    def __init__(
            self,
            corpus: List[str],
            query_docs: List[str]
    ):
        """
        Constructor
        :param corpus:
        """
        try:
            self.__raw_corpus = corpus
            self.__raw_query_docs = query_docs
            self.__n_docs = len(self.__raw_corpus)
            self.__clean_corpus = self.__cleanCorpus(self.__raw_corpus)
            self.__clean_query_docs = self.__cleanCorpus(self.__raw_query_docs)
            self.__sentences = None
            self.__corpus_sentence_embeddings = None
            self.__query_sentence_embeddings = None
            self.__model = None
            self.__sim_scores = None
        except Exception as ex:
            logging.info(f"Incorrect sentence embedding type: '{sentence_embedding_type.name}' specified!!")
            logging.info(f"Error: {str(ex)}")

    def __cleanCorpus(self, raw_docs: List[str]) -> Optional[List[str]]:
        """
        Applies NLP pre-processing to the raw corpus
        :param: raw_docs: Raw documents/sentences
        :return: Cleans documents/sentences
        """
        if raw_docs:
            nlp_engine = nlp.NLPEngine(raw_docs)
            preprocessed_docs = nlp_engine.preprocessDocs()
            joined_preprocessed_docs = [" ".join(s) for s in preprocessed_docs]
            return joined_preprocessed_docs
        else:
            return None

    def __embed(self, input: List[str]) -> np.ndarray:
        """
        Create the USE embedding of the input corpus
        :param input: Input corpus
        :return:
        """
        return np.array(self.__model(input))

    def __trainModel(self):
        """
        Apply SIF using the FSE library to train the sentence embedding model
        """
        self.__corpus_sentence_embeddings = self.__embed(self.__clean_corpus)
        if self.__clean_query_docs:
            self.__query_sentence_embeddings = self.__embed((self.__clean_query_docs))

    def __computeSelfSimilarities(self, doc_pair_indices: Tuple[int, int]) -> Dict[str, float]:
        """
        Computes the sentence embedding similarity metrics for query documents
        extracted from the corpus document (itself)
        :param doc_pair_indices: sentence pairs (i.e. b/w query and corpus sentences)
        :return: Similarity scores
        """
        sims = {}
        for i, j in doc_pair_indices:
            key = "{0}_{1}".format(i, j)
            corpus_sentence_tensor = torch.from_numpy(self.__corpus_sentence_embeddings)
            query_sentence_tensor = torch.from_numpy(self.__corpus_sentence_embeddings[i])
            cosine_scores = util.cos_sim(query_sentence_tensor, corpus_sentence_tensor)[0]
            sims[key] = round(cosine_scores, 4)
        return sims

    def computeSimilarities(self) -> Dict[str, float]:
        """
        Computes the sentence similarities
        :return:
        """
        self.__trainModel()
        sentence_pairs = u.Helpers.createLowerTriangularMatrixOfPairs(n_sentences=self.__n_docs)
        self.__sim_scores = self.__computeSelfSimilarities(sentence_pairs)
        return self.__sim_scores

    @property
    def sentence_embeddings(self) -> List[np.ndarray]:
        """
        Getter property for sentence embeddings
        :return: Sentence embeddings
        """
        return self.__corpus_sentence_embeddings

    @property
    def similarity_scores(self) -> List[float]:
        """
        Getter property for sentence similarity scores
        :return: Similarity scores
        """
        return self.__sim_scores


