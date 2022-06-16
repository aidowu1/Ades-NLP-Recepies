from sentence_transformers import SentenceTransformer, util
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
from typing import List, Dict, Tuple, Optional, Any
import enum
import logging

import Constants as c
import Utils as u
import NLPProcessing.NLPEngineComponent as nlp
import FeatureExtraction.SentenceEmbeddingTypes as se
import FeatureExtraction.TfidfFeatureExtraction as tf

logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)

class USEStsCalculator(object):
  """
  Class used to compute Semantic Text Similarity (STS) using Google's USE
  """
  def __init__(
      self,
      corpus:  List[str],
      query_docs: List[str] = None,
      sentence_embedding_type: se.SentenceEmbeddingType = se.SentenceEmbeddingType.use
      ):
    """
    Constructor
    """
    print("Constructing 'StsCalculatorUse' model..")
    self.__raw_corpus = corpus
    self.__raw_query_docs = query_docs
    self._n_docs = len(self.__raw_corpus)
    self._clean_query_docs = None
    if self.__raw_query_docs:
        self._clean_query_docs = self.__cleanCorpus(self.__raw_query_docs)
    self._clean_corpus = self.__cleanCorpus(self.__raw_corpus)
    if sentence_embedding_type == se.SentenceEmbeddingType.use:
        self._model = hub.load(c.USE_MODEL_URL)
    elif sentence_embedding_type == se.SentenceEmbeddingType.sbert:
        self._model = SentenceTransformer(c.SBERT_MODEL_CONFIG)
    elif sentence_embedding_type == se.SentenceEmbeddingType.tfidf:
        self._model = tf.TfidfFeatureVectorizer()
    self._corpus_embeddings = None
    self._similarity_matrix_self = None
    self._results = {}
    self._results_df = None
    self._sentence_ids = []
    self._query_ids = []
    self._corpus_sentences = []
    self._query_sentences = []
    self._scores = []

  def _embed(self, input: List[str]) -> np.ndarray:
    """
    Create the USE embedding of the input corpus
    :param input: Input corpus
    :return:
    """
    return np.array(self._model(input))

  def computeSelfSimilarities(self) -> Dict[str, float]:
    """
    Computes the sentence embedding similarity metrics for query documents
    extracted from the corpus document (itself)
    :return: Similarity scores
    """
    logging.info("Computing STS, this will take a couple of minutes..")
    doc_pair_indices = u.Helpers.createLowerTriangularMatrixOfPairs(len(self._clean_corpus))
    self._corpus_embeddings = self._embed(self._clean_corpus)
    corpus_sentence_tensor = torch.from_numpy(self._corpus_embeddings)
    sims = {}
    self._similarity_matrix_self = np.zeros((self._n_docs, self._n_docs))
    for i, j in doc_pair_indices:
        key = "{0}_{1}".format(i, j)
        sentence_tensor_i = corpus_sentence_tensor[i]
        sentence_tensor_j = corpus_sentence_tensor[j]
        cosine_score = util.cos_sim(sentence_tensor_i, sentence_tensor_j)[0]
        sims[key] = round(cosine_score.numpy()[0], 4)
        self._similarity_matrix_self[i, j] = sims[key]
    return sims

  def computeSimilarities(self, top_k: int = 5) -> pd.DataFrame:
    """
    Execute the STS calculation
    :param top_k: Number of top K sentences
    """

    logging.info("Computing STS, this will take a couple of minutes..")
    self._corpus_embeddings = self._embed(self._clean_corpus)
    top_k = min(5, len(self._clean_corpus))
    for query_id, query in enumerate(self._clean_query_docs):
        query_embedding = self._embed([query])

        # We use cosine-similarity and torch.topk to find the highest 5 scores
        query_embedding_tensor = torch.from_numpy(query_embedding)
        corpus_embeddings_tensor = torch.from_numpy(self._corpus_embeddings)
        cos_scores = util.cos_sim(query_embedding_tensor, corpus_embeddings_tensor)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        for score, idx in zip(top_results[0], top_results[1]):
          self._query_ids.append(query_id + self._n_docs)
          self._sentence_ids.append(idx.numpy())
          self._corpus_sentences.append(self._clean_corpus[idx])
          self._query_sentences.append(query)
          self._scores.append(score.numpy())
    self._results = {
        "query_sentence_id": self._query_ids,
        "corpus_sentence_id": self._sentence_ids,
        "corpus_sentences": self._corpus_sentences,
        "query_sentences": self._query_sentences,
        "scores": self._scores
        }
    self._results_df = pd.DataFrame(data=self._results)
    self._results_df.sort_values(by=['scores'], ascending=False, inplace=True)
    return self._results_df

  def __cleanCorpus(self, raw_docs: List[str]) -> Optional[List[str]]:
      """
      Applies NLP pre-processing to the raw corpus
      :param raw_docs: Documents/sentences
      :return: Cleans documents/sentences
      """
      return USEStsCalculator._cleanCorpusStaticVersion(raw_docs=raw_docs)

  @property
  def results_df(self) -> pd.DataFrame:
    """
    Getter property for the STS results table
    :return: Results table
    """
    return self._results_df

  @property
  def similarity_matrix_self(self) -> np.ndarray:
      """
      Getter property for the similarity matrix (i.e. self-similarity measure)
      :return: Similarity matrix
      """
      return self._similarity_matrix_self

  @property
  def similarity_matrix(self) -> np.ndarray:
      """
      Getter property for the similarity matrix (non-self-similarity measure)
      :return: Similarity matrix
      """
      similarity_matrix_cols = ["query_sentence_id", "corpus_sentence_id", "scores"]
      score_df = self._results_df[similarity_matrix_cols]
      score_df.query_sentence_id = pd.array(score_df.query_sentence_id.tolist())
      score_df.corpus_sentence_id = pd.array(score_df.corpus_sentence_id.tolist())
      score_df.scores = pd.array(score_df.scores.tolist())
      similarity_matrix = score_df.pivot(
          index="query_sentence_id",
          columns="corpus_sentence_id",
          values="scores"
      ).to_numpy()
      return similarity_matrix

  @property
  def corpus_embeddings(self) -> List[np.ndarray]:
      """
      Getter property for corpus embeddings
      :return: Sentence embeddings
      """
      return self._corpus_embeddings

  @property
  def query_embeddings(self) -> List[np.ndarray]:
      """
      Getter property for sentence embeddings
      :return: Sentence embeddings
      """
      query_embeddings = self._embed(self._clean_query_docs)
      return query_embeddings

  @staticmethod
  def _cleanCorpusStaticVersion(raw_docs: List[str]) -> Optional[List[str]]:
      """
      Applies NLP pre-processing to the raw corpus
      :param raw_docs: Documents/sentences
      :return: Cleans documents/sentences
      """
      if raw_docs:
          nlp_engine = nlp.NLPEngine(raw_docs)
          preprocessed_docs = nlp_engine.preprocessDocs()
          joined_preprocessed_docs = [" ".join(s) for s in preprocessed_docs]
          return joined_preprocessed_docs
      else:
          return None

  @staticmethod
  def _embedStaticVersion(input: List[str]) -> Any:
    """
    Create the USE embedding of the input corpus
    :param input: Input corpus
    :return:
    """
    model = hub.load(c.USE_MODEL_URL)
    return np.array(model(input))

  @staticmethod
  def createEmbeddingForDocs(docs: List[str]) -> np.ndarray:
      """
      Creates an embedding tensor for documents/sentences
      :param docs: Documents being embedded
      :return:Embedded documents
      """
      clean_docs = USEStsCalculator._cleanCorpusStaticVersion(docs)
      docs_embeddings = USEStsCalculator._embedStaticVersion(clean_docs)
      return docs_embeddings


