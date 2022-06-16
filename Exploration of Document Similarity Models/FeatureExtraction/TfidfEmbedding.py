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
import FeatureExtraction.SentenceEmbeddingTypes as se
import FeatureExtraction.USESentenceEmbedding as us
import FeatureExtraction.TfidfFeatureExtraction as tf
import Utils as u
import NLPProcessing.NLPEngineComponent as nlp

logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)

class TFIDFStsCalculator(us.USEStsCalculator):
  """
  Class used to compute Semantic Text Similarity (STS) using TF-IDF
  """
  def __init__(
      self,
      corpus:  List[str],
      query_docs: List[str] = None,
      sentence_embedding_type: se.SentenceEmbeddingType = se.SentenceEmbeddingType.tfidf
  ):
    """
    Constructor
    """
    logging.info("Constructing 'TFIDFStsCalculator' model..")
    super().__init__(
        corpus,
        query_docs,
        sentence_embedding_type)
    self._all_corpus_embeddings = None


  def _embed(self, input: List[str]) -> Any:
    """
    Create the USE embedding of the input corpus
    :param input: Input corpus
    :return:
    """
    return self._model(docs=input).toarray()

  def computeSelfSimilarities(self) -> Dict[str, float]:
    """
    Computes the sentence embedding similarity metrics for query documents
    extracted from the corpus document (itself)
    :return: Similarity scores
    """
    print("Computing STS, this will take a couple of minutes..")
    doc_pair_indices = u.Helpers.createLowerTriangularMatrixOfPairs(len(self._clean_corpus))
    self._corpus_embeddings = self._embed(self._clean_corpus)
    sims = {}
    self._similarity_matrix_self = np.zeros((self._n_docs, self._n_docs))
    for i, j in doc_pair_indices:
        key = "{0}_{1}".format(i, j)
        sentence_tensor_i = self._corpus_embeddings[i]
        sentence_tensor_j = self._corpus_embeddings[j]
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
    self._all_corpus_embeddings = self._embed(self._clean_corpus + self._clean_query_docs)
    self._corpus_embeddings = self._all_corpus_embeddings[:self._n_docs]

    top_k = min(5, len(self._clean_corpus))
    for query_id, query in enumerate(self._clean_query_docs):
        query_embedding = self._all_corpus_embeddings[self._n_docs + query_id]

        # We use cosine-similarity and torch.topk to find the highest 5 scores
        cos_scores = util.cos_sim(query_embedding, self._corpus_embeddings)[0]
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

  @property
  def query_embeddings(self) -> List[np.ndarray]:
      """
      Getter property for sentence embeddings
      :return: Sentence embeddings
      """
      n_query_docs = len(self._clean_query_docs)
      query_embeddings = self._all_corpus_embeddings[self._n_docs:self._n_docs + n_query_docs]
      return query_embeddings

  @staticmethod
  def _embedStaticVersion(input: List[str]) -> Any:
    """
    Create the S-BERT embedding of the input corpus
    :param input: Input corpus
    :return:
    """
    model = tf.TfidfFeatureVectorizer()
    return model(input)

  @staticmethod
  def createEmbeddingForDocs(docs: List[str]) -> np.ndarray:
    """
    Creates an embedding tensor for documents/sentences
    :param docs: Documents being embedded
    :return:Embedded documents
    """
    clean_docs = us.USEStsCalculator._cleanCorpusStaticVersion(docs)
    docs_embeddings = TFIDFStsCalculator._embedStaticVersion(clean_docs)
    return docs_embeddings