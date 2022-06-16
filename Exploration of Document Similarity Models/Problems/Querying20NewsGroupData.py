"""
20-News groups simple search-engine problem
===========================================
    - The 20 newsgroups dataset comprises around 18000 newsgroups posts on 20 topics split in two subsets:
        one for training (or development) and the other one for testing (or for performance evaluation).
    - It is a popular benchmark dataset used demo NLP and document clustering/classification solutions
    - This program provides a simple demo of document search engine. The engine will leverage STS to match a search
      with top k (10) documents similar to the search query
    - The computed cosine similarity measures will be based on using embedding models such as:
        - Google's USE
        - S-Bert
    - The 20-News group dataset was sourced directly for Sci Kit Learn's sklearn.datasets module.
      Further details of the api can be found here:
        - https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html

"""

from sklearn.datasets import fetch_20newsgroups
import tensorflow as tf
import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import os
import operator
from typing import List, Tuple, Dict, Any, Iterable
import math
import logging

import FeatureExtraction.SentenceEmbeddingTypes as se
import FeatureExtraction.SBERTSentenceEmbedding as ss
import FeatureExtraction.USESentenceEmbedding as us
import NLPProcessing.NLPEngineComponent as nlp
import Constants as c
import Utils as u

logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)

class TwentyNewsGroupDemo(object):
    """
    20-News Group Problem demo
    """
    def __init__(
            self,
            n_docs: int = 100,
            sentence_embedding_type: se.SentenceEmbeddingType = se.SentenceEmbeddingType.use
    ):
        """
        Constructor
        :return:
        """
        logging.info("Starting 20-News Group Demo, it will take a couple of seconds to instantiate the "
                     "document embedding models..")
        self.__sentence_embedding_type = sentence_embedding_type
        if self.__sentence_embedding_type == se.SentenceEmbeddingType.use:
            self.__model_use = hub.load(c.USE_MODEL_URL)
        elif self.__sentence_embedding_type == se.SentenceEmbeddingType.sbert:
            self.__model_sbert = SentenceTransformer(c.SBERT_MODEL_CONFIG)
        self.__n_docs = n_docs
        self.__raw_corpus = self.__getRadomSampleOf20NewsGroupData()
        self.__clean_corpus = self.__cleanCorpus(self.__raw_corpus)


    def __getRadomSampleOf20NewsGroupData(self) -> List[str]:
        """
        Returns a specified sample of the 20 News Group data
        based on a sub-set of categories
        :param n_samples: Number of samples
        :return: Corpus of data
        """
        categories = ['alt.atheism', 'talk.religion.misc',
              'comp.graphics', 'sci.space']
        newsgroups_train = fetch_20newsgroups(
            subset='train',
            remove=('headers', 'footers', 'quotes'),
            categories=categories)
        sample_corpus = newsgroups_train.data[:self.__n_docs]
        return sample_corpus

    def __cleanCorpus(self, docs: List[str]) -> List[List[str]]:
        """
        Cleans/normalises the corpus
        """
        nlp_engine = nlp.NLPEngine(docs)
        clean_docs = nlp_engine.preprocessDocs()
        return clean_docs

    def __embedUse(self, docs: List[str]) -> np.ndarray:
        """
        Computes the USE embedding of the corpus
        :param docs: Corpus
        :return: Embedding
        """
        return self.__model_use(docs)

    def __embedSBert(self, docs: List[str]) -> np.ndarray:
        """
        Computes the SBert embedding of the corpus
        :param docs: Corpus
        :return: Embedding
        """
        return self.__model_sbert.encode(docs, convert_to_numpy=True)

    def __embed(self, docs: List[str]):
        """
        Computes the embedding of a corpus per sentence embedding type
        :param docs: Corpus
        :return: Embedding
        """
        if self.__sentence_embedding_type == se.SentenceEmbeddingType.use:
            return self.__embedUse(docs)
        elif self.__sentence_embedding_type == se.SentenceEmbeddingType.sbert:
            return self.__embedSBert(docs)

    def __computeSimilarityMetricsPerBatch(
            self,
            query_embedding: np.ndarray,
            corpus_embedding: np.ndarray,
            ) -> np.ndarray:
        """
        Computes the similarity metrics per batch
        :param batch: The batch iterator of docs
        :param: sentence_embedding_type: Sentence embedding type
        :return: Returns the similarity scores
        """
        logging.info(f"Processing the search engine for query: {query_doc} ")
        sts_encode1 = tf.nn.l2_normalize(query_embedding, axis=1)
        sts_encode2 = tf.nn.l2_normalize(corpus_embedding, axis=1)
        cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
        clip_cosine_similarities = tf.clip_by_value(cosine_similarities, -1.0, 1.0)
        scores = 1.0 - tf.acos(clip_cosine_similarities) / math.pi
        return scores.numpy()

    def runDocumentSearch(self, query_doc: str, top_k: int = 10) -> pd.DataFrame:
        """
        Runs the document query for a specified query document against the entire corpus
        and retrieves the top K documents that are similar to the query document
        :param query_doc: Query document
        :param top_k: Top K
        :return: Table of results
        """
        clean_query_sentence = [" ".join(s) for s in self.__cleanCorpus([query_doc])]
        clean_sentences = [" ".join(s) for s in self.__clean_corpus]
        corpus_embedding = self.__embed(clean_sentences)
        query_embedding = self.__embed(clean_query_sentence)
        scores = self.__computeSimilarityMetricsPerBatch(query_embedding=query_embedding, corpus_embedding=corpus_embedding)
        sim_scores_rounded = [round(s, 4) for s in scores]
        corpus_docs_ids = [x for x in range(self.__n_docs)]
        corpus_docs = [" ".join(s) for s in self.__clean_corpus]
        results_df = pd.DataFrame({
            "doc_id": corpus_docs_ids,
            "documents": corpus_docs,
            "scores": sim_scores_rounded
        })
        results_df.sort_values(by=['scores'], ascending=False, inplace=True)
        return results_df.head(top_k)


    @property
    def corpus(self) -> List[str]:
        """
        getter for the 20-News Group sample data
        :return: 20 News Group sample data
        """
        return self.__raw_corpus




if __name__ == "__main__":
    problem = TwentyNewsGroupDemo(sentence_embedding_type=se.SentenceEmbeddingType.sbert)
    data = problem.corpus
    query_doc = "Computer programming algorithms"
    top_k = 10
    results_df = problem.runDocumentSearch(query_doc=query_doc, top_k=top_k)
    logging.info(f"The results of the top {top_k} documents that match the query are:\n\n")
    print(f"Query doc is: {query_doc}\n")
    print(u.Helpers.tableize(results_df))
    #print(results_df)
    print("")

