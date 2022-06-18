"""

STS Benchmark problem
=====================
    - STS Benchmark comprises a selection of the English datasets used in the STS (Sematic text Similarity) tasks.
    - This program provides a simple demo of the evaluation of the similarity between sentence pairs. The sentence pairs
      are computed using embedding models such as:
        - Google's USE
        - S-Bert
      The computed similarities are then compared to the actual similarity score using Pearson Correlation metric
    - The STS sentence pair  data used in this demo was sourced from here:
      http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz
    - This demo was inspired by the tutorial provided by Google on USE that can be sourced here:
      https://www.tensorflow.org/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder
    - Further details on the STS benchmarks can be found here: https://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark

"""
import pathlib as p
import pandas as pd
import logging
import scipy
import math
import tensorflow as tf
import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer, util, evaluation
import numpy as np
import os
import re
import seaborn as sns
from typing import Tuple, List, Iterable, Any
from sklearn.metrics.pairwise import paired_cosine_distances

import Constants as c
import FeatureExtraction.SentenceEmbeddingTypes as se
import NLPProcessing.NLPEngineComponent as nlp
import Utils as u
import FeatureExtraction.TfidfEmbedding as tfidf
import FeatureExtraction.SentenceEmbedding as embed

logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)

class STSbenchmarkProblemDemo():
    """
    Demo of the STS benchmark problem - sentence pair similarity
    """
    def __init__(self):
        """
        Constructor
        """
        logging.info("Starting the STS Benchmark Problem demo!!")
        self.current_path = p.Path.cwd().parent
        logging.info(f"Current path is: {self.current_path}...{c.NEW_LINE}")
        logging.info("Constructing sentence embedding models of USE and S-BERT..")
        os.chdir(self.current_path)
        self.corpus_df = pd.read_csv(r"Data\sts_dev_df.csv")
        self.corpus_df["clean_sent_1"] = self.__cleanCorpus(self.corpus_df.sent_1.tolist())
        self.corpus_df["clean_sent_2"] = self.__cleanCorpus(self.corpus_df.sent_2.tolist())
        self.corpus_df["norm_sim"] = self.corpus_df.sim / c.STS_LABEL_MAX_VAL
        self.__model_use = hub.load(c.USE_MODEL_URL)
        self.__model_sbert = SentenceTransformer(c.SBERT_MODEL_CONFIG_1)
        self.__sbert_batch_size = 16
        self.show_progress_bar = True

    def __cleanCorpus(self, docs: List[str]) -> List[str]:
        """
        Cleans/normalises the corpus
        """
        nlp_engine = nlp.NLPEngine(docs)
        clean_docs = [" ".join(s) for s in nlp_engine.preprocessDocs()]
        return clean_docs

    def __embedUse(self, docs: List[str]) -> np.ndarray:
        """
        Computes the USE embedding of the corpus
        :param docs: Corpus
        :return: Embedding
        """
        return self.__model_use(docs)

    def __computeSimilarityMetricsPerBatchForUse(self, batch: Iterable, embedding_type: se.SentenceEmbeddingType):
        """
        Computes the similarity metrics per batch
        :param batch: The batch iterator of docs
        :param: sentence_embedding_type: Sentence embedding type
        :return: Returns the similarity scores
        """
        sts_embedding_1 = self.__embedUse(tf.constant(batch['clean_sent_1'].tolist()))
        sts_embedding_2 = self.__embedUse(tf.constant(batch['clean_sent_2'].tolist()))
        cosine_scores = 1 - paired_cosine_distances(sts_embedding_1, sts_embedding_2)
        return cosine_scores

    def __computeSimilarityMetrics(
            self,
            sentence_embedding_type: se.SentenceEmbeddingType) -> List[float]:
        """
        Computes the similarity metrics
        :param sentence_embedding_type: Sentence embedding type
        """
        if sentence_embedding_type == se.SentenceEmbeddingType.use:
            return self.__computeSimilarityMatrixForUse(sentence_embedding_type)
        elif sentence_embedding_type == se.SentenceEmbeddingType.sbert:
            return self.__computeSimilarityMatrixForSbert()
        elif sentence_embedding_type == se.SentenceEmbeddingType.tfidf:
            return self.__computeSimilarityMatrixForTfidf()
        elif (sentence_embedding_type == se.SentenceEmbeddingType.average_word_embedding or
                sentence_embedding_type == se.SentenceEmbeddingType.sif_word_embedding):
            return self.__computeSimilarityMatrixForWordEmbed(sentence_embedding_type)


    def __computeSimilarityMatrixForSbert(self) -> List[float]:
        """
        Compute the similarity for Sbert embedding
        :return: Cosine scores
        """
        sentences_1 = self.corpus_df["clean_sent_1"].tolist()
        sentences_2 = self.corpus_df["clean_sent_2"].tolist()
        embeddings1 = self.__model_sbert.encode(sentences_1, batch_size=self.__sbert_batch_size,
                                   show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
        embeddings2 = self.__model_sbert.encode(sentences_2, batch_size=self.__sbert_batch_size,
                                   show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        return cosine_scores

    def __computeSimilarityMatrixForTfidf(self) -> List[float]:
        """
        Compute the similarity for TF-IDF embedding
        :return: Cosine scores
        """
        sentences_1 = self.corpus_df["clean_sent_1"].tolist()
        sentences_2 = self.corpus_df["clean_sent_2"].tolist()
        all_sentences = sentences_1 + sentences_2
        n_all_sentences = len(all_sentences)
        n_sentences_1 = len(sentences_1)
        all_embeddings = tfidf.TFIDFStsCalculator.createEmbeddingForDocs(all_sentences)
        embeddings1 = all_embeddings[:n_sentences_1]
        embeddings2 = all_embeddings[n_sentences_1:n_all_sentences]
        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        return cosine_scores

    def __computeSimilarityMatrixForWordEmbed(
            self,
            sentence_embedding_type: se.SentenceEmbeddingType) -> List[float]:
        """
        Compute the similarity for Average and SIF embedding
        :return: Cosine scores
        """
        sentences_1 = self.corpus_df["clean_sent_1"].tolist()
        sentences_2 = self.corpus_df["clean_sent_2"].tolist()
        embeddings1 = embed.SentenceEmbedding.createEmbeddingForDocs(sentences_1, sentence_embedding_type)
        embeddings2 = embed.SentenceEmbedding.createEmbeddingForDocs(sentences_2, sentence_embedding_type)
        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        return cosine_scores

    def __computeSimilarityMatrixForUse(self, sentence_embedding_type) -> List[float]:
        scores = []
        for batch in np.array_split(self.corpus_df, 10):
            scores.extend(self.__computeSimilarityMetricsPerBatchForUse(batch, sentence_embedding_type))
        return scores

    def computePearsonCorrelation(
            self,
            labels: List[float],
            sim_scores: List[float]) -> Tuple[float, float]:
        """
        Coputes the Pearson correlation
        :param sim_scores: Similarity scores
        :param label: Actual label scores
        :param sentence_embedding_type: Sentence embedding type
        :return: Pearson correlation metrics (coefficient and p-value)
        """
        pearson_metrics = scipy.stats.pearsonr(
            labels,
            sim_scores
        )
        pearson_correlation = round(pearson_metrics[0], 4)
        p_value = round(pearson_metrics[1], 4)
        return pearson_correlation, p_value


    def computeSimilarityPearsonCoefficient(
            self) -> pd.DataFrame:
        """
        Computes the Pearson coefficient b/w 'actual' and 'predicted' similarities
        :param sentence_embedding_type: Sentence embedding type
        """
        sentence_embedding_enums = [se.SentenceEmbeddingType.use,
                                    se.SentenceEmbeddingType.sbert,
                                    se.SentenceEmbeddingType.tfidf,
                                    se.SentenceEmbeddingType.average_word_embedding,
                                    se.SentenceEmbeddingType.sif_word_embedding
                                    ]
        sentence_embedding_types = []
        pearson_coefs = []
        p_values = []
        label_scores = self.corpus_df['norm_sim'].tolist()
        for sentence_embedding_enum in sentence_embedding_enums:
            predicted_sim_scores = self.__computeSimilarityMetrics(sentence_embedding_enum)
            pearson_correlation, p_value = self.computePearsonCorrelation(
                label_scores,
                predicted_sim_scores
            )
            sentence_embedding_types.append(sentence_embedding_enum.name)
            pearson_coefs.append(pearson_correlation)
            p_values.append(p_value)
        results_df = pd.DataFrame({
            "Embedding type": sentence_embedding_types,
            "Pearson correlation": pearson_coefs,
            "p-value": p_values
        })
        results_df.sort_values(by=['Pearson correlation'], ascending=False, inplace=True)
        return results_df

    def evaluateSBert(self):
        """
        Computes STS benchmack using SBERT and stores the results in a file
        :return: None
        """
        sentences_1 = self.corpus_df["clean_sent_1"].tolist()
        sentences_2 = self.corpus_df["clean_sent_2"].tolist()
        scores = self.corpus_df["norm_sim"].tolist()
        evaluator = evaluation.EmbeddingSimilarityEvaluator(
            sentences1=sentences_1,
            sentences2=sentences_2,
            scores=scores)
        evaluator(model=self.__model_sbert, output_path="Data")



if __name__ == "__main__":
    demo = STSbenchmarkProblemDemo()
    result_df = demo.computeSimilarityPearsonCoefficient()
    print("The results of the STS validation of the actual vs predicted (computed) sentence pair similarity are:\n")
    print(u.Helpers.tableize(result_df))



