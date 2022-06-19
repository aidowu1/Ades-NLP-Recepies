"""

Book title STS problem
======================
    - Goodreads-books is data pulled from popular book review and recommendation site Goodreads. Each entry in this
      dataset is a unique book, and contains information like author, pages, average rating, and number of
      reviews.
    - For this exercise only the book title dataset will be used.
    - This program provides a simple demo of the following themes:
        - NLP pre-processing of the books text data
        - Sentence embedding techniques such as:
            - TF-IDF: Term Frequency - Document Frequency
            - word embedding (with averaging)
            - word embedding (with SIF - Smooth Inverse Frequency)
            - USE - Universal Sentence Encoder
            - S-Bert
        - Strategies for computing sentence similarity:
            - Cosine
            - Euclidean
            - Jaccard
        - Dimensionality reduction methods:
            - PCA - Principal Component Analysis
            - MDS - Multi Dimensional Scaling
            - t-SNE - T-distributed Stochastic Neighbor Embedding
            - UMAP - Uniform Manifold Approximation and Projection
        - STS results and visualizations using scatter plots and heatmaps
    - The Goodreads-books dataset was sourced from here:
        - https://www.kaggle.com/jealousleopard/goodreadsbooks

"""

import unittest as ut
import inspect
import os
from pprint import pprint
import pathlib as p
import numpy as np
import pandas as pd
from typing import Any, List, Tuple, Dict
import logging

import FeatureExtraction.SentenceEmbedding as se
import FeatureExtraction.TfidfEmbedding as tf
import FeatureExtraction.USESentenceEmbedding as us
import FeatureExtraction.SBERTSentenceEmbedding as ss
import FeatureExtraction.SentenceEmbeddingTypes as st
import Constants as c
import Utils as u
import Visualization.VisualizeSimilarity as v
import Visualization.DimensionReductionTypes as dr
import SimilarityCalculators.SimilarityCalculations as sc

logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)

class BookTitleDemo(object):
    """
    Book title demo
    """
    def __init__(self, data_path):
        """
        Constructor
        """
        self.current_path = p.Path.cwd().parent
        print(f"Current path is: {self.current_path}...{c.NEW_LINE}")
        os.chdir(self.current_path)
        df_corpus = pd.read_csv(data_path, encoding='utf8', error_bad_lines=False)
        n_samples = 60
        df_sample_corpus = df_corpus[:n_samples]
        self.__docs = df_sample_corpus.title.tolist()

    def initializeVisualizer(
            self,
            embedder: Any,
            dim_reduction_type: dr.DataReductionType = dr.DataReductionType.pca,
            case_type: str = "case_1"
    ):
        """
        Initializes/instantiates the visualizer component
        This initialization is for use cases:
            1: Which involves the similarity measure between a query document and corpus documents
            2: Which involves the similarity measure between documents in the corpus (i.e. self-similarity)
        :param embedder: Embedding strategy
        :param dim_reduction_type: Dimension reduction type
        :param case_type: Case type i.e. self similarity (case 1) or
               similarity between query doc and corpus docs (case 2)
        :return: Visualizer object that can be used to create Scatter and heatmap plots
        """
        if case_type == "case_1":
            sim_scores = embedder.computeSelfSimilarities()
            sim_scores_df = u.Helpers.displaySortedSimilarityMeasures(
                docs=self.__docs,
                similarity_result=sim_scores,
            )
            raw_feature_matrix = np.array(embedder.corpus_embeddings)
            similarity_matrix = embedder.similarity_matrix_self
        else:
            sim_scores_df = embedder.computeSimilarities()
            corpus_embedding = embedder.corpus_embeddings
            query_embedding = embedder.query_embeddings
            raw_feature_matrix = np.vstack((corpus_embedding, query_embedding))
            similarity_matrix = embedder.similarity_matrix

        doc_ids = list(range(raw_feature_matrix.shape[0]))

        titles = {
            "2D-Similarity-Plot": f"Plot of Feature Matrix points reduced to 2D by {dim_reduction_type.name}",
            "Similarity-Heatmap": "Plot of Similarity Matrix Heatmap"
        }
        x_labels = {
            "2D-Similarity-Plot": "X coordinates",
            "Similarity-Heatmap": "Doc Ids"
        }
        y_labels = {
            "2D-Similarity-Plot": "Y coordinates",
            "Similarity-Heatmap": "Doc Ids"
        }
        visualizer = v.Visualizer(
            raw_feature_matrix,
            doc_ids, titles,
            x_labels,
            y_labels,
            data_reduction_method=dim_reduction_type,
            similarity_matrix_scores=similarity_matrix
        )
        print(u.Helpers.tableize(sim_scores_df.head(30)))
        return visualizer


    def runSentenceEmbeddingCaseOne(self):
        """
        Test the validity of 'SentenceEmbedding' self-similarity visualizations (Case 1)
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        corpus_docs = self.__docs[:40]
        query_docs = None
        embedder = se.SentenceEmbedding(
            corpus=corpus_docs,
            query_docs=query_docs,
            sentence_embedding_type=st.SentenceEmbeddingType.sif_word_embedding)
        visualizer = self.initializeVisualizer(
            embedder=embedder,
            dim_reduction_type=dr.DataReductionType.mds, case_type="case_1")
        assert visualizer != None, error_msg
        visualizer.plot2DRepresentation()
        visualizer.plotSimilarityMatrixHeatmap()

    def runUSEStsCalculatorCaseOne(self):
        """
        Test the validity of 'USEStsCalculator' self-similarity visualizations (Case 1)
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        corpus_docs = self.__docs[:40]
        query_docs = None
        embedder = us.USEStsCalculator(
            corpus=corpus_docs,
            query_docs=query_docs,
            sentence_embedding_type=st.SentenceEmbeddingType.use)
        visualizer = self.initializeVisualizer(
            embedder=embedder,
            dim_reduction_type=dr.DataReductionType.tsne, case_type="case_1")
        assert visualizer != None, error_msg
        visualizer.plot2DRepresentation()
        visualizer.plotSimilarityMatrixHeatmap()

    def runSBERTStsCalculatorCaseOne(self):
        """
        Test the validity of 'SBERTStsCalculator' self-similarity visualizations (Case 1)
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        corpus_docs = self.__docs[:40]
        query_docs = None
        embedder = ss.SBERTStsCalculator(
            corpus=corpus_docs,
            query_docs=query_docs,
            sentence_embedding_type=st.SentenceEmbeddingType.sbert)
        visualizer = self.initializeVisualizer(
            embedder=embedder,
            dim_reduction_type=dr.DataReductionType.umap, case_type="case_1")
        assert visualizer != None, error_msg
        visualizer.plot2DRepresentation()
        visualizer.plotSimilarityMatrixHeatmap()

    def runTFIDFStsCalculatorCaseOne(self):
        """
        Test the validity of 'TFIDFStsCalculator' self-similarity visualizations (Case 1)
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        corpus_docs = self.__docs[:40]
        query_docs = None
        embedder = tf.TFIDFStsCalculator(
            corpus=corpus_docs,
            query_docs=query_docs,
            sentence_embedding_type=st.SentenceEmbeddingType.tfidf)
        visualizer = self.initializeVisualizer(
            embedder=embedder,
            dim_reduction_type=dr.DataReductionType.pca, case_type="case_1")
        assert visualizer != None, error_msg
        visualizer.plot2DRepresentation()
        visualizer.plotSimilarityMatrixHeatmap()

    def runSBERTStsCalculatorCaseTwo(self):
        """
        Test the validity of 'USEStsCalculator' Query doc vs Corpus docs similarity visualizations (Case 2)
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        corpus_docs = self.__docs[:40]
        n_corpus_docs = len(corpus_docs)
        query_docs = self.__docs[n_corpus_docs:n_corpus_docs+1]
        embedder = ss.SBERTStsCalculator(
            corpus=corpus_docs,
            query_docs=query_docs,
            sentence_embedding_type=st.SentenceEmbeddingType.sbert)
        visualizer = self.initializeVisualizer(
            embedder=embedder,
            dim_reduction_type=dr.DataReductionType.umap, case_type="case_2")
        assert visualizer != None, error_msg
        visualizer.plot2DRepresentation()
        visualizer.plotSimilarityMatrixHeatmap()

    def runSimilarityCalculationStrategies(self):
        """
        Test the validity of 'SimilarityCalculator' computation of Cosine similarity metric
        :return: None
        """
        logging.info("Starting demo of the Similarity calculations...")
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"

        self.__corpus_docs = self.__docs[:20]
        self.__query_docs = self.__docs[10:11]
        self.__query_docs_embedding = ss.SBERTStsCalculator.createEmbeddingForDocs(self.__query_docs)
        self.__corpus_docs_embedding = ss.SBERTStsCalculator.createEmbeddingForDocs(self.__corpus_docs)

        # Cosine
        cosine_sim_results = sc.SimilarityCalculator.computeCosineSimilarityMetric(
            x_query_docs=self.__query_docs_embedding, y_query_docs=self.__corpus_docs_embedding)
        assert cosine_sim_results.shape[0] > 0, error_msg

        # Euclidean
        euclidean_sim_results = sc.SimilarityCalculator.computeEuclideanSimilarityMetric(
            x_query_docs=self.__query_docs_embedding, y_query_docs=self.__corpus_docs_embedding)
        assert euclidean_sim_results.shape[0] > 0, error_msg

        # Jaccard
        corpus_docs_tokenized = [s.split() for s in self.__corpus_docs]
        query_docs_tokenized = [s.split() for s in self.__query_docs]
        jaccard_sim_results = sc.SimilarityCalculator.computeJaccardSimilarityMetric(
            x_query_docs=query_docs_tokenized, y_query_docs=corpus_docs_tokenized)
        assert jaccard_sim_results.shape[0] > 0, error_msg

        results_df = pd.DataFrame({
            "corpus_docs": self.__corpus_docs,
            "jaccard scores": jaccard_sim_results.flatten().tolist(),
            "euclidean_scores": euclidean_sim_results.flatten().tolist(),
            "cosine_scores": cosine_sim_results.flatten()
        })
        results_df.sort_values(by=['cosine_scores', 'euclidean_scores', 'jaccard scores'], ascending=False, inplace=True)

        results_df.euclidean_scores = results_df.euclidean_scores.apply(lambda x: round(x, 4))
        results_df.cosine_scores = results_df.cosine_scores.apply(lambda x: round(x, 4))

        print("Demo the similarity calculations of 'Jaccard', 'Euclidean' and 'Cosine'")
        print(f"The query document is:{self.__query_docs}")
        print("The results are:")
        print(u.Helpers.tableize(results_df))

def runBookTitleDemo(data_path: str):
    """
    Runs the book title demo
    :return: None
    """
    demo = BookTitleDemo(data_path=data_path)
    demo.runSentenceEmbeddingCaseOne()
    demo.runUSEStsCalculatorCaseOne()
    demo.runSBERTStsCalculatorCaseOne()
    demo.runTFIDFStsCalculatorCaseOne()
    demo.runSBERTStsCalculatorCaseTwo()
    demo.runSimilarityCalculationStrategies()


if __name__ == "__main__":
    runBookTitleDemo(data_path="Data/books.csv")