import unittest as ut
import inspect
import os
from pprint import pprint
import pathlib as p

import numpy as np
import pandas as pd

import FeatureExtraction.SentenceEmbedding as se
import FeatureExtraction.SentenceEmbeddingTypes as st
import Constants as c
import Utils as u
import Visualization.VisualizeSimilarity as v
import Visualization.DimensionReductionTypes as dr


class TestSentenceEmbedding(ut.TestCase):
    """
    Test suit for 'SentenceEmbedding' component
    """
    def setUp(self) -> None:
        """
        Test setup fixture
        :return: None
        """
        self.current_path = p.Path.cwd().parent.parent
        print(f"Current path is: {self.current_path}...{c.NEW_LINE}")
        os.chdir(self.current_path)
        df_corpus = pd.read_csv(r"Data\books.csv",encoding='utf8', error_bad_lines=False)
        n_samples = 60
        df_sample_corpus = df_corpus[:n_samples]
        self.__docs = df_sample_corpus.title.tolist()

    def tearDown(self) -> None:
        """
        Test tearDown fixture
        :return: None
        """
        print(f"Test is being torn down..")
        os.chdir(os.path.dirname(__file__))
        print(f"Current path is: {os.getcwd()}...{c.NEW_LINE}")

    def test_SentenceEmbedding_Construction_Is_Valid(self):
        """
        Test the validity of 'SentenceEmbedding' construction
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        embedder = se.SentenceEmbedding(self.__docs)
        self.assertIsNotNone(embedder, msg=error_msg)

    def test_SentenceEmbedding_computeSelfSimilarities_Using_Average_Is_Valid(self):
        """
        Test the validity of 'SentenceEmbedding' similarity measure using Average
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        embedder = se.SentenceEmbedding(
            self.__docs,
            sentence_embedding_type=st.SentenceEmbeddingType.average_word_embedding)
        sim_scores = embedder.computeSelfSimilarities()
        self.assertIsNotNone(sim_scores, msg=error_msg)
        print(f"First 10 sample similarity scores:")
        pprint(u.Helpers.sliceDict(sim_scores, 10))
        results_df = u.Helpers.displaySortedSimilarityMeasures(
            docs=self.__docs,
            similarity_result=sim_scores,
        )
        print(u.Helpers.tableize(results_df.head(50)))

    def test_SentenceEmbedding_computeSelfSimilarities_Using_SIF_Is_Valid(self):
        """
        Test the validity of 'SentenceEmbedding' similarity measure using SIF
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        embedder = se.SentenceEmbedding(
            self.__docs,
            sentence_embedding_type=st.SentenceEmbeddingType.sif_word_embedding)
        sim_scores = embedder.computeSelfSimilarities()
        self.assertIsNotNone(sim_scores, msg=error_msg)
        print(f"First 10 sample similarity scores:")
        pprint(u.Helpers.sliceDict(sim_scores, 10))
        results_df = u.Helpers.displaySortedSimilarityMeasures(
            docs=self.__docs,
            similarity_result=sim_scores,
        )
        print(u.Helpers.tableize(results_df.head(50)))

    def test_SentenceEmbedding_computeSimilarities_Using_Average_Is_Valid(self):
        """
        Test the validity of 'SentenceEmbedding' similarity measure using Average
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        corpus_docs = self.__docs[:40]
        query_docs = self.__docs[40:]
        embedder = se.SentenceEmbedding(
            corpus=corpus_docs,
            query_docs=query_docs,
            sentence_embedding_type=st.SentenceEmbeddingType.average_word_embedding)
        results_df = embedder.computeSimilarities()
        self.assertIsNotNone(results_df, msg=error_msg)
        print(u.Helpers.tableize(results_df.head(50)))

    def test_SentenceEmbedding_computeSimilarities_Using_SIF_Is_Valid(self):
        """
        Test the validity of 'SentenceEmbedding' similarity measure using SIF
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        corpus_docs = self.__docs[:40]
        query_docs = self.__docs[40:]
        embedder = se.SentenceEmbedding(
            corpus=corpus_docs,
            query_docs=query_docs,
            sentence_embedding_type=st.SentenceEmbeddingType.sif_word_embedding)
        results_df = embedder.computeSimilarities()
        self.assertIsNotNone(results_df, msg=error_msg)
        print(u.Helpers.tableize(results_df.head(50)))

    def test_SentenceEmbedding_Getting_Corpus_And_Query_Embeddings_Is_Valid(self):
        """
        Test the validity of 'SentenceEmbedding' getting Corpus and Query embeddings
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        corpus_docs = self.__docs[:40]
        query_docs = self.__docs[40:]
        embedder = se.SentenceEmbedding(
            corpus=corpus_docs,
            query_docs=query_docs,
            sentence_embedding_type=st.SentenceEmbeddingType.sif_word_embedding)
        embedder()
        corpus_embedding = embedder.corpus_embeddings
        query_embedding = embedder.query_embeddings
        self.assertIsNotNone(corpus_embedding, msg=error_msg)
        self.assertIsNotNone(query_embedding, msg=error_msg)
        print("Sample of 1st vector of Corpus embedding:")
        pprint(corpus_embedding[0])
        print("\n\nSample of 1st vector of Query embedding:")
        pprint(query_embedding[0])

    def test_SentenceEmbedding_Visualizing_Embeddings_Case_One_Is_Valid(self):
        """
        Test the validity of 'SentenceEmbedding' visualizing Corpus and Query embeddings
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        visualizer = self.initializeVisualizer(dim_reduction_type=dr.DataReductionType.mds, case_type="case_1")
        self.assertIsNotNone(visualizer, msg=error_msg)
        visualizer.plot2DRepresentation()
        visualizer.plotSimilarityMatrixHeatmap()

    def test_SentenceEmbedding_Visualizing_Embeddings_Case_Two_Is_Valid(self):
        """
        Test the validity of 'SentenceEmbedding' visualizing Corpus and Query embeddings
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        visualizer = self.initializeVisualizer(dim_reduction_type=dr.DataReductionType.umap, case_type="case_2")
        self.assertIsNotNone(visualizer, msg=error_msg)
        visualizer.plot2DRepresentation()
        visualizer.plotSimilarityMatrixHeatmap()


    def initializeVisualizer(
            self,
            dim_reduction_type: dr.DataReductionType = dr.DataReductionType.pca,
            case_type: str = "case_1"
    ):
        """
        Initializes/instantiates the visualizer component
        This initialization is for use cases:
            1: Which involves the similarity measure between a query document and corpus documents
            2: Which involves the similarity measure between documents in the corpus (i.e. self-similarity)
        :param dim_reduction_type: Dimension reduction type
        :return:
        """
        corpus_docs = self.__docs[:40]
        n_corpus_docs = len(corpus_docs)
        if case_type == "case_2":
            query_docs = self.__docs[n_corpus_docs:n_corpus_docs+1]
        else:
            query_docs = None
        embedder = se.SentenceEmbedding(
            corpus=corpus_docs,
            query_docs=query_docs,
            sentence_embedding_type=st.SentenceEmbeddingType.sif_word_embedding)
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



if __name__ == '__main__':
    ut.main()
