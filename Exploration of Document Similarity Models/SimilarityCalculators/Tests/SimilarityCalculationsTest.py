import unittest as ut
import inspect
import os
from pprint import pprint
import pathlib as p
import pandas as pd
import numpy as np

import SimilarityCalculators.SimilarityCalculations as sc
import NLPProcessing.NLPEngineComponent as nlp
import FeatureExtraction.TfidfFeatureExtraction as tf
import Constants as c


class TestSimilarityCalculator(ut.TestCase):
    """
    Test suite for the 'SimilarityCalculator'
    """
    def setUp(self) -> None:
        """
        Test setup fixture
        :return: None
        """
        self.current_path = p.Path.cwd().parent.parent
        print(f"Current path is: {self.current_path}...{c.NEW_LINE}")
        os.chdir(self.current_path)
        df_corpus = pd.read_csv(r"Data\questions.csv")
        n_samples = 60
        df_sample_corpus = df_corpus[:n_samples]
        docs = df_sample_corpus.question1.tolist()
        nlp_engine = nlp.NLPEngine(docs)
        self.clean_docs = nlp_engine.preprocessDocs()
        self.all_docs = [" ".join(s) for s in self.clean_docs]
        self.query_docs = ['career launcher good rbi grade b preparation',
                            'blu ray play regular dvd player', 'nd always sad',
                            'memorable thing ever eaten',
                            'gst affects cas tax officers']
        tfidf_vectorizer = tf.TfidfFeatureVectorizer()
        self.corpus_docs_embedding = tfidf_vectorizer(docs=self.all_docs)
        self.query_docs_embedding = tfidf_vectorizer.transform(docs=self.query_docs)

    def tearDown(self) -> None:
        """
        Test tearDown fixture
        :return: None
        """
        print(f"Test is being torn down..")
        os.chdir(os.path.dirname(__file__))
        print(f"Current path is: {os.getcwd()}...{c.NEW_LINE}")

    def test_SimilarityCalculator_computeCosineSimilarityMetric_Is_Valid(self):
        """
        Test the validity of 'SimilarityCalculator' computation of Cosine similarity metric
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        sim_results = sc.SimilarityCalculator.computeCosineSimilarityMetric(
            x_query_docs=self.query_docs_embedding, y_query_docs=self.corpus_docs_embedding)
        self.assertIsNotNone(sim_results, msg=error_msg)

    def test_SimilarityCalculator_computeEuclideanSimilarityMetric_Is_Valid(self):
        """
        Test the validity of 'SimilarityCalculator' computation of Euclidean similarity metric
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        sim_results = sc.SimilarityCalculator.computeEuclideanSimilarityMetric(
            x_query_docs=self.query_docs_embedding.toarray(), y_query_docs=self.corpus_docs_embedding.toarray())
        self.assertIsNotNone(sim_results, msg=error_msg)

    def test_SimilarityCalculator_computeJaccardSimilarityMetric_Is_Valid(self):
        """
        Test the validity of 'SimilarityCalculator' computation of Jaccard similarity metric
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        corpus_docs = self.clean_docs
        query_docs = [s.split() for s in self.query_docs]
        sim_results = sc.SimilarityCalculator.computeJaccardSimilarityMetric(
            x_query_docs=query_docs, y_query_docs=corpus_docs )
        self.assertIsNotNone(sim_results, msg=error_msg)

