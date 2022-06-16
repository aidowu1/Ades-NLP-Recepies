import unittest as ut
import inspect
import pandas as pd
import numpy as np
import pathlib as p
import os

import FeatureExtraction.TfidfFeatureExtraction as tf
import NLPProcessing.NLPEngineComponent as nlp
import Constants as c

class TestTfidfFeatureVectorizer(ut.TestCase):
    """
    Test suit for 'TfidfFeatureVectorizer' component
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
        n_samples = 100
        df_sample_corpus = df_corpus[:n_samples]
        docs = df_sample_corpus.question1.tolist()
        nlp_engine = nlp.NLPEngine(docs)
        clean_docs = nlp_engine.preprocessDocs()
        self.corpus_docs = [" ".join(s) for s in clean_docs[:50]]

    def test_TfidfFeatureVectorizer_Constructor_Is_Valid(self):
        """
        Test the validity of the 'TfidfFeatureVectorizer' constructor
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        tfidf_vectorizer = tf.TfidfFeatureVectorizer()
        self.assertIsNotNone(tfidf_vectorizer, msg=error_msg)

    def test_TfidfFeatureVectorizer_Call_Is_Valid(self):
        """
        Test the validity of the 'TfidfFeatureVectorizer' callable
        :return:
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        tfidf_vectorizer = tf.TfidfFeatureVectorizer()
        docs_embedding = tfidf_vectorizer(docs=self.corpus_docs)
        self.assertIsNotNone(docs_embedding, msg=error_msg)
        print(f"docs_embedding.shape: {docs_embedding.shape}")


if __name__ == '__main__':
    ut.main()
