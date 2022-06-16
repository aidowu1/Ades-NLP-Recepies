import unittest as ut
import inspect
import os
from pprint import pprint
import pathlib as p
import pandas as pd

import NLPProcessing.NLPEngineComponent as nlp
import Constants as c


class TestNLPEngine(ut.TestCase):
    """
    Test suite for the 'NLPEngine'
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
        self.df_sample_corpus = df_corpus[:n_samples]

    def tearDown(self) -> None:
        """
        Test tearDown fixture
        :return: None
        """
        print(f"Test is being torn down..")
        os.chdir(os.path.dirname(__file__))
        print(f"Current path is: {os.getcwd()}...{c.NEW_LINE}")

    def test_NLPEngine_Construction_Is_Valid(self):
        """
        test the validity of 'NLPEngine' constructor
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        docs = self.df_sample_corpus.question1.tolist()
        nlp_engine = nlp.NLPEngine(docs)
        self.assertIsNotNone(nlp_engine, msg=error_msg)
        pprint(f"NLP Engine: {nlp_engine}")

    def test_NLPEngine_Preprocessing_Is_Valid(self):
        """
        test the validity of 'NLPEngine' pre-processing
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        docs = self.df_sample_corpus.question1.tolist()
        nlp_engine = nlp.NLPEngine(docs)
        self.assertIsNotNone(nlp_engine, msg=error_msg)
        pprint(f"NLP Engine: {nlp_engine}{c.NEW_LINE}")
        preprocessed_docs = nlp_engine.preprocessDocs()
        print(f"First 10 sample docs before preprocessing:{c.NEW_LINE}")
        pprint(docs[:10])
        print(f"{c.NEW_LINE}{c.NEW_LINE}{c.LINE_DIVIDER}{c.NEW_LINE}")
        print(f"First 10 docs after preprocessing:{c.NEW_LINE}")
        pprint(preprocessed_docs[:10])
        print(f"{c.NEW_LINE}{c.NEW_LINE}")



if __name__ == '__main__':
    ut.main()
