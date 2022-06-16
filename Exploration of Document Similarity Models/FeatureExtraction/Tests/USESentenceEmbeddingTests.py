import unittest as ut
import inspect
import os
from pprint import pprint
import pathlib as p
import pandas as pd

import FeatureExtraction.USESentenceEmbedding as us
import Constants as c
import Utils as u


class TestUseStsCalculator(ut.TestCase):
    """
    Test suit for 'USEStsCalculator' component
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

    def test_UseStsCalculator_Construction_Is_Valid(self):
        """
        Test the validity of 'USEStsCalculator' construction
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        embedder = us.USEStsCalculator(corpus=self.__docs)
        self.assertIsNotNone(embedder, msg=error_msg)

    def test_UseStsCalculator_computeSelfSimilarities_Is_Valid(self):
        """
        Test the validity of 'USEStsCalculator' similarity measure extracting query from the corpus (itself)
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        embedder = us.USEStsCalculator(self.__docs)
        sim_scores = embedder.computeSelfSimilarities()
        self.assertIsNotNone(sim_scores, msg=error_msg)
        print(f"First 10 sample similarity scores:")
        pprint(u.Helpers.sliceDict(sim_scores, 10))
        results_df = u.Helpers.displaySortedSimilarityMeasures(
            docs=self.__docs,
            similarity_result=sim_scores,
        )
        print(u.Helpers.tableize(results_df.head(50)))

    def test_UseStsCalculator_computeSimilarities_Is_Valid(self):
        """
        Test the validity of 'USEStsCalculator' similarity measure - b/w query and corpus docs
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        corpus_docs = self.__docs[:40]
        query_docs = self.__docs[40:]
        embedder = us.USEStsCalculator(corpus_docs, query_docs=query_docs)
        results_df = embedder.computeSimilarities()
        self.assertIsNotNone(results_df, msg=error_msg)
        print(u.Helpers.tableize(results_df.head(50)))

    def test_UseStsCalculator_createEmbeddingForDocs_Is_Valid(self):
        """
        Test the validity of 'USEStsCalculator' compute embeddings for specified docs
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        corpus_docs = self.__docs[:40]
        corpus_embedding = us.USEStsCalculator.createEmbeddingForDocs(docs=corpus_docs)
        self.assertIsNotNone(corpus_embedding, msg=error_msg)
        print("First vector of the corpus embedding:")
        pprint(corpus_embedding[0])



if __name__ == '__main__':
    ut.main()
