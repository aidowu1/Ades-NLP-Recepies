import abc
import numpy as np
import scipy.sparse as s
from typing import Tuple, Dict, List, Union
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

class SimilarityCalculator(object):
    """
    Component for similarity calculation specification
    """

    @staticmethod
    def computeCosineSimilarityMetric(
            x_query_docs: Union[np.ndarray, s.csr_matrix],
            y_query_docs: Union[np.ndarray, s.csr_matrix]
            ) -> Union[np.ndarray, s.csr_matrix]:
        """
        Computes the 'Cosine' similarity metrics between documents
        :params x_query_docs: Query x documents with shape n_samples by n_features
        :params y_query_docs: Query y documents with shape n_samples by n_features
        :return: Similarity metric results as dictionary and matrix
        :return:
        """
        sim_results = cosine_similarity(x_query_docs, y_query_docs)
        return sim_results

    @staticmethod
    def computeEuclideanSimilarityMetric(
            x_query_docs: np.ndarray,
            y_query_docs: np.ndarray
    ) -> np.ndarray:
        """
        Computes the 'Cosine' similarity metrics between documents
        :params x_query_docs: Query x documents with shape n_samples by n_features
        :params y_query_docs: Query y documents with shape n_samples by n_features
        :return: Similarity metric results as dictionary and matrix
        :return:
        """
        def distance2Similarity(distance: np.ndarray):
            """
            Normalises the Euclidean distance to a similarity measure
            :param distance: Distance measure
            :return: Similarity measure
            """
            return 1 / np.exp(distance)

        sim_results = distance2Similarity(euclidean_distances(x_query_docs, y_query_docs))
        return sim_results

    @staticmethod
    def __computeJaccardSimilarityMetricScalar(
            x_query_doc: List[str],
            y_query_doc: List[str]
    ) -> float:
        """
        Computes the 'Jaccard' similarity metrics between 2 documents
        :params x_query: Query x document
        :params y_query_docs: Query y document
        :return: Jaccard similarity score
        """
        intersection_cardinality = len(set.intersection(*[set(x_query_doc), set(y_query_doc)]))
        union_cardinality = len(set.union(*[set(x_query_doc), set(y_query_doc)]))
        if union_cardinality == 0:
            jaccard_score = 0.0
        else:
            jaccard_score = round(intersection_cardinality / union_cardinality, 4)
        return jaccard_score

    @staticmethod
    def computeJaccardSimilarityMetric(
            x_query_docs: List[List[str]],
            y_query_docs: List[List[str]]
    ) -> np.ndarray:
        """
        Computes the 'Jaccard' similarity metrics between documents
        :param x_query_docs: Query documents
        :param y_query_docs: Corpus documents
        :return: Similarity results
        """
        n_rows = len(x_query_docs)
        n_cols = len(y_query_docs)
        sim_results = np.zeros((n_rows, n_cols))
        for i in range(n_rows):
            for j in range(n_cols):
                sim_results[i, j] = SimilarityCalculator.__computeJaccardSimilarityMetricScalar(
                                    x_query_doc=x_query_docs[i],
                                    y_query_doc=y_query_docs[j]
                )
        return sim_results


