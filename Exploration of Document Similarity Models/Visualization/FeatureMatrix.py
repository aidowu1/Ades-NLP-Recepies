import numpy as np
from typing import List, Dict

import Visualization.Point as p


class FeatureMatrixData(object):
    """
    Encapsulates reduced dimension Feature Matrix Data fields:
        - Similarity Matrix (reduced dimension eg. 2)
        - List of Document IDs
        - Feature Data dictionary with attributes:
            - Key: Document ID
            - Value:
                - Point
    """
    def __init__(
            self,
            feature_matrix_2d: np.ndarray,
            document_ids: List[int]
    ):
        """
        Constructor
        :param feature_matrix_2d: 2-D feature matrix
        :param document_ids: List of document IDs
        """
        self.__feature_matrix_2d = feature_matrix_2d
        self.__document_ids = document_ids
        self.__feature_data_matrix = self.setFeatureData()
        self.__actual_doc_ids = self.__document_ids[:-1]
        self.__expected_doc_id = self.__document_ids[-1]

    def setFeatureData(self) -> Dict[int, p.Point]:
        """
        Setter for the feature data
        :return: Feature data as a dictionary dataset
        """
        feature_data_dict={}
        feature_matrix_per_doc_id = zip(self.__document_ids, self.__feature_matrix_2d)
        for (doc_id, item) in feature_matrix_per_doc_id:
          feature_data_dict[doc_id] = p.Point(x=item[0], y=item[1])
        return feature_data_dict

    @property
    def feature_data_matrix(self) -> Dict[int, p.Point]:
        """
        getter property for the feature data matrix
        :return:
        """
        return self.__feature_data_matrix

    @property
    def actual_doc_ids(self) -> List[int]:
        """
        Getter property for actual document IDs
        :return: Actual document IDs
        """
        return self.__actual_doc_ids

    @property
    def expected_doc_id(self) -> int:
        """
        Getter property for the expected (query) document ID
        :return: Expected (query) document ID
        """
        return self.__expected_doc_id

    def __str__(self):
        """
        String representation of the feature matrix
        :return: Feature matrix as a string
        """
        msg = "FeatureDataMatrix contains:\n"
        msg = msg + f"doc_ids: {list(self.__feature_data_matrix.keys())}\n"
        features = [str(feature) for feature in list(self.__feature_data_matrix.values())]
        msg = msg + f"feature_data values: {features}\n"
        return msg
