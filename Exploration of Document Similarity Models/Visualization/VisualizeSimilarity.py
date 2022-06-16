from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import  List, Dict, Tuple, Optional
import math
import os
from enum import Enum
from pprint import pprint

import Visualization.DimensionReductionTypes as dt
import Visualization.FeatureMatrix as f

sns.set()

class Visualizer(object):
    def __init__(
            self,
            vectorized_corpus: np.ndarray,
            document_ids: List[int],
            titles: Optional[Dict[str, str]] = None,
            x_labels: Optional[Dict[str, str]] = None,
            y_labels: Optional[Dict[str, str]] = None,
            similarity_matrix_scores: np.ndarray = None,
            data_reduction_method: dt.DataReductionType = dt.DataReductionType.mds,
            is_one_2_many_match: bool = True
    ):
        self.__is_one_2_many_match = is_one_2_many_match
        self.__vectorized_corpus = vectorized_corpus
        self.__document_ids = document_ids
        self.__titles = titles
        self.__x_labels = x_labels
        self.__y_labels = y_labels
        self.__similarity_matrix_scores = similarity_matrix_scores
        self.__plot_figure = None
        self.__reduced_dim_feature_data = None
        self.__dim_reduction_func_map = {
            dt.DataReductionType.mds: self.mdsDataReductionTo2D,
            dt.DataReductionType.pca: self.pcaDataReductionTo2D,
            dt.DataReductionType.tsne: self.tsneDataReductionTo2D,
            dt.DataReductionType.umap: self.umapDataReductionTo2D
        }
        self.__data_reduction_method = data_reduction_method
        self.__reduced_dim_feature_data = self.reduceDimensionTo2D()

    def reduceDimensionTo2D(self):
        """
        Method to used to reduce the dimensionality of Target/Reference vectors to 2-D using Multi Dimension Scaling (MDS)
        :return: None
        """
        return self.__dim_reduction_func_map[self.__data_reduction_method]()

    def plot2DRepresentation(self):
        """
        Plots a 2-D graph of the Similarity measure of the Target corpus to the Reference document
        :param pos: x, y coordinate positions of the Target and Reference locations
        :return:
        """
        fig = plt.figure(figsize=(8, 6))
        actual_points, expected_points = self.getActualAndExpectedPoints()
        for i, (key, point) in enumerate(actual_points):
            plt.scatter(point.x, point.y)
            plt.text(point.x, point.y, str(key))
        if self.__is_one_2_many_match:
            reference_label, point2 = expected_points
            plt.scatter(point2.x, point2.y, c='Red', marker='+')
            plt.text(point2.x, point2.y, reference_label)
            plt.xlabel(self.__x_labels["2D-Similarity-Plot"])
            plt.ylabel(self.__y_labels["2D-Similarity-Plot"])
            plt.title(self.__titles["2D-Similarity-Plot"])
        plt.show()
        return fig

    def getActualAndExpectedPoints(self) -> Tuple:
        """
        Gets the actual and expected spacial points of the text vectors
        :return:
        """
        doc_id_to_point_zip = list(zip(list(self.__reduced_dim_feature_data.feature_data_matrix.keys()),
                                       list(self.__reduced_dim_feature_data.feature_data_matrix.values())))
        print(f"doc_id_to_point_zip:\n {doc_id_to_point_zip}")
        if self.__is_one_2_many_match:
            actual_points = doc_id_to_point_zip[:-1]
            expected_points = doc_id_to_point_zip[-1]
        else:
            actual_points = doc_id_to_point_zip
            expected_points = None
        return actual_points, expected_points

    def mdsDataReductionTo2D(self):
        """
        MDS - Multi- Dimensional Scaling method to used to reduce the dimensionality of Target/Reference vectors to 2-D using Multi Dimension Scaling (MDS)
        :return: None
        """
        mds = MDS(n_components=2, random_state=1)
        reduced_feature_matrix = mds.fit_transform(self.__vectorized_corpus)
        return f.FeatureMatrixData(reduced_feature_matrix, self.__document_ids)

    def pcaDataReductionTo2D(self):
        """
        PCA - Principle Component Analysis method to used to reduce the dimensionality of Target/Reference vectors to 2-D using Multi Dimension Scaling (MDS)
        :return: None
        """
        pca = PCA(n_components=2, random_state=1)
        reduced_feature_matrix = pca.fit_transform(self.__vectorized_corpus)
        return f.FeatureMatrixData(reduced_feature_matrix, self.__document_ids)

    def tsneDataReductionTo2D(self):
        """
        T-SNE - T-distribution Stochastic Neighbourhood Embedding method to used to reduce the dimensionality of Target/Reference vectors to 2-D using Multi Dimension Scaling (MDS)
        :return: None
        """
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, random_state=1)
        reduced_feature_matrix = tsne.fit_transform(self.__vectorized_corpus)
        return f.FeatureMatrixData(reduced_feature_matrix, self.__document_ids)

    def umapDataReductionTo2D(self):
        """
        UMAP - Uniform Manifold Approximation and Projection method to used to reduce the dimensionality of Target/Reference vectors to 2-D using Multi Dimension Scaling (MDS)
        :return: None
        """
        umap = UMAP(n_components=2, random_state=1)
        reduced_feature_matrix = umap.fit_transform(self.__vectorized_corpus)
        return f.FeatureMatrixData(reduced_feature_matrix, self.__document_ids)

    def plotSimilarityMatrixHeatmap(self):
        ax = sns.heatmap(self.__similarity_matrix_scores)
        plt.xlabel(self.__x_labels["Similarity-Heatmap"])
        plt.ylabel(self.__y_labels["Similarity-Heatmap"])
        plt.title(self.__titles["Similarity-Heatmap"])
        plt.show()

    def __str__(self) -> str:
        """
        String representation of the 'Visualizer' component
        :return:
        """
        msg = f"Visualizer of Reduced Dimension Feature matrix to 2D space"
        msg = msg + f"""
         Dimension reduction is from original matrix shape: {self.__vectorized_corpus.shape} to 
                            ({len(list(self.__reduced_dim_feature_data.feature_data_matrix.values()))} ,2)
        """
        return msg
