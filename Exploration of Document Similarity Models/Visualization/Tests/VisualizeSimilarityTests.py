import unittest as ut
import inspect
import numpy as np
from pprint import pprint

import Visualization.VisualizeSimilarity as v
import Visualization.DimensionReductionTypes as dr


class TestFeatureMatrixData(ut.TestCase):
    """
    Test suit for 'Visualizer' class
    """
    def test_Visualizer_Construction_Is_Valid(self):
        """
        Test the validity of 'Visualizer' Constructor
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        visualizer = self.initializeVisualizer()
        self.assertIsNotNone(visualizer, msg=error_msg)

    def test_Visualizer_2D_Plot_Is_Valid(self):
        """
        Test the validity of 'Visualizer' 2D plot
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        visualizer = self.initializeVisualizer()
        self.assertIsNotNone(visualizer, msg=error_msg)
        visualizer.plot2DRepresentation()

    def test_Visualizer_Heatmap_Plot_Is_Valid(self):
        """
        Test the validity of 'Visualizer' Heatmap plot
        :return: None
        """
        error_msg = f"Invalid tests: Error testing function: {inspect.stack()[0][3]}()"
        visualizer = self.initializeVisualizer()
        self.assertIsNotNone(visualizer, msg=error_msg)
        visualizer.plotSimilarityMatrixHeatmap()

    def initializeVisualizer(self, dim_reduction_type: dr.DataReductionType = dr.DataReductionType.pca):
        """
        Initializes/instantiates the visualizer component
        :param dim_reduction_type: Dimension reduction type
        :return:
        """
        np.random.seed(1)
        n_samples = 20
        n_features = 10
        raw_feature_matrix = np.random.rand(n_samples, n_features)
        doc_ids = list(range(n_samples))
        dummy_similarity_matrix = self.createDummyTrangularSimilarityMatrixData(n_docs=n_samples)
        titles = {
            "2D-Similarity-Plot": "Plot of Feature Matrix points reduced to 2D by MDS",
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
            similarity_matrix_scores=dummy_similarity_matrix
        )
        return visualizer

    def createDummyTrangularSimilarityMatrixData(self, n_docs: int):
        """
        Create a lower triangular matrix of dummy similarity values
        :param n_docs: Number of document
        :return: lower triangular matrix
        """
        np.random.seed(1)
        full_dummy_similarity_matrix = np.random.rand(n_docs, n_docs)
        iu = np.triu_indices(n_docs)
        # Make the lower traingular section of this matrix 'Nan'
        full_dummy_similarity_matrix[iu] = float('nan')
        return full_dummy_similarity_matrix


if __name__ == '__main__':
    ut.main()
