from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import math
import os
from enum import Enum
from pprint import pprint


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        msg = f"(x, y) = ({self.x}, {self.y})"
        return msg


class DataReductionMethod(Enum):
    mds = 0
    pca = 1
    tsne = 3
    umap = 4

class FeatureMatrixData(object):
    """
    Encapsulates reduced dimension Feature Matrix Data fields:
        - Similarity Matrix (reduced dimension eg. 2)
        - List of Document IDs
        - Feature Data dictionary with attribtes:
            - Key: Document ID
            - Value: 
                - Point 
    """
    def __init__(self, feature_matrix_2d, document_ids):
        self.__feature_matrix_2d = feature_matrix_2d
        self.__document_ids = document_ids
        self.__feature_data_matrix = self.setFeatureData()        
        self.__actual_doc_ids = self.__document_ids[:-1]
        self.__expected_doc_id = self.__document_ids[-1]

    def setFeatureData(self):
        feature_data_dict={}
        feature_matrix_per_doc_id = zip(self.__document_ids, self.__feature_matrix_2d)
        for (doc_id, item) in feature_matrix_per_doc_id:
          feature_data_dict[doc_id] = Point(x=item[0], y=item[1])
        return feature_data_dict

    @property
    def feature_data_matrix(self):
        return self.__feature_data_matrix

    @property
    def actual_doc_ids(self):
        return self.__actual_doc_ids

    @property
    def expected_doc_id(self):
        return self.__expected_doc_id

    def __str__(self):
        msg = "FeatureDataMatrix contains:\n"
        msg = msg + f"doc_ids: {list(self.__feature_data_matrix.keys())}\n"
        features = [str(feature) for feature in list(self.__feature_data_matrix.values())]
        msg = msg + f"feature_data values: {features}\n"        
        return msg

class Visualizer(object):
    def __init__(
            self,            
            vectorized_corpus,
            document_ids,
            titles = None,
            x_labels = None,
            y_labels = None,
            similarity_matrix_scores = None,
            data_redution_method = DataReductionMethod.mds,
            is_one_2_many_match = True,
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
        self.__data_reduction_method = data_redution_method
        self.reduceDimensionTo2D()

    def reduceDimensionTo2D(self):
        """
        Method to used to reduce the dimensionality of Target/Reference vectors to 2-D using Multi Dimension Scaling (MDS)
        :return: None
        """        
        if self.__data_reduction_method == DataReductionMethod.mds:
            return self.mdsDataReductionTo2D()
        elif self.__data_reduction_method == DataReductionMethod.pca:
            return self.pcaDataReductionTo2D()
        elif self.__data_reduction_method == DataReductionMethod.tsne:
            return self.tsneDataReductionTo2D()
        elif self.__data_reduction_method == DataReductionMethod.umap:
            return self.umapDataReductionTo2D()
        else:
            return self.mdsDataReductionTo2D()

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

    def getActualAndExpectedPoints(self):
        doc_id_to_point_zip = list(zip(list(self.__reduced_dim_feature_data.feature_data_matrix.keys()), 
                                list(self.__reduced_dim_feature_data.feature_data_matrix.values())) )
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
        self.__reduced_dim_feature_data = FeatureMatrixData(reduced_feature_matrix, self.__document_ids)

    def pcaDataReductionTo2D(self):
        """
        PCA - Principle Component Analysis method to used to reduce the dimensionality of Target/Reference vectors to 2-D using Multi Dimension Scaling (MDS)
        :return: None
        """
        pca = PCA(n_components=2, random_state=1)
        reduced_feature_matrix = pca.fit_transform(self.__vectorized_corpus)
        self.__reduced_dim_feature_data = FeatureMatrixData(reduced_feature_matrix, self.__document_ids)

    def tsneDataReductionTo2D(self):
        """
        T-SNE - T-distribution Stochastic Neighbourhood Embedding method to used to reduce the dimensionality of Target/Reference vectors to 2-D using Multi Dimension Scaling (MDS)
        :return: None
        """
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, random_state=1)
        reduced_feature_matrix = tsne.fit_transform(self.__vectorized_corpus)
        self.__reduced_dim_feature_data = FeatureMatrixData(reduced_feature_matrix, self.__document_ids)

    def umapDataReductionTo2D(self):
        """
        UMAP - Uniform Manifold Approximation and Projection method to used to reduce the dimensionality of Target/Reference vectors to 2-D using Multi Dimension Scaling (MDS)
        :return: None
        """
        umap = UMAP(n_components=2, random_state=1)
        reduced_feature_matrix = umap.fit_transform(self.__vectorized_corpus)
        self.__reduced_dim_feature_data = FeatureMatrixData(reduced_feature_matrix, self.__document_ids)

    def plotSimilarityMatrixHeatmap(self):
        ax = sns.heatmap(self.__similarity_matrix_scores)
        plt.xlabel(self.__x_labels["Similarity-Heatmap"])
        plt.ylabel(self.__y_labels["Similarity-Heatmap"])
        plt.title(self.__titles["Similarity-Heatmap"])
        plt.show()

    def __str__(self):
        msg = f"Visualizer of Reduced Dimension Feature matrix to 2D space"
        msg = msg + f"""
         Dimension reduction is from original matrix shape: {self.__vectorized_corpus.shape} to 
                            ({len(list(self.__reduced_dim_feature_data.feature_data_matrix.values()))} ,2)
        """
        return msg
        

def test_FeatureMatrixData_Is_Valid():
    np.random.seed(1)
    data_2d = np.random.randint(10, size=(20,2))
    doc_ids = list(range(20))
    feature_data = FeatureMatrixData(data_2d, doc_ids)
    pprint(f"feature_data: {feature_data}")

def test_Visualizer_Is_Valid():
    np.random.seed(1)
    raw_feature_matrix = np.random.rand(20,10)
    doc_ids = list(range(20))
    titles = { 
        "2D-Similarity-Plot":"Plot of Feature Matrix points reduced to 2D by MDS",
        "Similarity-Heatmap": "Plot of Similarity Matrix Heatmap"
    }
    x_labels = { 
        "2D-Similarity-Plot":"X coordinates",
        "Similarity-Heatmap": "Doc Ids"
    }
    y_labels = { 
        "2D-Similarity-Plot":"Y coordinates",
        "Similarity-Heatmap": "Doc Ids"
    }
    visualizer = Visualizer(raw_feature_matrix, doc_ids, titles, x_labels, y_labels)
    print(f"\nVisualizer details are: {visualizer}")
    visualizer.plot2DRepresentation()

def test_Visualizer_With_PCA_Reduction_Is_Valid():
    np.random.seed(1)
    raw_feature_matrix = np.random.rand(20,10)
    doc_ids = list(range(20))
    titles = { 
        "2D-Similarity-Plot":"Plot of Feature Matrix points reduced to 2D by UMAP",
        "Similarity-Heatmap": "Plot of Similarity Matrix Heatmap"
    }
    x_labels = { 
        "2D-Similarity-Plot":"X coordinates",
        "Similarity-Heatmap": "Doc Ids"
    }
    y_labels = { 
        "2D-Similarity-Plot":"Y coordinates",
        "Similarity-Heatmap": "Doc Ids"
    }
    visualizer = Visualizer(raw_feature_matrix, doc_ids, titles, 
                       x_labels, y_labels, DataReductionMethod.umap)
    print(f"\nVisualizer details are: {visualizer}")
    visualizer.plot2DRepresentation()

def createDummyTrangularSimilarityMatrixData(n_docs):
    np.random.seed(1)
    full_dummy_similarity_matrix = np.random.rand(n_docs, n_docs)
    iu = np.triu_indices(n_docs)
    # Make the lower traingular section of this matrix 'Nan'
    full_dummy_similarity_matrix[iu] = float('nan')
    return full_dummy_similarity_matrix


def test_Visualizer_Plot_Similarity_Matrix_Heatmap_Is_valid():
    np.random.seed(1)
    n_docs = 20
    n_features = 10
    raw_feature_matrix = np.random.rand(n_docs, n_features)
    doc_ids = list(range(n_docs))
    dummy_similarity_matrix = createDummyTrangularSimilarityMatrixData(n_docs)
    titles = { 
        "2D-Similarity-Plot":"Plot of Feature Matrix points reduced to 2D by MDS",
        "Similarity-Heatmap": "Plot of Similarity Matrix Heatmap"
    }
    x_labels = { 
        "2D-Similarity-Plot":"X coordinates",
        "Similarity-Heatmap": "Doc Ids"
    }
    y_labels = { 
        "2D-Similarity-Plot":"Y coordinates",
        "Similarity-Heatmap": "Doc Ids"
    }
    visualizer = Visualizer(raw_feature_matrix, doc_ids, titles, x_labels, y_labels,
                            dummy_similarity_matrix, DataReductionMethod.mds)
    print(f"\nVisualizer details are: {visualizer}")
    visualizer.plotSimilarityMatrixHeatmap()




if __name__ == "__main__":
    #test_Visualizer_Is_Valid()
    #test_Visualizer_With_PCA_Reduction_Is_Valid()
    test_Visualizer_Plot_Similarity_Matrix_Heatmap_Is_valid()
    #pass