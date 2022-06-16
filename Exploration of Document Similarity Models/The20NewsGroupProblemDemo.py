"""
20 News Group Problem
Description:    
The 20 newsgroups dataset comprises around 18000 newsgroups posts on 20 topics 
split in two subsets: one for training (or development) and the other one for 
testing (or for performance evaluation). The split between the train and test 
set is based upon a messages posted before and after a specific date.

Steps:
    - Getting the 20 News Group Problem corpus data via the Scikit Learn's sklearn.datasets.fetch_20newsgroups module 
    - Text data NLP pre-processing using the NLPEngineComponent module
    - Feature extration using TFIDF vectorization provided by the TFIDFDocumentVectorExtractor module
    - Document similarity computation using Cosine Similarity measure in the TFIDFDocumentVectorExtractor module
    - 2D Visualization of the documents using Multi-Dimensional Scaling (MDS) provided in the DocumentFeatureVisualization module 

Data/reference was based on online SciKit Learn documentation lacated here: https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html
"""

# Define imports
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import numpy as np
import os
import operator
from pprint import pprint
import GenericDataSerializerComponent as s
import ProblemSpecificationInterface as p
import NLPEngineComponent as nlp
import TFIDFDocmentVectorExtractor as tf_idf
import DocumentFeatureVisualization as dfv


class NewsGroup20Problem(p.IProblemSpec):
    def __init__(self):
        self.__num_docs = -1
        self.__corpus = None
        self.__clean_corpus = None
        self.__nlp_engine = None
        self.__sparse_feature_matrix = None
        self.__dense_feature_matrix = None
        self.__similarity_matrix_dict = {}
        self.__similarity_matrix = None
        self.__doc_ids = None
        self.__visualizer = None

    def extractRadomSampleOf20NewsGroupData(self, n_samples=50):
        """
        Returns a specified sample of the 20 News Group data
        based on a sub-set of categories 
        """
        categories = ['alt.atheism', 'talk.religion.misc',
              'comp.graphics', 'sci.space']
        newsgroups_train = fetch_20newsgroups(subset='train',
                                      categories=categories)
        sample_corpus = newsgroups_train.data[:n_samples]
        return sample_corpus

    def getCorpus(self):
        """
        Get the corpus data for this problem
        """
        self.__corpus = self.extractRadomSampleOf20NewsGroupData()
        self.__num_docs = len(self.__corpus)
        if self.__doc_ids is None:
            self.__doc_ids = list(range(self.__num_docs))
        print("Number of rows for the Toy Problem data are: {}".format(self.__num_docs))
        return self.__corpus

    def cleanCorpus(self):
        """
        Cleans/normalises the corpus
        """
        self.__nlp_engine = nlp.NLPEngine(self.__corpus)
        self.__clean_corpus = self.__nlp_engine.preprocessDocs()
        return self.__clean_corpus

    def computeDocSimilarity(self):
        tfidf_extractor = tf_idf.TFIDFVectorExtraction()
        self.__sparse_feature_matrix, self.__dense_feature_matrix = tfidf_extractor.createFeatureMatrix(self.__clean_corpus)
        n_docs = self.__dense_feature_matrix.shape[0]
        print(f"Shape of Feature Matrix is: {n_docs}")
        doc_pair_indices = self.createLowerTriangularMatrixOfPairs()
        self.__similarity_matrix = np.zeros((n_docs, n_docs))
        for ind in doc_pair_indices:
            doc_1_ind, doc_2_ind = ind
            similarity = tfidf_extractor.measureSimilarity(self.__dense_feature_matrix[doc_1_ind], self.__dense_feature_matrix[doc_2_ind])
            key = "{0}_{1}".format(doc_1_ind, doc_2_ind)
            self.__similarity_matrix_dict[key] = similarity
            self.__similarity_matrix[doc_1_ind, doc_2_ind] = similarity
        return self.__similarity_matrix_dict         
    
    def createLowerTriangularMatrixOfPairs(self):
        """
        Create triangular matrix indices doc_pair_indices for the similarity measure
        """
        matrix = np.zeros((self.__num_docs, self.__num_docs))
        indices = np.tril_indices_from(matrix)
        n_rows = indices[0].shape[0]
        pairs = [(indices[0][i], indices[1][i]) for i in range(n_rows) if not indices[0][i] == indices[1][i]]
        return pairs

    def displaySortedSimilarityMeasures(self):
        print("\nSorted Similarity Measures (in descending) order are:\n\n")
        for key, value in sorted(self.__similarity_matrix_dict.items(), key=lambda item: item[1], reverse=True):
            print("{0}: {1}".format(key, value))

    def visualizeDocumentSimilarity(self):
        """
        Provides the visualization of similarity between the Documents in a corpus
        Each document is labelled with a ID (a sequence number)
        """
        titles = { 
        "2D-Similarity-Plot":"Plot of Feature Matrix points reduced to 2D by MDS for '20 News Group' Problem",
        "Similarity-Heatmap": "Plot of Similarity Matrix Heatmap for '20 News Group' Problem"
        }
        x_labels = { 
        "2D-Similarity-Plot":"X coordinates",
        "Similarity-Heatmap": "Document Ids"
        }
        y_labels = { 
            "2D-Similarity-Plot":"Y coordinates",
            "Similarity-Heatmap": "Document Ids"
        }
        if  self.__visualizer is None:       
            self.__visualizer = dfv.Visualizer(self.__dense_feature_matrix, self.__doc_ids,
                                               titles, x_labels, y_labels, self.__similarity_matrix, dfv.DataReductionType.mds)
        print(f"\nVisualizer details are: {self.__visualizer}")
        self.__visualizer.plot2DRepresentation()

    def plotSimilarityMatrixHeatmap(self):
        """
        Plots the Heatmap of the similarity matrix scores
        """
        titles = { 
        "2D-Similarity-Plot":"Plot of Feature Matrix points reduced to 2D by MDS for '20 News Group' Problem",
        "Similarity-Heatmap": "Plot of Similarity Matrix Heatmap for '20 News Group' Problem"
        }
        x_labels = { 
        "2D-Similarity-Plot":"X coordinates",
        "Similarity-Heatmap": "Book Title Ids"
        }
        y_labels = { 
            "2D-Similarity-Plot":"Y coordinates",
            "Similarity-Heatmap": "Book Title Ids"
        }
        if  self.__visualizer is None:
            self.__visualizer = dfv.Visualizer(self.__dense_feature_matrix, self.__doc_ids, titles, x_labels, y_labels,
                                               self.__similarity_matrix, dfv.DataReductionType.mds)
        print(f"\nVisualizer details are: {self.__visualizer}")
        self.__visualizer.plotSimilarityMatrixHeatmap()

    def reportBasicStatsOfSimilarityMeasures(self):        
        #sorted_sim_measures = sorted(self.__similarity_matrix_dict.items(), key=lambda item: item[1], reverse=True)
        #print("{0}: {1}".format(key, value))
        def getMedian():
            sorted_sim_measures = sorted(self.__similarity_matrix_dict.items(), key=lambda item: item[1], reverse=True)
            key_value_pairs = [(k, v) for k, v in sorted_sim_measures]
            med_result = key_value_pairs[len(key_value_pairs)//2][0], key_value_pairs[len(key_value_pairs)//2][1]
            return med_result
        max_result = max(self.__similarity_matrix_dict.items(), key=operator.itemgetter(1))
        min_result = min(self.__similarity_matrix_dict.items(), key=operator.itemgetter(1))
        med_result = getMedian()
        ids = [max_result[0], med_result[0], min_result[0]]
        result_values = [max_result[1], med_result[1], min_result[1]]
        result_types = ["Max", "Med", "Min"]
        results = pd.DataFrame({'IDs':ids, 'Metrics':result_values, 'Metric Type':result_types})
        print(f"\nMax, Median and Min Similarity Measures are:\n{results}")
        

    @property
    def raw_corpus(self):
        return self.__corpus

    @property
    def clean_corpus(self):
        return self.__clean_corpus

    @property
    def similarity_matrix_dict(self):
        return self.__similarity_matrix_dict
    
    @property
    def similarity_matrix(self):
        return self.__similarity_matrix


def demo20NewsGroupProblem():
    news_group_20_problem = NewsGroup20Problem()
    raw_corpus = news_group_20_problem.getCorpus()
    clean_corpus = news_group_20_problem.cleanCorpus()
    similarity_measures = news_group_20_problem.computeDocSimilarity()
    news_group_20_problem.displaySortedSimilarityMeasures()
    news_group_20_problem.visualizeDocumentSimilarity()
    news_group_20_problem.plotSimilarityMatrixHeatmap()
    news_group_20_problem.reportBasicStatsOfSimilarityMeasures()

if __name__ == "__main__":
    demo20NewsGroupProblem()
    

