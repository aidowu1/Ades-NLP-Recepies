"""
Toy Problem
Description:    
This is a contrived example which demonstates how to compute the similarity between documents
in a corpus which contains 24 documents (Book titles). The solution to this problem demonstates the following
steps:
    - Getting the source Toy Problem corpus data via the GenericDataSerializerComponent module 
    - Text data NLP pre-processing using the NLPEngineComponent module
    - Feature extration using TFIDF vectorization provided by the TFIDFDocumentVectorExtractor module
    - Document similarity computation using Cosine Similarity measure in the TFIDFDocumentVectorExtractor module
    - 2D Visualization of the documents using Multi-Dimensional Scaling (MDS) provided in the DocumentFeatureVisualization module 

Data/reference was based on the blog provided by https://shravan-kuchkula.github.io/nlp/document_similarity/#
"""

import pandas as pd
import numpy as np
import os
import GenericDataSerializerComponent as s
import ProblemSpecificationInterface as p
import NLPEngineComponent as nlp
import TFIDFDocmentVectorExtractor as tf_idf
import DocumentFeatureVisualization as dfv


# Specify contants and variables
DATA_ROOT_PATH = 'Data'
 
SAMPLE_PROBLEM_DATASET_PATHS = {
                    'Toy_Problem':os.path.join(DATA_ROOT_PATH, 'Book_List_Dataset.pl'),
                                }


class ToyProblem(p.IProblemSpec):
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
    
    def getCorpus(self):
        """
        Get the corpus data for this problem
        """
        self.__corpus = s.GenericDataSerializer.deSerializeCache(SAMPLE_PROBLEM_DATASET_PATHS['Toy_Problem'])
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
        Create triangular matrix indices pairs for the similarity measure
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
        "2D-Similarity-Plot":"Plot of Feature Matrix points reduced to 2D by MDS for 'Book Titles Toy' Problem",
        "Similarity-Heatmap": "Plot of Similarity Matrix Heatmap for 'Book Titles Toy' Problem"
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
            self.__visualizer = dfv.Visualizer(self.__dense_feature_matrix, self.__doc_ids, 
                    titles, x_labels, y_labels, self.__similarity_matrix, dfv.DataReductionMethod.mds)
        print(f"\nVisualizer details are: {self.__visualizer}")
        self.__visualizer.plot2DRepresentation()

    def plotSimilarityMatrixHeatmap(self):
        """
        Plots the Heatmap of the similarity matrix scores
        """
        titles = { 
        "2D-Similarity-Plot":"Plot of Feature Matrix points reduced to 2D by MDS for 'Book Titles Toy' Problem",
        "Similarity-Heatmap": "Plot of Similarity Matrix Heatmap for 'Book Titles Toy' Problem"
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
                                self.__similarity_matrix, dfv.DataReductionMethod.mds)
        print(f"\nVisualizer details are: {self.__visualizer}")
        self.__visualizer.plotSimilarityMatrixHeatmap()

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

def demoToyProblem():
    toy_problem = ToyProblem()
    raw_corpus = toy_problem.getCorpus()
    clean_corpus = toy_problem.cleanCorpus()
    similarity_measures = toy_problem.computeDocSimilarity()
    toy_problem.displaySortedSimilarityMeasures()
    toy_problem.visualizeDocumentSimilarity()
    toy_problem.plotSimilarityMatrixHeatmap()

if __name__ == "__main__":
    demoToyProblem()


