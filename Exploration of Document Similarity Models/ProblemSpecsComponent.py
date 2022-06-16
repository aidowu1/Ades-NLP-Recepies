#!/usr/bin/env python
# coding: utf-8

# ## Document Similarity Problem Specifications

# ## Imports

# In[3]:


import pandas as pd
import numpy as np
import os
from yellowbrick.text import TSNEVisualizer
from yellowbrick.datasets import load_hobbies

import GenericDataSerializerComponent as s
import ProblemSpecificationInterface as p
import NLPEngineComponent as nlp
import TFIDFDocmentVectorExtractor as tf_idf
import DocumentFeatureVisualization as dfv


# ### Specify contants and variables

# In[5]:


DATA_ROOT_PATH = 'Data'
TALENT36T_VACANCY_ID_36_DATA = {
                    'employee':os.path.join(DATA_ROOT_PATH, 'Employee_doc_store.csv'),
                    'employer':os.path.join(DATA_ROOT_PATH, 'Employer_doc_store.csv')
}
 
SAMPLE_PROBLEM_DATASET_PATHS = {
                    'Toy_Problem':os.path.join(DATA_ROOT_PATH, 'Book_List_Dataset.pl'),
                    'Talent36T':TALENT36T_VACANCY_ID_36_DATA    
                }


# In[8]:


class ToyProblem(p.IProblemSpec):
    def __init__(self):
        self.__num_docs = -1
        self.__corpus = None
        self.__clean_corpus = None
        self.__nlp_engine = None
        self.__sparse_feature_matrix = None
        self.__dense_feature_matrix = None
        self.__similarity_matrix = {}

    
    def getCorpus(self):
        """
        Get the corpus data for this problem
        """
        self.__corpus = s.GenericDataSerializer.deSerializeCache(SAMPLE_PROBLEM_DATASET_PATHS['Toy_Problem'])
        self.__num_docs = len(self.__corpus)
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
        print(f"Shape of Feature Matrix is: {self.__dense_feature_matrix.shape}")
        doc_pair_indices = self.createLowerTriangularMatrixOfPairs()
        for ind in doc_pair_indices:
            doc_1_ind, doc_2_ind = ind
            similarity = tfidf_extractor.measureSimilarity(self.__dense_feature_matrix[doc_1_ind], self.__dense_feature_matrix[doc_2_ind])
            key = "{0}_{1}".format(doc_1_ind, doc_2_ind)
            self.__similarity_matrix[key] = similarity
        return self.__similarity_matrix         
    
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
        for key, value in sorted(self.__similarity_matrix.items(), key=lambda item: item[1], reverse=True):
            print("{0}: {1}".format(key, value))

    def visualizeDocumentSimilarity(self):
        """
        Provides the visualization of similarity between the Documents in a corpus
        Each document is labelled with a ID (a sequence number)
        """
        # X = self.__dense_feature_matrix
        # y = np.array(range(self.__num_docs))
        # tsne = TSNEVisualizer()
        # tsne.fit(X, y)
        # tsne.show()
        title = "Plot of 'Toy Example' Feature Matrix points reduced to 2D by MDS"
        doc_ids = list(range(self.__num_docs))
        visualizer = dfv.Visualizer(self.__dense_feature_matrix, title, doc_ids, dfv.DataReductionType.mds)
        print(f"\nVisualizer details are: {visualizer}")
        visualizer.plot2DRepresentation()

    @property
    def corpus(self):
        return self.__corpus

    @property
    def clean_corpus(self):
        return self.__clean_corpus

    @property
    def similarity_measures(self):
        return self.__similarity_measures
        
class AiraProblem(p.IProblemSpec):
    def getCorpus(self):
        """
        Get the corpus data for this problem
        """
        self.__df_employee = pd.read_csv(TALENT36T_VACANCY_ID_36_DATA['employee'], encoding="ISO-8859-1")
        self.__df_employer = pd.read_csv(TALENT36T_VACANCY_ID_36_DATA['employer'], encoding="ISO-8859-1")
        # print("self.__df_employee:\n{}".format(self.__df_employee.head(3)))
        # print("self.__df_employer:\n{}".format(self.__df_employer))
        employer_columns_to_retain = ['key_skills_doc', 'CV_details_doc', 'employer_user_name']
        self.__df_employer = self.__df_employer[employer_columns_to_retain]
        self.__df_employer = self.__df_employer.rename(columns=
            {
                'key_skills_doc':'key_skills_doc',
                'CV_details_doc':'CV_details_doc', 
                'employer_user_name':'username'
            })
        # print("self.__df_employer:\n{}".format(self.__df_employer))
        self.__df_corpus = pd.concat([self.__df_employee, self.__df_employer]).reset_index()
        corpus_columns_to_retain = ['key_skills_doc', 'CV_details_doc', 'username']
        self.__df_corpus = self.__df_corpus[corpus_columns_to_retain]
        # print("self.__df_corpus:\n{}".format(self.__df_corpus))
        self.__num_docs = self.__df_corpus.shape[0]
        print("Number of rows for the Toy Problem data are: {}".format(self.__num_docs))
        return self.__df_corpus

    @property
    def corpus(self):
        return self.__df_corpus

# In[ ]:

def demoToyProblem():
    toy_problem = ToyProblem()
    raw_corpus = toy_problem.getCorpus()
    clean_corpus = toy_problem.cleanCorpus()
    similarity_measures = toy_problem.computeDocSimilarity()
    toy_problem.displaySortedSimilarityMeasures()
    toy_problem.visualizeDocumentSimilarity()

def demoAiraProblem():
    aira_problem = AiraProblem()
    corpus = aira_problem.getCorpus()
    print("aira_problem.corpus:\n{}".format(aira_problem.corpus))

if __name__ == "__main__":
    demoToyProblem()



