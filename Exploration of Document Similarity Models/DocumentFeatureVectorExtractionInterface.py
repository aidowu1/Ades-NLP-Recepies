
# coding: utf-8

# ## Document Feature Vector Extraction and Similarity Measure Component

# ## imports

# In[1]:

import abc


# In[2]:

class IFeatureVectorExtraction(abc.ABC):
    """
    Interface used for the abstraction of corpus Feature Vector and document Similarity Measure
    """
    @abc.abstractmethod
    def createFeatureMatrix(corpus):
        """
        Creates a collection of Feature vectors for corpus or pair of corpora
        """
        raise NotImplementedError("Need to implement abstrct method: createFeatureVector()..")
    
    @abc.abstractmethod
    def measureSimilarity(doc_vec_1, doc_vec_2):
        """
        Computes the similrity measure between 2 document vectors
        """
        raise NotImplementedError("Need to implement abstrct method: measureSimilarity()..")
    
    


# In[ ]:



