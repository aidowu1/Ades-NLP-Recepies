import abc

class IProblemSpec(abc.ABC):
    """
    Interface used for the abstraction of Problem Specification
    """
    @abc.abstractmethod
    def getCorpus(self):
        """
        Creates a collection of Feature vectors for corpus or pair of corpora
        """
        raise NotImplementedError("Need to implement abstract method: getCorpus()..")

    @abc.abstractmethod
    def cleanCorpus(self):
        """
        Cleans/normalises the corpus
        """
        raise NotImplementedError("Need to implement abstract method: cleanCorpus()..")

    @abc.abstractmethod
    def computeDocSimilarity(self):
        """
        Computes the similarity between documents in corpus
        """
        raise NotImplementedError("Need to implement abstract method: computeDocSimilarity()..")

    @abc.abstractmethod
    def visualizeDocumentSimilarity(self):
        """
        Computes the Visualization of documents in a corpus
        """
        raise NotImplementedError("Need to implement abstract method: visualizeDocumentSimilarity()..")

    @abc.abstractmethod
    def plotSimilarityMatrixHeatmap(self):
        """
        Computes the Visualization Document Similarity Matrix Heatmap
        """
        raise NotImplementedError("Need to implement abstract method: plotSimilarityMatrixHeatmap()..")
