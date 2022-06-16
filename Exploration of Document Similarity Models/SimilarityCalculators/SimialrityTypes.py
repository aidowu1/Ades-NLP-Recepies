import enum

class SimilarityCalcTypes(enum.Enum):
    """
    Similarity calculation types
    """
    cosine = 0
    euclidian = 1
    jaccard = 2
    pearson = 3
