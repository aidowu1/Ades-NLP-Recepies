import enum

class FeatureVectorizerType(enum.Enum):
    """
    Enumerations for the feature extraction types
    """
    tfidf = 0
    word_2_vec = 1
    glove = 2
    fast_text = 3
    sif_with_glove = 4

