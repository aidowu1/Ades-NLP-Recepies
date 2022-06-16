import enum


class SentenceEmbeddingType(enum.Enum):
    """
    Sentence Embedding Type
    """
    tfidf = 0
    average_word_embedding = 1
    sif_word_embedding = 2
    use = 3
    sbert = 4