from collections import namedtuple

import FeatureExtraction.FeatureVectorizerEnums as e

TfidfConstants = namedtuple("TfidfConstants", "n_features stop_words use_idf")
TFIDF_CONSTANTS = TfidfConstants(n_features=1000, stop_words="english", use_idf=True)

WORD_EMBEDDING_TYPE = {
    e.FeatureVectorizerType.glove: "",
    e.FeatureVectorizerType.fast_text: "",
    e.FeatureVectorizerType.word_2_vec: ""
}

EMBEDDING_ENCODING_TYPE = "cp437"

EMBEDDING_VEC_SIZE = 150
LOG_SKIP_LINES = 10000
EMBEDDING_FILE_PATH = "MODELS/Word2Vec.model"
MAX_N_TOKENS_PER_SENTENCES = 50


