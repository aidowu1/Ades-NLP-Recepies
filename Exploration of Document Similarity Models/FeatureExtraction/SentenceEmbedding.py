import logging
import math
import numpy as np
from fse import Vectors
from typing import Tuple, List, Dict
from fse.models import uSIF, SIF, Average
from fse import IndexedList
import pandas as pd

import FeatureExtraction.SentenceEmbeddingTypes as se
import NLPProcessing.NLPEngineComponent as nlp
import Constants as c
import Utils as u

logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)

class SentenceEmbedding(object):
    """
    Component used to create the sentence embedding using any one of the following:
        1) Average of the word embedding vectors per sentence
        2) Apply Smooth Inverse Frequency to the word embeddings
    """
    def __init__(
            self,
            corpus: List[str],
            query_docs: List[str] = None,
            sentence_embedding_type: se.SentenceEmbeddingType = se.SentenceEmbeddingType.average_word_embedding):
        """
        Constructor
        :param corpus:
        """
        try:
            self.__raw_corpus = corpus
            self.__raw_query_docs = query_docs
            self.__n_corpus_docs = len(self.__raw_corpus)
            self.__n_query_docs = None
            self.__clean_corpus_docs = self.__cleanCorpus(self.__raw_corpus)
            if self.__raw_query_docs:
                self.__clean_query_docs = self.__cleanCorpus(self.__raw_query_docs)
                self.__n_query_docs = len(self.__raw_query_docs)
                self.__super_set_corpus = self.__clean_corpus_docs + self.__clean_query_docs
            else:
                self.__super_set_corpus = self.__clean_corpus_docs
            self.__super_set_corpus_sentences = None
            assert sentence_embedding_type.name in [se.SentenceEmbeddingType.average_word_embedding.name,
                                                    se.SentenceEmbeddingType.sif_word_embedding.name]
            self.__sentence_embedding_type = sentence_embedding_type
            self.__word_embedding = Vectors.from_pretrained(c.GLOVE_EMBEDDING_NAME)
            self.__corpus_embeddings = None
            self.__query_embeddings = None
            self.__similarity_matrix_self = None
            self.__model = None
            self.__sim_scores = None
            self.__sentence_ids = []
            self.__query_ids = []
            self.__corpus_sentences = []
            self.__query_sentences = []
            self.__scores = []
        except Exception as ex:
            logging.info(f"Incorrect sentence embedding type: '{sentence_embedding_type.name}' specified!!")
            logging.info(f"Error: {str(ex)}")

    def __call__(self):
        """
        Callable use to ONLY train the embedding model
        :return: None
        """
        self.__trainModel()

    def __cleanCorpus(self, docs: List[str]) -> List[List[str]]:
        """
        Applies NLP pre-processing to the raw corpus
        :param docs: Documents/sentences
        :return: Cleans documents/sentences
        """
        nlp_engine = nlp.NLPEngine(docs)
        preprocessed_docs = nlp_engine.preprocessDocs()
        return preprocessed_docs

    def __trainModel(self):
        """
        Apply SIF using the FSE library to train the sentence embedding model
        """
        self.__super_set_corpus_sentences = IndexedList(self.__super_set_corpus)
        if self.__sentence_embedding_type is se.SentenceEmbeddingType.average_word_embedding:
            self.__model = Average(self.__word_embedding, workers=1, lang_freq="en")
        elif self.__sentence_embedding_type is se.SentenceEmbeddingType.sif_word_embedding:
            self.__model = SIF(self.__word_embedding, workers=1, lang_freq="en")
        self.__model.train(self.__super_set_corpus_sentences)
        if self.__n_query_docs:
            self.__query_embeddings = [self.__model.sv[self.__n_corpus_docs + i] for i in range(self.__n_query_docs)]
        self.__corpus_embeddings = [self.__model.sv[i] for i in range(self.__n_corpus_docs)]

    def __computeSelfSimilarities(self, doc_pair_indices: Tuple[int, int]) -> Dict[str, float]:
        """
        Computes the sentence embedding similarity metrics for query docs derived from
        the coprpus (itself)
        :param doc_pair_indices: sentence pairs (i.e. b/w query and corpus sentences)
        :return: Similarity scores
        """
        sims = {}
        sims_normalized = {}
        self.__similarity_matrix_self = np.zeros((self.__n_corpus_docs, self.__n_corpus_docs))
        for i, j in doc_pair_indices:
            key = "{0}_{1}".format(i, j)
            cosine_scores = round(self.__model.sv.similarity(i, j), 4)
            scores_normalized = 1.0 - np.arccos(cosine_scores) / math.pi
            sims[key] = cosine_scores
            sims_normalized[key] = round(scores_normalized, 4)
            self.__similarity_matrix_self[i, j] = cosine_scores
        return sims

    def __computeSimilarities(self, top_k: int = None) -> pd.DataFrame:
        """
        Computes the sentence embedding similarity metrics between query docs and
        corppus docs
        :param top_k: Top k similarity scores
        :return: Similarity scores
        """
        sims = {}
        sims_normalized = {}
        for i in range(self.__n_query_docs):
            for j in range(self.__n_corpus_docs):
                key = "{0}_{1}".format(i, j)
                cosine_scores = self.__model.sv.similarity(i + self.__n_corpus_docs, j)
                scores_normalized = 1.0 - np.arccos(cosine_scores) / math.pi
                sims[key] = round(cosine_scores, 4)
                sims_normalized[key] = round(scores_normalized, 4)
            sims_sorted = self.GetTopResults(sims, top_k)
            for k, v in sims_sorted.items():
                ids = k.split("_")
                query_id = int(ids[0])
                corpus_id = int(ids[1])
                self.__query_ids.append(query_id  + self.__n_corpus_docs)
                self.__sentence_ids.append(corpus_id)
                self.__corpus_sentences.append(self.__raw_corpus[corpus_id])
                self.__query_sentences.append(self.__raw_query_docs[query_id])
                self.__scores.append(v)
        self.__results = {
            "query_sentence_id": self.__query_ids,
            "corpus_sentence_id": self.__sentence_ids,
            "corpus_sentences": self.__corpus_sentences,
            "query_sentences": self.__query_sentences,
            "scores": self.__scores
        }
        self.__results_df = pd.DataFrame(data=self.__results)
        self.__results_df.drop_duplicates(subset=["query_sentence_id", "corpus_sentence_id"], inplace=True)
        self.__results_df.sort_values(by=['scores'], ascending=False, inplace=True)
        return self.__results_df

    def GetTopResults(self, sims: Dict[str, float], top_k: int = None):
        """
        Sorts the similarity results
        :param sims: Similarity results
        :param top_k: Number of Top K
        :return: Top K results
        """
        if top_k:
            sims_sorted = u.Helpers.sliceDict({
                k: v for k, v in sorted(sims.items(), key=lambda x: x[1], reverse=True)
            }, n_samples=top_k)
            return sims_sorted
        else:
            return sims

    def computeSelfSimilarities(self) -> Dict[str, float]:
        """
        Computes the sentence similarities
        :return:
        """
        self.__trainModel()
        sentence_pairs = u.Helpers.createLowerTriangularMatrixOfPairs(n_sentences=self.__n_corpus_docs)
        self.__sim_scores = self.__computeSelfSimilarities(sentence_pairs)
        return self.__sim_scores

    def computeSimilarities(self) -> pd.DataFrame:
        """
        Computes the sentence similarities
        :return:
        """
        self.__trainModel()
        result_df = self.__computeSimilarities()
        return result_df

    @property
    def corpus_embeddings(self) -> List[np.ndarray]:
        """
        Getter property for corpus embeddings
        :return: Sentence embeddings
        """
        return self.__corpus_embeddings

    @property
    def query_embeddings(self) -> List[np.ndarray]:
        """
        Getter property for sentence embeddings
        :return: Sentence embeddings
        """
        return self.__query_embeddings

    @property
    def similarity_scores(self) -> List[float]:
        """
        Getter property for sentence similarity scores
        :return: Similarity scores
        """
        return self.__scores

    @property
    def similarity_matrix(self) -> np.ndarray:
        """
        Getter property for the similarity matrix (non-self-similarity measure)
        :return: Similarity matrix
        """
        similarity_matrix_cols = ["query_sentence_id", "corpus_sentence_id", "scores"]
        score_df = self.__results_df[similarity_matrix_cols]
        similarity_matrix = score_df.pivot(
            index="query_sentence_id",
            columns="corpus_sentence_id",
            values="scores"
        ).to_numpy()
        return similarity_matrix

    @property
    def similarity_matrix_self(self) -> np.ndarray:
        """
        Getter property for the similarity matrix (i.e. self-similarity measure)
        :return: Similarity matrix
        """
        return self.__similarity_matrix_self

    @staticmethod
    def createEmbeddingForDocs(
            sentences: List[str],
            sentence_embedding_type: se.SentenceEmbeddingType
    ) -> List[float]:
        """
        Create the average and SIF sentence embeddings
        :param sentence: Sentence
        :param sentence_embedding_type: Sentence type
        :return: Sentence embedding
        """
        model = None
        n_sentences = len(sentences)
        word_embedding = Vectors.from_pretrained(c.GLOVE_EMBEDDING_NAME)
        if sentence_embedding_type is se.SentenceEmbeddingType.average_word_embedding:
            model = Average(word_embedding, workers=1, lang_freq="en")
        elif sentence_embedding_type is se.SentenceEmbeddingType.sif_word_embedding:
            model = SIF(word_embedding, workers=1, lang_freq="en")
        sentences_tokenized = [s.split() for s in sentences]
        indexed_sentences = IndexedList(sentences_tokenized)
        model.train(indexed_sentences)
        corpus_embeddings = [model.sv[i].tolist() for i in range(n_sentences)]
        return corpus_embeddings





