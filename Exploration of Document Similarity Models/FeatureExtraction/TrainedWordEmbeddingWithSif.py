from typing import List, Iterable
import numpy as np
import time
from tqdm import tqdm
import gensim
import logging

import FeatureExtraction.Constants as c

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

class TrainedWordEmbeddingWithSif(object):
    """
    Component used to train word embedding component (with Smooth Inverse Frequency -SIF)
    """
    def __init__(
            self,
            corpus: List[str]
                 ):
        """
        Constructor
        :param corpus:
        """
        self.__corpus = corpus
        self.__sentences = []
        self.__model = None
        self.__alpha = None

    def trainModel(self):
        """
        Trains the model to create word embeddings
        :return:
        """
        documents = self.createCorpusDataIterator()
        self.__model = gensim.models.Word2Vec(
            sentences=documents,
            vector_size=c.EMBEDDING_VEC_SIZE,
            window=10,
            min_count=2,
            workers=10)
        print(f"Saving the trained IB Chat embedding to the path: {c.EMBEDDING_FILE_PATH}..")
        self.__model.save(c.EMBEDDING_FILE_PATH)

    @staticmethod
    def getModelFromCache():
        """
        Gets a pre-generated model from the cache
        :return:
        """
        print(f"Loads a pre-trained word embedding model for the path: {c.EMBEDDING_FILE_PATH}")
        model = gensim.models.Word2Vec.load(c.EMBEDDING_FILE_PATH)
        return model

    def createCorpusDataIterator(self, log_skip_lines=c.LOG_SKIP_LINES) -> Iterable:
        """
        Create data iterator for the corpus
        :param log_skip_lines: Logging skip lines
        :return: Corpus iterator
        """
        for i, line in enumerate(self.__sentences):
            if (i%log_skip_lines == 0):
                print(f"Read {i} sentences")
            yield gensim.utils.simple_preprocess(line, max_len=c.MAX_N_TOKENS_PER_SENTENCES)

    def createSifEmbeddings(self) -> np.ndarray:
        """
        Creates SIF (Smooth Inverse Embeddings)
        :return: SIF sentence embedding of shape number of sentences by dimension of the embedding vector
        """
        print("Starting the SIF run..")
        start_time = time.time()
        v_lookup = self.__model.wv.vocab
        vectors = self.__model.wv
        size = self.__model.vector_size

        Z = 0
        for k in v_lookup:
            Z += v_lookup[k].count

        output = []

        for s in tqdm(self.__sentences):
            count = 0
            v = np.zeros(size, dtype=np.float32)
            for w in s:
                if w in s:
                    if w in v_lookup:
                        v +=(self.__alpha / (self.__alpha + (v_lookup[w].count / Z))) * vectors[w]

            if count > 0:
                for i in range(size):
                    v[i] *=1/count
            output.append(v)
        sentence_vectors = np.vstack(output).astype(np.float32)
        print(f"The size of the sentence vectors are: {sentence_vectors.shape}\n")
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 2)
        print(f"End of SIF embedding computation, the run took {elapsed_time} seconds..")
        return sentence_vectors

