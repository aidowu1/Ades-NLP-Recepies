from typing import List, Iterable

import numpy as np
import time
from tqdm import tqdm
import gensim
import logging

import FeatureExtraction.Constants as c

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

class TrainedWordEmbeddingWithFse(object):
    """
    Component used to train word embedding component (with Smooth Inverse Frequency - SIF) using the FSE library
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