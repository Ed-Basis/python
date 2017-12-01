#! /usr/bin/env python3
import logging
import sys

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Embeddings(object):
    """
    Examples of text embedding uses
    TODO: Accept URI inputs
    """
    rosette = None

    def __init__(self, rosette):
        self.rosette = rosette

    def similarity(self, data1, data2):
        """
        Calculates the text embedding similarity of two text arguments.
        :param data1: first text argument
        :param data2: second text argument
        :return: [-1,1] similarity, where 0 is no similarity and 1 is perfect similarity
        """
        e1 = self.rosette.text_embedding(data1)['embedding']
        e2 = self.rosette.text_embedding(data2)['embedding']
        similarity = cosine_similarity([e1, e2])
        return similarity[0, 1]

    def coherence(self, *data):
        """
        Calculates the text embedding similarity of N text arguments.
        Result is the average of pairwise similarities.
        :param data:
        :return: [-1,1] similarity, where 0 is no similarity and 1 is perfect similarity
        """
        embeddings = [self.rosette.text_embedding(d) for d in data]
        X = [e['embedding'] for e in embeddings]
        similarity = cosine_similarity(X)
        # exclude identity values from average
        weights = np.ones_like(similarity) - np.identity(len(similarity))
        return np.average(similarity, weights=weights)

    def mean(self, *data):
        """
        Calculates the mean (average) text embedding vector over a array of texts.
        :param data: texts
        :return: mean text embedding vector
        """
        embeddings = [self.rosette.text_embedding(d, 'eng')['embedding'] for d in data]
        result = np.mean(embeddings, axis=0, dtype=np.float64)
        return result

    def match(self, data, category, lang=None):
        """
        Calculates the text embedding similarity of a text argument to a text embedding.
        :param data: text argument
        :param category: text embedding argument
        :return: [-1,1] similarity, where 0 is no similarity and 1 is perfect similarity
        """
        e1 = self.rosette.text_embedding(data, lang)['embedding']
        similarity = cosine_similarity([e1, category])
        return similarity[0, 1]


def main(args):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-7s %(message)s')
    pass


if __name__ == '__main__':
    main(sys.argv[1:])
