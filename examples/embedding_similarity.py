# -*- coding: utf-8 -*-
"""
Example code to call Rosette API to get text vectors from a piece of text.
"""
from __future__ import print_function

import argparse

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from examples.Rosette import Rosette
from rosette.api import DocumentParameters


class Embeddings(object):
    rosette = None

    def __init__(self, rosette):
        self.rosette = rosette

    @staticmethod
    def vector_similarity(v1, v2):
        if not isinstance(v1, np.ndarray):
            v1 = np.array(v1)
        if not isinstance(v2, np.ndarray):
            v2 = np.array(v2)
        return np.divide(
            np.dot(v1, v2),
            np.sqrt(np.dot(v1, v1)) * np.sqrt(np.dot(v2, v2))
        )

    def text_embedding(self, data):
        params = DocumentParameters()
        params["content"] = data
        params["language"] = "eng"
        return self.rosette.text_embedding(params)

    def similarity(self, data1, data2):
        e1 = self.rosette.text_embedding(data1)['embedding']
        e2 = self.rosette.text_embedding(data2)['embedding']
        return cosine_similarity([e1, e2])[0,1]
        # return self.vector_similarity(e1, e2)


def embedding_similarity(key, url, data1, data2):
    rosette = Rosette(key, url)
    return Embeddings(rosette).similarity(data1, data2)


def parse_command_line():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Calculates the embedding similarity between two texts')
    parser.add_argument('-k', '--key', help='Rosette API Key')
    parser.add_argument('-u', '--url', help="API URL or 'local'", default='https://api.rosette.com/rest/v1/')
    parser.add_argument('-f', '--file', action='store_true', help='interpret arguments as file names')
    parser.add_argument('arg1', help='first data argument or text file')
    parser.add_argument('arg2', help='second data argument or text file')
    parser.add_argument('-l1', '--lang1', help='language of first item')
    parser.add_argument('-l2', '--lang2', help='language of second item')
    args = parser.parse_args()
    if args.url == 'local':
        args.url = 'http://localhost:8181/rest/v1/'
    return args


if __name__ == '__main__':
    args = parse_command_line()
    if args.file:
        with open(args.arg1) as f:
            data1 = f.read()
        with open(args.arg2) as f:
            data2 = f.read()
    else:
        data1 = args.arg1
        data2 = args.arg2

    result = embedding_similarity(args.key, args.url, data1, data2)

    print(result)
