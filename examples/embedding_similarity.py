# -*- coding: utf-8 -*-
"""
Example code to call Rosette API to get text vectors from a piece of text.
"""
from __future__ import print_function

import argparse
import logging

from examples.rosette import Rosette
from examples.embeddings import Embeddings


def embedding_similarity(key, url, arg1, arg2):
    rosette = Rosette(key, url)
    return Embeddings(rosette).similarity(arg1, arg2)


def parse_command_line():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-7s %(message)s')
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
