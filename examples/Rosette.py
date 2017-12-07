#! /usr/bin/env python3

from __future__ import print_function

import argparse
import json
import logging

import numpy as np

from rosette.api import API, DocumentParameters, RosetteException


class Rosette(object):
    """
    Simplified encapsulation of Rosette API (api.py)
    TODO: process lang argument
    TODO: Accept URI data source
    """

    api = None

    def __init__(self, key=None, url='https://api.rosette.com/rest/v1/', threads=4):
        # Create an API instance
        self.api = API(user_key=key, service_url=url)
        self.api.max_pool_size = threads
        self.api.set_custom_headers('X-RosetteAPI-App', 'python-app')

    def call(self, endpoint, data=None, lang=None):
        """
        Calls specified Rosette API endpoint
        :param endpoint: endpoint to call, e.g. 'entities'
        :param data: flat text data to process
        :param lang: language of text (optional)
        :return:
        """
        return self.__getattribute__(endpoint)(data=data, lang=lang)

    def ping(self, data=None, lang=None):
        return self.api.ping()

    def info(self, data=None, lang=None):
        return self.api.info()

    def language(self, data, lang=None):
        logging.debug('Rosette.language: "%s"', data)
        params = DocumentParameters()
        params['content'] = data
        return self.api.language(params)

    def transliteration(self, data, lang=None):
        logging.debug('Rosette.transliteration: "%s"', data)
        params = DocumentParameters()
        params['content'] = data
        return self.api.transliteration(params)

    def sentences(self, data, lang=None):
        logging.debug('Rosette.sentences: "%s"', data)
        params = DocumentParameters()
        params['content'] = data
        return self.api.sentences(params)

    def morphology(self, data, lang=None):
        logging.debug('Rosette.morphology: "%s"', data)
        params = DocumentParameters()
        params['content'] = data
        return self.api.morphology(params)

    def tokens(self, data, lang=None):
        logging.debug('Rosette.tokens: "%s"', data)
        params = DocumentParameters()
        params['content'] = data
        return self.api.tokens(params)

    def entities(self, data, lang=None):
        logging.debug('Rosette.entities: "%s"', data)
        params = DocumentParameters()
        params['content'] = data
        params['genre'] = 'social-media'
        return self.api.entities(params)

    def entities_offsets(self, data, lang=None):
        logging.debug('Rosette.entities_offsets: "%s"', data)
        self.api.set_url_parameter("output", "rosette")
        params = DocumentParameters()
        params['content'] = data
        params['genre'] = 'social-media'
        result = self.api.entities(params)
        self.api.set_url_parameter("output", None)
        return result

    def sentiment(self, data, lang=None):
        logging.debug('Rosette.sentiment: "%s"', data)
        params = DocumentParameters()
        params['content'] = data
        return self.api.sentiment(params)

    def categories(self, data, lang=None):
        logging.debug('Rosette.categories: "%s"', data)
        params = DocumentParameters()
        params['content'] = data
        return self.api.categories(params)

    def text_embedding(self, data, lang=None):
        """
        Returns the text embedding vector for the specified data as a numpy array.
        :param data: textual input
        :param lang: language of text (optional)
        :return: numpy array of decimal values
        """
        logging.debug('Rosette.text_embedding: "%s"', data)
        params = DocumentParameters()
        params['content'] = data
        if (lang):
            params['language'] = lang
        try:
            return np.asarray(self.api.text_embedding(params)['embedding'])
        except RosetteException as e:
            if e.status == 'tooMuchData':
                logging.info("%s: %s. Truncating data.", e.status, e.message)
                return self.text_embedding(data[:50000])
            elif e.status == 'unsupportedLanguage':
                if 'xxx' in e.message:
                    logging.debug("%s: %s. Returning zeros.", e.status, e.message)
                else:
                    logging.info("%s: %s. Processing as English: '%s'", e.status, e.message, data[:30])
                    return self.text_embedding(data, 'eng')
            elif e.status == 'notEnoughData':
                logging.debug("%s: %s. Returning zeros.", e.status, e.message)
            elif logging.getLogger().getEffectiveLevel() >= logging.INFO:
                logging.info("%s: %s. Returning zeros: '%s'", e.status, e.message, data[:30])
            else:
                raise
            return np.zeros(300)

    def dependencies(self, data, lang=None):
        logging.debug('Rosette.dependencies: "%s"', data)
        params = DocumentParameters()
        params['content'] = data
        return self.api.syntax_dependencies(params)

    def relationships(self, data, lang=None):
        logging.debug('Rosette.relationships: "%s"', data)
        params = DocumentParameters()
        params['content'] = data
        self.api.set_option('accuracyMode', 'PRECISION')
        result = self.api.relationships(params)
        self.api.set_option('accuracyMode', None)
        return result


def parse_command_line():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Calculates the embedding similarity between two texts')
    parser.add_argument('endpoint', help='endpoint to call')
    parser.add_argument('-k', '--key', help='Rosette API Key')
    parser.add_argument('-u', '--url', help="API URL or 'local'", default='https://api.rosette.com/rest/v1/')
    parser.add_argument('-d', '--data', help='data to process')
    parser.add_argument('-t', '--threads', help='max number of threads')
    args = parser.parse_args()
    if args.url == 'local':
        args.url = 'http://localhost:8181/rest/v1/'
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-7s %(message)s')
    return args


if __name__ == '__main__':
    args = parse_command_line()
    rosette = Rosette(args.key, args.url, args.threads)
    result = rosette.call(args.endpoint, args.data)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True).encode("utf8"))
