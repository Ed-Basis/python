#! /usr/bin/env python3
import argparse
import json
import logging

from rosette.api import API, DocumentParameters


class Rosette(object):
    """
    Rosette API implementation
    See: https://github.com/rosette-api/python/tree/develop/examples
    """

    api = None

    def __init__(self, key, url, threads=4):
        # Create an API instance
        self.api = API(user_key=key, service_url=url)
        self.api.max_pool_size = threads
        self.api.set_custom_headers('X-RosetteAPI-App', 'python-app')

    def call(self, endpoint, data=None):
        return self.__getattribute__(endpoint)(data)

    def ping(self, data=None):
        return self.api.ping()

    def info(self, data=None):
        return self.api.info()

    def language(self, data):
        logging.debug('Rosette.language: "%s"', data)
        params = DocumentParameters()
        params['content'] = data
        return self.api.language(params)

    def transliteration(self, data):
        logging.debug('Rosette.transliteration: "%s"', data)
        params = DocumentParameters()
        params['content'] = data
        return self.api.transliteration(params)

    def sentences(self, data):
        logging.debug('Rosette.sentences: "%s"', data)
        params = DocumentParameters()
        params['content'] = data
        return self.api.sentences(params)

    def morphology(self, data):
        logging.debug('Rosette.morphology: "%s"', data)
        params = DocumentParameters()
        params['content'] = data
        return self.api.morphology(params)

    def tokens(self, data):
        logging.debug('Rosette.tokens: "%s"', data)
        params = DocumentParameters()
        params['content'] = data
        return self.api.tokens(params)

    def entities(self, data):
        logging.debug('Rosette.entities: "%s"', data)
        params = DocumentParameters()
        params['content'] = data
        params['genre'] = 'social-media'
        return self.api.entities(params)

    def entities_offsets(self, data):
        logging.debug('Rosette.entities_offsets: "%s"', data)
        self.api.set_url_parameter("output", "rosette")
        params = DocumentParameters()
        params['content'] = data
        params['genre'] = 'social-media'
        result = self.api.entities(params)
        self.api.set_url_parameter("output", None)
        return result

    def sentiment(self, data):
        logging.debug('Rosette.sentiment: "%s"', data)
        params = DocumentParameters()
        params['content'] = data
        return self.api.sentiment(params)

    def categories(self, data):
        logging.debug('Rosette.categories: "%s"', data)
        params = DocumentParameters()
        params['content'] = data
        return self.api.categories(params)

    def text_embedding(self, data):
        logging.debug('Rosette.text_embedding: "%s"', data)
        params = DocumentParameters()
        params['content'] = data
        return self.api.text_embedding(params)

    def dependencies(self, data):
        logging.debug('Rosette.dependencies: "%s"', data)
        params = DocumentParameters()
        params['content'] = data
        return self.api.syntax_dependencies(params)

    def relationships(self, data):
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
    return args


if __name__ == '__main__':
    args = parse_command_line()
    rosette = Rosette(args.key, args.url, args.threads)
    result = rosette.call(args.endpoint, args.data)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True).encode("utf8"))
