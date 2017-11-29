# -*- coding: utf-8 -*-
"""
Example code to call Rosette API to determine the language of a piece of text.
"""
from __future__ import print_function

import argparse
import json
import os

from rosette.api import API, DocumentParameters, RosetteException


def run(key, alt_url):
    """ Run the example """
    # Create an API instance
    api = API(user_key=key, service_url=alt_url)

    language_data = "Por favor Señorita."
    params = DocumentParameters()
    params["content"] = language_data
    api.set_custom_headers("X-RosetteAPI-App", "python-app")
    try:
        return api.language(params)
    except RosetteException as exception:
        print(exception)


PARSER = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Calls the ' +
                                             os.path.splitext(os.path.basename(__file__))[0] + ' endpoint')
PARSER.add_argument('-k', '--key', help='Rosette API Key')
PARSER.add_argument('-u', '--url', help="Alternative API URL",
                    default='https://api.rosette.com/rest/v1/')

if __name__ == '__main__':
    ARGS = PARSER.parse_args()
    RESULT = run(ARGS.key, ARGS.url)
    print(json.dumps(RESULT, indent=2, ensure_ascii=False, sort_keys=True).encode("utf8"))
