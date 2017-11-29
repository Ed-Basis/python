# -*- coding: utf-8 -*-
"""
Example code to send Rosette API a ping to check its reachability.
"""
from __future__ import print_function

import argparse
import json
import os

from rosette.api import API, RosetteException


def run(key, alt_url):
    """ Run the example """
    # Create an API instance
    api = API(user_key=key, service_url=alt_url)

    try:
        return api.ping()
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
