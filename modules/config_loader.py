import json


def load_config():
    with open('config.json') as json_data_file:
        data = json.load(json_data_file)
    return data
