import json
import pickle
import torch

from embedding import get_angles, get_embeddings, get_coordinates

if __name__ == '__main__':
    with open('../config.json') as json_data_file:
        config = json.load(json_data_file)
        model = torch.load(config["PATH_TO_PRETRAINED_EMBEDDING_MODEL"])
        seq = get_embeddings(config["PATH_TO_PRETRAINED_EMBEDDING_MODEL"], config["PATH_TO_SEQ_DATA"])
        angles = get_angles(config["PATH_TO_ANGLES_DATA"])
        coord = get_coordinates(config["PATH_TO_COORD_DATA"])
        with open(config["PATH_TO_SEQ_EMBEDDED"], "wb") as seq_file:
            pickle.dump(seq, seq_file, protocol=pickle.HIGHEST_PROTOCOL)
        with open(config["PATH_TO_ANGLES_EMBEDDED"], "wb") as angles_file:
            pickle.dump(angles, angles_file, protocol=pickle.HIGHEST_PROTOCOL)
        with open(config["PATH_TO_COORD_EMBEDDED"], "wb") as coord_file:
            pickle.dump(coord, coord_file, protocol=pickle.HIGHEST_PROTOCOL)
