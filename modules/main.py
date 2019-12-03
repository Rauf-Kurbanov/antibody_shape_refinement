import logging

from models.simplemodel.train import simplemodel_train
from models.simplemodel.train_coordinates import simplemodel_coord_train

from config_loader import load_config

def get_logger():

    logger = logging.getLogger('Basic model (coordinates) train')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    config = load_config()
    file_handler = logging.FileHandler(config["DEBUG_LOG"])
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


if __name__ == '__main__':
    main_logger = get_logger()
    try:
        simplemodel_coord_train(main_logger)
        # simplemodel_train(main_logger)
    except:
        main_logger.exception('Exception while training a model')
