import logging
import argparse

from models.simplemodel.train import simplemodel_train
from models.simplemodel_coordinates.train_coordinates import simplemodel_coord_train

from config_loader import load_config

MODELS_NAMES = ['simple', 'simple_coordinates']


def get_logger():
    logger = logging.getLogger('Basic model (coordinates) train')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    config = load_config()
    file_handler = logging.FileHandler(config["DEBUG_LOG"])
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='simple',
                        choices=MODELS_NAMES,
                        help='Which model to run')
    parser.add_argument('--use_backup', type=bool, default=False,
                        help='Use a backup or not')
    return parser


if __name__ == '__main__':
    main_logger = get_logger()
    main_parser = get_parser()
    args = vars(main_parser.parse_known_args()[0])
    use_backup = args['use_backup']
    choice_model = args['model']
    try:
        if choice_model == 'simple':
            simplemodel_train(main_logger, use_backup=use_backup)
        elif choice_model == 'simple_coordinates':
            simplemodel_coord_train(main_logger, use_backup=use_backup)
    except:
        main_logger.exception('Exception while training a model')
