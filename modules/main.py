import logging

from models.simplemodel.train import simplemodel_train


def get_logger():
    logger = logging.getLogger('Basic model train')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler("rnn2.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


if __name__ == '__main__':
    main_logger = get_logger()
    try:
        simplemodel_train(main_logger)
    except:
        main_logger.exception('Exception while training a model')
