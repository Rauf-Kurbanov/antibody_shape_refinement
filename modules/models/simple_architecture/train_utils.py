import os

import torch
import torch.nn.utils.rnn as rnn_utils
import wandb
from tqdm import tqdm
from metrics.metric import angle_metrics, coordinate_metrics
from torch import autograd


def train_model(train_dataloader, val_dataloader, model, model_name, loss, optimizer, scheduler, num_epochs, logger,
                device,
                config,
                metrics_logger,
                model_backup_path=None, start_epoch=0, num_epoch_before_backup=100, debug=False):
    for epoch in range(start_epoch, num_epochs):
        logger.info(f'Epoch {epoch}/{num_epochs - 1}')

        for phase in ['train', 'val']:
            if phase == 'train':
                dataloader = train_dataloader
                scheduler.step()
                model.train()
            else:
                dataloader = val_dataloader
                model.eval()

            running_loss = 0.

            for inputs, targets, lengths in tqdm(dataloader):
                inputs = inputs.to(device)

                targets = rnn_utils.pack_padded_sequence(targets, lengths, batch_first=True)
                targets, _ = rnn_utils.pad_packed_sequence(targets, batch_first=True)
                targets = targets.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    preds, lengths, hiddens = model(inputs, lengths)

                    loss_value = loss(preds, targets, lengths)

                    if phase == 'train':
                        # with autograd.detect_anomaly():

                        loss_value.backward()
                        optimizer.step()

                    else:
                        if epoch % 10 == 0:
                            if debug:
                                metrics_logger(preds, targets, lengths, logger, on_cpu=True)
                            else:
                                metrics_logger(preds, targets, lengths, wandb, on_cpu=False)

                # statistics
                running_loss += loss_value.item()

            epoch_loss = running_loss / len(dataloader)

            logger.info(f'{phase} Loss: {epoch_loss}')
            if phase == 'train':
                if not debug:
                    wandb.log({"Train loss": epoch_loss})
                print(epoch_loss)
            else:
                if not debug:
                    wandb.log({"Test loss": epoch_loss})

        if epoch % num_epoch_before_backup == 0 and model_backup_path:
            torch.save(model.state_dict(), model_backup_path)
            write_training_epoch(config, epoch, model_name, logger)

    return model


def write_training_epoch(config, epoch, model, logger):
    if model == 'simple':
        config_path_name = "PATH_TO_FINISHED_TRAINING_SIMPLEMODEL"
    elif model == 'simple-coordinates':
        config_path_name = "PATH_TO_FINISHED_TRAINING_SIMPLEMODEL_COORD"
    else:
        logger.error(f'Error: no such model {model}')
        return
    with open(config[config_path_name], 'w') as f:
        f.write(f"{epoch}")


def check_training_epoch(config, model, logger):
    if model == 'simple':
        config_path_name = "PATH_TO_FINISHED_TRAINING_SIMPLEMODEL"
    elif model == 'simple-coordinates':
        config_path_name = "PATH_TO_FINISHED_TRAINING_SIMPLEMODEL_COORD"
    else:
        logger.error(f'Error: no such model {model}')
        return
    if not os.path.isfile(config[config_path_name]):
        return 0
    with open(config[config_path_name], 'r') as f:
        epoch = int(f.read())
        return epoch


def try_load_unfinished_model(logger, config, model):
    if model == 'simple':
        config_path_name = "PATH_TO_SIMPLEMODEL_BACKUP"
    elif model == 'simple-coordinates':
        config_path_name = "PATH_TO_SIMPLEMODEL_COORD_BACKUP"
    else:
        logger.error(f'Error: no such model {model}')
        return
    try:
        state = torch.load(config[config_path_name])
        return state
    except:
        logger.exception(f'Error loading unfinished {model}')


def coordinates_metrics_logger(preds, targets, lengths, logger, on_cpu=False):
    metrics = coordinate_metrics(preds, targets, lengths, on_cpu)

    if on_cpu:
        logger.info(f"MAE batch: {metrics['mae']}")
        logger.info(f"Distance deviation between neighbours: {metrics['diff_neighbours_dist']}")
        logger.info(f"Angles deviation: {metrics['diff_angles']}")
    else:
        logger.log({"MAE batch": metrics['mae']})
        logger.log({"Distance deviation between neighbours": metrics['diff_neighbours_dist']})
        logger.log({"Angles deviation": metrics['diff_angles']})


def angles_metrics_logger(preds, targets, lengths, logger):
    threshold = 0.1
    mean_var_phi, accuracy_phi, mean_var_psi, accuracy_psi = angle_metrics(preds, targets,
                                                                           lengths,
                                                                           threshold=threshold)
    logger.log({"Phi MAE batch": mean_var_phi})
    logger.log({"Psi MAE batch": mean_var_psi})
    logger.log({f"Accuracy phi (threshold = {threshold}) batch": accuracy_phi})
    logger.log({f"Accuracy psi (threshold = {threshold}) batch": accuracy_psi})
    threshold = 0.5
    mean_var_phi, accuracy_phi, mean_var_psi, accuracy_psi = angle_metrics(preds, targets, lengths,
                                                                           threshold=threshold)
    logger.log({f"Accuracy phi (threshold = {threshold}) batch": accuracy_phi})
    logger.log({f"Accuracy psi (threshold = {threshold}) batch": accuracy_psi})


def try_load_model_backup(model, model_name, use_backup, logger, config):
    start_epoch = check_training_epoch(config, model_name, logger) if use_backup else 0
    logger.info(f'Starting training from epoch {start_epoch}')
    if start_epoch > 0:
        state = try_load_unfinished_model(logger, config, model_name) if use_backup else None
        if state:
            logger.info(f'Successfully loaded backup model')
            model.load_state_dict(state)
    return start_epoch, model


def initialize_wandb(model, config, n_layers, batch_size, model_name, hidden_dim):
    wandb.init(project=config["PROJECT_NAME"],
               name=f"{model_name} n_layers={n_layers} batch_size={batch_size} hidden_dim={hidden_dim} with_scheduler")
    wandb.watch(model)
