import os

import torch
import torch.nn.utils.rnn as rnn_utils
import wandb
from tqdm import tqdm
from metrics.metric import angle_metrics, coordinate_metrics


def train_model(train_dataloader, val_dataloader, model, loss, optimizer, num_epochs, logger, device, config,
                metrics_logger,
                model_backup_path=None, start_epoch=0, num_epoch_before_backup=100):
    for epoch in range(start_epoch, num_epochs):
        logger.info(f'Epoch {epoch}/{num_epochs - 1}')

        for phase in ['train', 'val']:
            if phase == 'train':
                dataloader = train_dataloader
                model.train()
            else:
                dataloader = val_dataloader
                model.eval()

            running_loss = 0.

            for inputs, targets, lengths in tqdm(dataloader):
                inputs = inputs.to(device)

                # repad target sequences
                targets = rnn_utils.pack_padded_sequence(targets, lengths, batch_first=True)
                targets, _ = rnn_utils.pad_packed_sequence(targets, batch_first=True)
                targets = targets.to(device)

                optimizer.zero_grad()

                # forward and backward
                with torch.set_grad_enabled(phase == 'train'):
                    preds, lengths, hiddens = model(inputs, lengths)
                    loss_value = loss(preds, targets)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss_value.backward()
                        optimizer.step()

                    else:
                        metrics_logger(wandb, preds, targets, lengths)

                # statistics
                running_loss += loss_value.item()

            epoch_loss = running_loss / len(dataloader)

            logger.info(f'{phase} Loss: {epoch_loss}')
            if phase == 'train':
                wandb.log({"Train loss": epoch_loss})
            else:
                wandb.log({"Test loss": epoch_loss})

        if epoch % num_epoch_before_backup == 0 and model_backup_path:
            torch.save(model.state_dict(), model_backup_path)
            write_training_epoch(config, epoch)

    return model


def write_training_epoch(config, epoch):
    with open(config["PATH_TO_FINISHED_TRAINING_SIMPLEMODEL"], 'w') as f:
        f.write(f"{epoch}")


def check_training_epoch(config):
    if not os.path.isfile(config["PATH_TO_FINISHED_TRAINING_SIMPLEMODEL"]):
        return 0
    with open(config["PATH_TO_FINISHED_TRAINING_SIMPLEMODEL"], 'r') as f:
        epoch = int(f.read())
        return epoch


def try_load_unfinished_model(logger, config):
    try:
        state = torch.load(config["PATH_TO_SIMPLEMODEL_BACKUP"])
        return state
    except:
        logger.exception('Error loading unfinished simplemodel')


def coordinates_metrics_logger(preds, targets, lengths, logger):
    metrics = coordinate_metrics(preds, targets, lengths)

    logger.log({"MAE batch": metrics['mae']})
    logger.log({"Distance deviation between ends": metrics['diff_ends_dist']})
    logger.log({"Distance deviation between neighbours": metrics['diff_neighbours_dist']})
    logger.log({"Percent distance deviation between ends": metrics['diff_ends_dist_p']})
    logger.log({"Percent distance deviation between neighbours": metrics['diff_neighbours_dist_p']})


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


def try_load_model_backup(model, use_backup, logger, config):
    start_epoch = check_training_epoch(config) if use_backup else 0
    logger.info(f'Starting training from epoch {start_epoch}')
    if start_epoch > 0:
        state = try_load_unfinished_model(logger, config) if use_backup else None
        if state:
            logger.info(f'Successfully loaded backup model')
            model.load_state_dict(state)
    return start_epoch, model


def initialize_wandb(model, config, n_layers, batch_size, model_name):
    wandb.init(project=config["PROJECT_NAME"],
               name=f"{model_name} n_layers={n_layers} batch_size={batch_size}")
    wandb.watch(model)
