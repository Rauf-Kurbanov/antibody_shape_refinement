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
                model_backup_path=None, start_epoch=0, num_epoch_before_backup=100):
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

                    loss_value = loss(preds, targets)

                    if phase == 'train':
                        # with autograd.detect_anomaly():

                        loss_value.backward()
                        optimizer.step()

                    else:
                        if epoch % 10 == 0:
                            pass
                            # metrics_logger(preds, targets, lengths, wandb)

                # statistics
                running_loss += loss_value.item()

            epoch_loss = running_loss / len(dataloader)

            logger.info(f'{phase} Loss: {epoch_loss}')
            if phase == 'train':
                # wandb.log({"Train loss": epoch_loss})
                print(epoch_loss)
            else:
                pass
                # wandb.log({"Test loss": epoch_loss})

        if epoch % num_epoch_before_backup == 0 and model_backup_path:
            torch.save(model.state_dict(), model_backup_path)
            write_training_epoch(config, epoch, model_name, logger)

    return model


def get_character_level_data(inputs, targets, hiddens, preds_seq, preds, idx):
    new_inputs = []
    new_targets = []
    new_h = []
    new_c = []
    new_preds = []
    new_preds_seq = []
    if preds_seq.shape[0] > 0:
        for input, target, h, c, pred_seq, pred in zip(inputs, targets, hiddens[0], hiddens[1], preds_seq, preds):

            if input.shape[0] > idx:
                assert target.shape[0] == input.shape[0]
                new_inputs.append(input[idx])
                new_targets.append(target[:idx + 1])
                new_h.append(h)
                new_c.append(c)
                new_preds_seq.append(pred_seq)
                new_preds.append(pred)
    else:
        for input, target, h, c, pred in zip(inputs, targets, hiddens[0], hiddens[1], preds):

            if input.shape[0] > idx:
                assert target.shape[0] == input.shape[0]
                new_inputs.append(input[idx])
                new_targets.append(target[:idx + 1])
                new_h.append(h)
                new_c.append(c)
                new_preds.append(pred)
    # new_preds = list(filter(lambda x: x.shape[0] > idx, inputs))

    # inputs = list(filter(lambda x: x.shape[0] > idx, inputs))
    # targets = list(filter(lambda x: x.shape[0] > idx, targets))
    # inputs_slice = torch.stack([item[idx] for item in inputs])
    # targets_slice = torch.stack([item[:idx + 1] for item in targets])
    new_inputs = torch.stack(new_inputs)
    new_targets = torch.stack(new_targets)
    new_h = torch.stack(new_h)
    new_c = torch.stack(new_c)
    new_preds_seq = torch.stack(new_preds_seq) if len(new_preds_seq) > 0 else preds_seq
    new_preds = torch.stack(new_preds)
    return new_inputs, new_targets, (new_h, new_c), new_preds_seq, new_preds


def unpad_data(inputs, targets, lengths):
    inputs_nopad = []
    targets_nopad = []
    for input, target, length in zip(torch.unbind(inputs), torch.unbind(targets), lengths):
        inputs_nopad.append(input[:length])
        targets_nopad.append(target[:length])
    return inputs_nopad, targets_nopad


def train_model_autoregressive(train_dataloader, val_dataloader, model, model_name, loss, optimizer, scheduler,
                               num_epochs, logger,
                               device,
                               config,
                               metrics_logger,
                               model_backup_path=None, start_epoch=0, num_epoch_before_backup=100):
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

                # targets = rnn_utils.pack_padded_sequence(targets, lengths, batch_first=True)
                # targets, _ = rnn_utils.pad_packed_sequence(targets, batch_first=True)
                targets = targets.to(device)
                inputs_nopad, targets_nopad = unpad_data(inputs, targets, lengths)
                max_length = max(lengths)

                optimizer.zero_grad()

                hiddens = model.init_hiddens(inputs.shape[0])
                preds = model.init_preds(inputs.shape[0])

                # как считать лосс тут? Накапливать последовательность и считать по ней
                with torch.set_grad_enabled(phase == 'train'):
                    loss_value_target = 0
                    preds_seq = torch.FloatTensor([])
                    for i in range(max_length):
                        input_char, target_char, hiddens, preds_seq, preds = get_character_level_data(inputs_nopad,
                                                                                               targets_nopad,
                                                                                               hiddens, preds_seq, preds, i)
                        model_input = torch.cat((input_char, preds), 1)
                        preds, hiddens = model(model_input, hiddens)
                        preds_seq = torch.cat((preds_seq, preds.unsqueeze(1)), 1)

                        loss_value = loss(preds_seq, target_char)
                        loss_value_target += loss_value.item()

                        if phase == 'train':
                            # with autograd.detect_anomaly():

                            loss_value.backward(retain_graph=True)
                            optimizer.step()

                        else:
                            if epoch % 10 == 0:
                                pass
                            # metrics_logger(preds, targets, lengths, wandb)

                    # statistics
                    running_loss += loss_value.item()

            epoch_loss = running_loss / len(dataloader)

            logger.info(f'{phase} Loss: {epoch_loss}')
            if phase == 'train':
                # wandb.log({"Train loss": epoch_loss})
                print(epoch_loss)
            else:
                pass
            # wandb.log({"Test loss": epoch_loss})

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


def coordinates_metrics_logger(preds, targets, lengths, logger):
    metrics = coordinate_metrics(preds, targets, lengths)

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
