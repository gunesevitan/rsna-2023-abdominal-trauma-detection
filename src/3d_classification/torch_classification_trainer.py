import sys
import argparse
import yaml
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.optim as optim

sys.path.append('..')
import settings
import metrics
import visualization
import torch_datasets
import torch_modules
import torch_utilities
import transforms


def train(
        training_loader, model,
        bowel_criterion, extravasation_criterion, kidney_criterion, liver_criterion, spleen_criterion,
        optimizer, device,
        scheduler=None, amp=False
):

    """
    Train given model on given data loader

    Parameters
    ----------
    training_loader: torch.utils.data.DataLoader
        Training set data loader

    model: torch.nn.Module
        Model to train

    bowel_criterion: torch.nn.Module
        Loss function for bowel labels

    extravasation_criterion: torch.nn.Module
        Loss function for extravasation labels

    kidney_criterion: torch.nn.Module
        Loss function for kidney labels

    liver_criterion: torch.nn.Module
        Loss function for liver labels

    spleen_criterion: torch.nn.Module
        Loss function for spleen labels

    optimizer: torch.optim.Optimizer
        Optimizer for updating model weights

    device: torch.device
        Location of the model and inputs

    scheduler: torch.optim.LRScheduler or None
        Learning rate scheduler

    amp: bool
        Whether to use auto mixed precision or not

    Returns
    -------
    training_loss: dict
        Dictionary of training losses after model is fully trained on training set data loader
    """

    model.train()
    progress_bar = tqdm(training_loader)

    running_loss_total = 0.0
    running_loss_bowel = 0.0
    running_loss_extravasation = 0.0
    running_loss_kidney = 0.0
    running_loss_liver = 0.0
    running_loss_spleen = 0.0

    if amp:
        grad_scaler = torch.cuda.amp.GradScaler()
    else:
        grad_scaler = None

    for step, (inputs, bowel_targets, extravasation_targets, kidney_targets, liver_targets, spleen_targets, any_targets) in enumerate(progress_bar):

        inputs = inputs.to(device)
        bowel_targets = bowel_targets.to(device)
        extravasation_targets = extravasation_targets.to(device)
        kidney_targets = kidney_targets.to(device)
        liver_targets = liver_targets.to(device)
        spleen_targets = spleen_targets.to(device)

        bowel_weights = torch.where(bowel_targets == 1, 2, 1).view(-1, 1)
        extravasation_weights = torch.where(extravasation_targets == 1, 6, 1).view(-1, 1)
        kidney_weights = torch.ones_like(kidney_targets)
        liver_weights = torch.ones_like(liver_targets)
        spleen_weights = torch.ones_like(spleen_targets)
        for label, weight in zip([2, 1], [4, 2]):
            kidney_weights[kidney_targets == label] = weight
            liver_weights[liver_targets == label] = weight
            spleen_weights[spleen_targets == label] = weight

        optimizer.zero_grad()

        if amp:
            with torch.cuda.amp.autocast():
                bowel_outputs, extravasation_outputs, kidney_outputs, liver_outputs, spleen_outputs = model(inputs.half())
        else:
            bowel_outputs, extravasation_outputs, kidney_outputs, liver_outputs, spleen_outputs = model(inputs)

        bowel_loss = bowel_criterion(bowel_outputs, bowel_targets.view(-1, 1), bowel_weights)
        extravasation_loss = extravasation_criterion(extravasation_outputs, extravasation_targets.view(-1, 1), extravasation_weights)
        kidney_loss = kidney_criterion(kidney_outputs, kidney_targets, kidney_weights)
        liver_loss = liver_criterion(liver_outputs, liver_targets, liver_weights)
        spleen_loss = spleen_criterion(spleen_outputs, spleen_targets, spleen_weights)
        loss_total = bowel_loss + extravasation_loss + kidney_loss + liver_loss + spleen_loss

        if amp:
            grad_scaler.scale(loss_total).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss_total.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss_total += loss_total.detach().item() * len(inputs)
        running_loss_bowel += bowel_loss.item() * len(inputs)
        running_loss_extravasation += extravasation_loss.item() * len(inputs)
        running_loss_kidney += kidney_loss.item() * len(inputs)
        running_loss_liver += liver_loss.item() * len(inputs)
        running_loss_spleen += spleen_loss.item() * len(inputs)
        lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
        progress_bar.set_description(f'lr: {lr:.8f} - training loss_total: {running_loss_total / len(training_loader.sampler):.4f} bowel_loss: {running_loss_bowel / len(training_loader.sampler):.4f} extravasation_loss: {running_loss_extravasation / len(training_loader.sampler):.4f} kidney_loss: {running_loss_kidney / len(training_loader.sampler):.4f} liver_loss: {running_loss_liver / len(training_loader.sampler):.4f} spleen_loss: {running_loss_spleen / len(training_loader.sampler):.4f}')

    training_loss_total = running_loss_total / len(training_loader.sampler)
    training_loss_bowel = running_loss_bowel / len(training_loader.sampler)
    training_loss_extravasation = running_loss_extravasation / len(training_loader.sampler)
    training_loss_kidney = running_loss_kidney / len(training_loader.sampler)
    training_loss_liver = running_loss_liver / len(training_loader.sampler)
    training_loss_spleen = running_loss_spleen / len(training_loader.sampler)

    training_loss = {
        'loss_total': training_loss_total,
        'loss_bowel': training_loss_bowel,
        'loss_extravasation': training_loss_extravasation,
        'loss_kidney': training_loss_kidney,
        'loss_liver': training_loss_liver,
        'loss_spleen': training_loss_spleen
    }

    return training_loss


def validate(
        validation_loader, model,
        bowel_criterion, extravasation_criterion, kidney_criterion, liver_criterion, spleen_criterion,
        device, amp=False
):

    """
    Validate given model on given data loader

    Parameters
    ----------
    validation_loader: torch.utils.data.DataLoader
        Validation set data loader

    model: torch.nn.Module
        Model to validate

    bowel_criterion: torch.nn.Module
        Loss function for bowel labels

    extravasation_criterion: torch.nn.Module
        Loss function for extravasation labels

    kidney_criterion: torch.nn.Module
        Loss function for kidney labels

    liver_criterion: torch.nn.Module
        Loss function for liver labels

    spleen_criterion: torch.nn.Module
        Loss function for spleen labels

    device: torch.device
        Location of the model and inputs

    amp: bool
        Whether to use auto mixed precision or not

    Returns
    -------
    validation_loss: dict
        Dictionary of validation losses after model is fully validated on validation set data loader
    """

    model.eval()
    progress_bar = tqdm(validation_loader)

    running_loss_total = 0.0
    running_loss_bowel = 0.0
    running_loss_extravasation = 0.0
    running_loss_kidney = 0.0
    running_loss_liver = 0.0
    running_loss_spleen = 0.0

    for step, (inputs, bowel_targets, extravasation_targets, kidney_targets, liver_targets, spleen_targets, any_targets) in enumerate(progress_bar):

        inputs = inputs.to(device)
        bowel_targets = bowel_targets.to(device)
        extravasation_targets = extravasation_targets.to(device)
        kidney_targets = kidney_targets.to(device)
        liver_targets = liver_targets.to(device)
        spleen_targets = spleen_targets.to(device)

        bowel_weights = torch.where(bowel_targets == 1, 2, 1).view(-1, 1)
        extravasation_weights = torch.where(extravasation_targets == 1, 6, 1).view(-1, 1)
        kidney_weights = torch.ones_like(kidney_targets)
        liver_weights = torch.ones_like(liver_targets)
        spleen_weights = torch.ones_like(spleen_targets)
        for label, weight in zip([2, 1], [4, 2]):
            kidney_weights[kidney_targets == label] = weight
            liver_weights[liver_targets == label] = weight
            spleen_weights[spleen_targets == label] = weight

        with torch.no_grad():
            if amp:
                with torch.cuda.amp.autocast():
                    bowel_outputs, extravasation_outputs, kidney_outputs, liver_outputs, spleen_outputs = model(inputs.half())
            else:
                bowel_outputs, extravasation_outputs, kidney_outputs, liver_outputs, spleen_outputs = model(inputs)

        bowel_loss = bowel_criterion(bowel_outputs, bowel_targets.view(-1, 1), bowel_weights)
        extravasation_loss = extravasation_criterion(extravasation_outputs, extravasation_targets.view(-1, 1), extravasation_weights)
        kidney_loss = kidney_criterion(kidney_outputs, kidney_targets, kidney_weights)
        liver_loss = liver_criterion(liver_outputs, liver_targets, liver_weights)
        spleen_loss = spleen_criterion(spleen_outputs, spleen_targets, spleen_weights)
        loss_total = bowel_loss + extravasation_loss + kidney_loss + liver_loss + spleen_loss

        running_loss_total += loss_total.detach().item() * len(inputs)
        running_loss_bowel += bowel_loss.item() * len(inputs)
        running_loss_extravasation += extravasation_loss.item() * len(inputs)
        running_loss_kidney += kidney_loss.item() * len(inputs)
        running_loss_liver += liver_loss.item() * len(inputs)
        running_loss_spleen += spleen_loss.item() * len(inputs)

        progress_bar.set_description(f'validation loss_total: {running_loss_total / len(validation_loader.sampler):.4f} bowel_loss: {running_loss_bowel / len(validation_loader.sampler):.4f} extravasation_loss: {running_loss_extravasation / len(validation_loader.sampler):.4f} kidney_loss: {running_loss_kidney / len(validation_loader.sampler):.4f} liver_loss: {running_loss_liver / len(validation_loader.sampler):.4f} spleen_loss: {running_loss_spleen / len(validation_loader.sampler):.4f}')

    validation_loss_total = running_loss_total / len(validation_loader.sampler)
    validation_loss_bowel = running_loss_bowel / len(validation_loader.sampler)
    validation_loss_extravasation = running_loss_extravasation / len(validation_loader.sampler)
    validation_loss_kidney = running_loss_kidney / len(validation_loader.sampler)
    validation_loss_liver = running_loss_liver / len(validation_loader.sampler)
    validation_loss_spleen = running_loss_spleen / len(validation_loader.sampler)

    validation_loss = {
        'loss_total': validation_loss_total,
        'loss_bowel': validation_loss_bowel,
        'loss_extravasation': validation_loss_extravasation,
        'loss_kidney': validation_loss_kidney,
        'loss_liver': validation_loss_liver,
        'loss_spleen': validation_loss_spleen
    }

    return validation_loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model_directory', type=str)
    parser.add_argument('mode', type=str)
    args = parser.parse_args()

    config = yaml.load(open(settings.MODELS / args.model_directory / 'config.yaml', 'r'), Loader=yaml.FullLoader)
    settings.logger.info(f'Running {config["persistence"]["model_directory"]} model in {args.mode} mode')

    # Create directory for models and visualizations
    model_root_directory = Path(settings.MODELS / args.model_directory)
    model_root_directory.mkdir(parents=True, exist_ok=True)

    dataset = config['dataset']['dataset_name']
    df = pd.read_parquet(settings.DATA / 'datasets' / dataset / 'metadata.parquet')
    df_train = pd.read_csv(settings.DATA / 'rsna-2023-abdominal-trauma-detection' / 'train.csv')
    df_folds = pd.read_csv(settings.DATA / 'folds.csv')
    df = df.merge(df_train, on='patient_id', how='left')
    df = df.merge(df_folds, on='patient_id', how='left')
    del df_train, df_folds

    # Convert one-hot encoded target columns to a single multi-class target columns
    for multiclass_target in ['kidney', 'liver', 'spleen']:
        df[multiclass_target] = 0
        for label, column in enumerate([f'{multiclass_target}_healthy', f'{multiclass_target}_low', f'{multiclass_target}_high']):
            df.loc[df[column] == 1, multiclass_target] = label

    if args.mode == 'training':

        dataset_transforms = transforms.get_classification_transforms(**config['transforms'])
        training_metadata = defaultdict(dict)

        for fold in config['training']['folds']:

            training_idx, validation_idx = df.loc[df[fold] != 1].index, df.loc[df[fold] == 1].index
            # Validate on training set if validation is set is not specified
            if len(validation_idx) == 0:
                validation_idx = training_idx

            # Create training and validation inputs and targets
            training_volume_paths, training_targets = torch_datasets.prepare_classification_data(df=df.loc[training_idx])
            validation_volume_paths, validation_targets = torch_datasets.prepare_classification_data(df=df.loc[validation_idx])

            settings.logger.info(
                f'''
                Fold: {fold}
                Training: {len(training_volume_paths)} ({len(training_volume_paths) // config["training"]["training_batch_size"] + 1} steps)
                Validation {len(validation_volume_paths)} ({len(validation_volume_paths) // config["training"]["test_batch_size"] + 1} steps)
                '''
            )

            # Create training and validation datasets and dataloaders
            training_dataset = torch_datasets.ClassificationDataset(
                volume_paths=training_volume_paths,
                **training_targets,
                transforms=dataset_transforms['training'],
            )
            training_loader = DataLoader(
                training_dataset,
                batch_size=config['training']['training_batch_size'],
                sampler=RandomSampler(training_dataset, replacement=False),
                pin_memory=False,
                drop_last=False,
                num_workers=config['training']['num_workers']
            )
            validation_dataset = torch_datasets.ClassificationDataset(
                volume_paths=validation_volume_paths,
                **validation_targets,
                transforms=dataset_transforms['inference'],
            )
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=config['training']['test_batch_size'],
                sampler=SequentialSampler(validation_dataset),
                pin_memory=False,
                drop_last=False,
                num_workers=config['training']['num_workers']
            )

            # Set model, device and seed for reproducible results
            torch_utilities.set_seed(config['training']['random_state'], deterministic_cudnn=config['training']['deterministic_cudnn'])
            device = torch.device(config['training']['device'])

            bowel_criterion = getattr(torch_modules, config['training']['bowel_loss_function'])(**config['training']['bowel_loss_function_args'])
            extravasation_criterion = getattr(torch_modules, config['training']['extravasation_loss_function'])(**config['training']['extravasation_loss_function_args'])
            kidney_criterion = getattr(torch_modules, config['training']['kidney_loss_function'])(**config['training']['kidney_loss_function_args'])
            liver_criterion = getattr(torch_modules, config['training']['liver_loss_function'])(**config['training']['liver_loss_function_args'])
            spleen_criterion = getattr(torch_modules, config['training']['spleen_loss_function'])(**config['training']['spleen_loss_function_args'])

            model = getattr(torch_modules, config['model']['model_class'])(**config['model']['model_args'])
            if config['model']['model_checkpoint_path'] is not None:
                model.load_state_dict(torch.load(config['model']['model_checkpoint_path']), strict=False)
            model.to(device)

            # Set optimizer, learning rate scheduler and stochastic weight averaging
            optimizer = getattr(torch.optim, config['training']['optimizer'])(model.parameters(), **config['training']['optimizer_args'])
            scheduler = getattr(optim.lr_scheduler, config['training']['lr_scheduler'])(optimizer, **config['training']['lr_scheduler_args'])
            amp = config['training']['amp']

            best_epoch = 1
            early_stopping = False
            training_history = {
                'training_loss': [],
                'validation_loss': []
            }

            for epoch in range(1, config['training']['epochs'] + 1):

                if early_stopping:
                    break

                training_loss = train(
                    training_loader=training_loader,
                    model=model,
                    bowel_criterion=bowel_criterion,
                    extravasation_criterion=extravasation_criterion,
                    kidney_criterion=kidney_criterion,
                    liver_criterion=liver_criterion,
                    spleen_criterion=spleen_criterion,
                    optimizer=optimizer,
                    device=device,
                    scheduler=scheduler,
                    amp=amp
                )

                validation_loss = validate(
                    validation_loader=validation_loader,
                    model=model,
                    bowel_criterion=bowel_criterion,
                    extravasation_criterion=extravasation_criterion,
                    kidney_criterion=kidney_criterion,
                    liver_criterion=liver_criterion,
                    spleen_criterion=spleen_criterion,
                    device=device,
                    amp=amp
                )

                settings.logger.info(
                    f'''
                    Epoch {epoch}
                    Training Loss: {json.dumps(training_loss, indent=2)}
                    Validation Loss: {json.dumps(validation_loss, indent=2)}
                    '''
                )

                if epoch in config['persistence']['save_epoch_model']:
                    # Save model if current epoch is specified to be saved
                    model_name = f'model_{fold}_epoch{epoch}.pt'
                    torch.save(model.state_dict(), model_root_directory / model_name)
                    settings.logger.info(f'Saved {model_name} to {model_root_directory}')

                best_validation_loss = np.min(training_history['validation_loss']) if len(training_history['validation_loss']) > 0 else np.inf
                last_validation_loss = validation_loss['loss_total']
                if last_validation_loss < best_validation_loss:
                    # Save model if validation loss improves
                    model_name = f'model_{fold}_best.pt'
                    torch.save(model.state_dict(), model_root_directory / model_name)
                    settings.logger.info(f'Saved {model_name} to {model_root_directory} (validation loss decreased from {best_validation_loss:.6f} to {last_validation_loss:.6f})\n')

                training_history['training_loss'].append(training_loss['loss_total'])
                training_history['validation_loss'].append(validation_loss['loss_total'])

                best_epoch = np.argmin(training_history['validation_loss'])
                if config['training']['early_stopping_patience'] > 0:
                    # Trigger early stopping if early stopping patience is greater than 0
                    if len(training_history['validation_loss']) - best_epoch > config['training']['early_stopping_patience']:
                        settings.logger.info(
                            f'''
                            Early Stopping (validation loss didn\'t improve for {config['training']["early_stopping_patience"]} epochs)
                            Best Epoch ({best_epoch + 1}) Validation Loss: {training_history["validation_loss"][best_epoch]:.4f}
                            '''
                        )
                        early_stopping = True

            training_metadata[fold] = {
                'best_epoch': int(best_epoch),
                'training_loss': float(training_history['training_loss'][best_epoch]),
                'validation_loss': float(training_history['validation_loss'][best_epoch]),
                'training_history': training_history
            }

            visualization.visualize_learning_curve(
                training_losses=training_metadata[fold]['training_history']['training_loss'],
                validation_losses=training_metadata[fold]['training_history']['validation_loss'],
                best_epoch=training_metadata[fold]['best_epoch'],
                path=model_root_directory / f'learning_curve_{fold}.png'
            )

            with open(model_root_directory / 'training_metadata.json', mode='w') as f:
                json.dump(training_metadata, f, indent=2, ensure_ascii=False)

    elif args.mode == 'test':

        dataset_transforms = transforms.get_classification_transforms(**config['transforms'])
        test_metadata = defaultdict(dict)

        df_predictions = []
        df_scores = []

        for fold, model_file_name in zip(config['test']['folds'], config['test']['model_file_names']):

            # Create validation inputs and targets
            validation_idx = df.loc[df[fold] == 1].index
            validation_volume_paths, _ = torch_datasets.prepare_classification_data(df=df.loc[validation_idx])

            settings.logger.info(
                f'''
                Fold: {fold} ({model_file_name})
                Validation {len(validation_volume_paths)} ({len(validation_volume_paths) // config["training"]["test_batch_size"] + 1} steps)
                '''
            )

            # Create validation datasets and dataloaders
            validation_dataset = torch_datasets.ClassificationDataset(
                volume_paths=validation_volume_paths,
                transforms=dataset_transforms['inference'],
            )
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=config['training']['test_batch_size'],
                sampler=SequentialSampler(validation_dataset),
                pin_memory=False,
                drop_last=False,
                num_workers=config['training']['num_workers']
            )

            # Set model, device and seed for reproducible results
            torch_utilities.set_seed(config['training']['random_state'], deterministic_cudnn=config['training']['deterministic_cudnn'])
            device = torch.device(config['training']['device'])
            amp = config['training']['amp']

            model = getattr(torch_modules, config['model']['model_class'])(**config['model']['model_args'])
            model.load_state_dict(torch.load(model_root_directory / model_file_name))
            model.to(device)
            model.eval()

            bowel_predictions = []
            extravasation_predictions = []
            kidney_predictions = []
            liver_predictions = []
            spleen_predictions = []

            for inputs in tqdm(validation_loader):

                inputs = inputs.to(device)

                with torch.no_grad():
                    bowel_outputs, extravasation_outputs, kidney_outputs, liver_outputs, spleen_outputs = model(inputs)

                bowel_outputs = bowel_outputs.cpu()
                extravasation_outputs = extravasation_outputs.cpu()
                kidney_outputs = kidney_outputs.cpu()
                liver_outputs = liver_outputs.cpu()
                spleen_outputs = spleen_outputs.cpu()

                if config['test']['tta']:

                    inputs = inputs.to('cpu')
                    tta_flip_dimensions = config['test']['tta_flip_dimensions']

                    tta_bowel_outputs = []
                    tta_extravasation_outputs = []
                    tta_kidney_outputs = []
                    tta_liver_outputs = []
                    tta_spleen_outputs = []

                    for dimensions in tta_flip_dimensions:

                        augmented_inputs = torch.flip(inputs, dims=dimensions).to(device)

                        with torch.no_grad():
                            augmented_bowel_outputs, augmented_extravasation_outputs, augmented_kidney_outputs, augmented_liver_outputs, augmented_spleen_outputs = model(augmented_inputs)

                        tta_bowel_outputs.append(augmented_bowel_outputs.cpu())
                        tta_extravasation_outputs.append(augmented_extravasation_outputs.cpu())
                        tta_kidney_outputs.append(augmented_kidney_outputs.cpu())
                        tta_liver_outputs.append(augmented_liver_outputs.cpu())
                        tta_spleen_outputs.append(augmented_spleen_outputs.cpu())

                    bowel_outputs = torch.stack(([bowel_outputs] + tta_bowel_outputs), dim=-1)
                    extravasation_outputs = torch.stack(([extravasation_outputs] + tta_extravasation_outputs), dim=-1)
                    kidney_outputs = torch.stack(([kidney_outputs] + tta_kidney_outputs), dim=-1)
                    liver_outputs = torch.stack(([liver_outputs] + tta_liver_outputs), dim=-1)
                    spleen_outputs = torch.stack(([spleen_outputs] + tta_spleen_outputs), dim=-1)

                    bowel_outputs = torch.mean(bowel_outputs, dim=-1)
                    extravasation_outputs = torch.mean(extravasation_outputs, dim=-1)
                    kidney_outputs = torch.mean(kidney_outputs, dim=-1)
                    liver_outputs = torch.mean(liver_outputs, dim=-1)
                    spleen_outputs = torch.mean(spleen_outputs, dim=-1)

                bowel_predictions += [bowel_outputs]
                extravasation_predictions += [extravasation_outputs]
                kidney_predictions += [kidney_outputs]
                liver_predictions += [liver_outputs]
                spleen_predictions += [spleen_outputs]

            bowel_predictions = torch.sigmoid(torch.cat(bowel_predictions, dim=0)).numpy()
            extravasation_predictions = torch.sigmoid(torch.cat(extravasation_predictions, dim=0)).numpy()
            kidney_predictions = torch.softmax(torch.cat(kidney_predictions, dim=0), dim=-1).numpy()
            liver_predictions = torch.softmax(torch.cat(liver_predictions, dim=0), dim=-1).numpy()
            spleen_predictions = torch.softmax(torch.cat(spleen_predictions, dim=0), dim=-1).numpy()

            df_fold_predictions = pd.DataFrame(np.hstack([
                validation_volume_paths.reshape(-1, 1),
                bowel_predictions,
                extravasation_predictions,
                kidney_predictions,
                liver_predictions,
                spleen_predictions
            ]), columns=[
                'volume_path', 'bowel_injury_prediction', 'extravasation_injury_prediction',
                'kidney_healthy_prediction', 'kidney_low_prediction', 'kidney_high_prediction',
                'liver_healthy_prediction', 'liver_low_prediction', 'liver_high_prediction',
                'spleen_healthy_prediction', 'spleen_low_prediction', 'spleen_high_prediction',
            ])
            df_fold_predictions['patient_id'] = df_fold_predictions['volume_path'].apply(lambda x: x.split('/')[-1].split('.')[0].split('_')[0]).astype(int)
            df_fold_predictions['scan_id'] = df_fold_predictions['volume_path'].apply(lambda x: x.split('/')[-1].split('.')[0].split('_')[-1]).astype(int)
            df_fold_predictions['bowel_healthy_prediction'] = 1 - df_fold_predictions['bowel_injury_prediction']
            df_fold_predictions['extravasation_healthy_prediction'] = 1 - df_fold_predictions['extravasation_injury_prediction']
            df_fold_predictions['any_injury_prediction'] = (1 - df_fold_predictions[[
                'bowel_healthy_prediction', 'extravasation_healthy_prediction',
                'kidney_healthy_prediction', 'liver_healthy_prediction', 'spleen_healthy_prediction'
            ]]).max(axis=1)
            df_fold_predictions = df_fold_predictions.merge(
                df.groupby(['patient_id', 'scan_id']).first().reset_index().loc[:, [
                        'patient_id', 'scan_id',
                        'bowel_injury', 'extravasation_injury', 'any_injury',
                        'kidney_healthy', 'kidney_low', 'kidney_high', 'kidney',
                        'liver_healthy', 'liver_low', 'liver_high', 'liver',
                        'spleen_healthy', 'spleen_low', 'spleen_high', 'spleen'
                   ]
                ],
                on=['patient_id', 'scan_id'], how='left'
            )

            fold_scores = {}
            df_fold_predictions = metrics.create_sample_weights(df=df_fold_predictions)

            binary_target_columns = ['bowel_injury', 'extravasation_injury', 'any_injury']
            for column in binary_target_columns:
                target_scores = metrics.binary_classification_scores(
                    y_true=df_fold_predictions[column],
                    y_pred=df_fold_predictions[f'{column}_prediction'],
                    sample_weights=df_fold_predictions[f'{column}_weight']
                )
                fold_scores[column] = target_scores
                settings.logger.info(f'Fold: {fold} {column} Scores: {json.dumps(target_scores)}')

            multiclass_target_column_groups = [
                ['kidney_healthy', 'kidney_low', 'kidney_high'],
                ['liver_healthy', 'liver_low', 'liver_high'],
                ['spleen_healthy', 'spleen_low', 'spleen_high']
            ]
            for multi_class_target_column, column_group in zip(['kidney', 'liver', 'spleen'], multiclass_target_column_groups):
                target_scores = metrics.multiclass_classification_scores(
                    y_true=df_fold_predictions[multi_class_target_column],
                    y_pred=df_fold_predictions[[f'{column}_prediction' for column in column_group]],
                    sample_weights=df_fold_predictions[f'{multi_class_target_column}_weight']
                )
                fold_scores[multi_class_target_column] = target_scores
                settings.logger.info(f'Fold: {fold} {multi_class_target_column} Scores: {json.dumps(target_scores)}')

            df_fold_scores = pd.DataFrame(fold_scores).T.reset_index().rename(columns={'index': 'target'})
            df_fold_scores['fold'] = fold

            df_predictions.append(df_fold_predictions)
            df_scores.append(df_fold_scores)

        df_predictions = pd.concat(df_predictions, axis=0, ignore_index=True).reset_index(drop=True)
        oof_scores = {}

        for column in binary_target_columns:
            target_scores = metrics.binary_classification_scores(
                y_true=df_predictions[column],
                y_pred=df_predictions[f'{column}_prediction'],
                sample_weights=df_predictions[f'{column}_weight']
            )
            oof_scores[column] = target_scores
            settings.logger.info(f'OOF {column} Scores: {json.dumps(target_scores)}')

        for multi_class_target_column, column_group in zip(['kidney', 'liver', 'spleen'], multiclass_target_column_groups):
            target_scores = metrics.multiclass_classification_scores(
                y_true=df_predictions[multi_class_target_column],
                y_pred=df_predictions[[f'{column}_prediction' for column in column_group]],
                sample_weights=df_predictions[f'{multi_class_target_column}_weight']
            )
            oof_scores[multi_class_target_column] = target_scores
            settings.logger.info(f'OOF {multi_class_target_column} Scores: {json.dumps(target_scores)}')

        df_oof_scores = pd.DataFrame(oof_scores).T.reset_index().rename(columns={'index': 'target'})
        df_oof_scores['fold'] = 'oof'

        df_scores.append(df_oof_scores)
        df_scores = pd.concat(df_scores, axis=0, ignore_index=True).reset_index(drop=True)

        for column in ['bowel_injury', 'extravasation_injury', 'kidney', 'liver', 'spleen', 'any_injury']:
            visualization.visualize_scores(
                df_scores=df_scores.loc[(df_scores['fold'] != 'oof') & (df_scores['target'] == column)].drop(columns=['fold', 'target']).reset_index(drop=True),
                title=f'{column} Fold Scores',
                path=model_root_directory / f'test_{column}_fold_scores.png'
            )

        visualization.visualize_scores(
            df_scores=df_scores.loc[df_scores['fold'] != 'oof'].drop(columns=['target']).groupby('fold').mean().reset_index(drop=True),
            title=f'Average Fold Scores',
            path=model_root_directory / f'test_average_fold_scores.png'
        )

        for column in [
            'bowel_injury', 'extravasation_injury', 'any_injury',
            'kidney_healthy', 'kidney_low', 'kidney_high',
            'liver_healthy', 'liver_low', 'liver_high',
            'spleen_healthy', 'spleen_low', 'spleen_high'
        ]:
            visualization.visualize_predictions(
                y_true=df_predictions[column],
                y_pred=df_predictions[f'{column}_prediction'],
                title=f'{column} Histogram',
                path=model_root_directory / f'test_{column}_predictions_histogram.png'
            )

        df_oof_scores = df_scores.loc[df_scores['fold'] == 'oof'].drop(columns=['fold'])
        df_oof_scores.to_csv(model_root_directory / 'oof_scores.csv', index=False)

        df_oof_score_aggregations = df_oof_scores.iloc[:, 1:].agg(['mean', 'std', 'min', 'max']).reset_index().rename(columns={'index': 'aggregation'})
        df_oof_score_aggregations.to_csv(model_root_directory / 'oof_score_aggregations.csv', index=False)

        df_predictions[[
            'patient_id', 'scan_id',
            'bowel_injury', 'bowel_injury_prediction', 'bowel_injury_weight',
            'extravasation_injury', 'extravasation_injury_prediction', 'extravasation_injury_weight',
            'kidney_healthy', 'kidney_low', 'kidney_high', 'kidney', 'kidney_weight',
            'kidney_healthy_prediction', 'kidney_low_prediction', 'kidney_high_prediction',
            'liver_healthy', 'liver_low', 'liver_high', 'liver', 'liver_weight',
            'liver_healthy_prediction', 'liver_low_prediction', 'liver_high_prediction',
            'spleen_healthy', 'spleen_low', 'spleen_high', 'spleen', 'spleen_weight',
            'spleen_healthy_prediction', 'spleen_low_prediction', 'spleen_high_prediction',
            'any_injury', 'any_injury_prediction', 'any_injury_weight'
        ]].to_csv(model_root_directory / 'oof_predictions.csv', index=False)
