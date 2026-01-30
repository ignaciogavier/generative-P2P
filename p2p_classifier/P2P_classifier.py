import os
import glob
import argparse
from tqdm import tqdm
from collections import Counter

import wandb
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, average_precision_score, roc_auc_score, confusion_matrix, classification_report, accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, cohen_kappa_score
from sklearn.metrics._ranking import _binary_clf_curve
from sklearn.model_selection import LeaveOneGroupOut
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
torch.backends.cudnn.enabled = False

from utils_architectures import *
from utils_data import *
import biobank_model_setup as bbms


def run_epoch(model, dataloader, criterion, optimizer=None, split='train', threshold=0.5):
    is_train = split == 'train'
    is_val = split == 'val'
    is_test = split == 'test'
    
    if is_train: model.train()
    else: model.eval()
    
    running_loss = 0.0
    y_true = []
    y_pred = []
    
    with torch.set_grad_enabled(is_train):
        for batch in tqdm(dataloader, total=len(dataloader)):
            inputs, labels = batch['data'].to(device), batch['label'].to(device)
            
            weights = WEIGHTS[0] * (1 - labels) + WEIGHTS[1] * labels
            criterion.weight = weights
            
            if is_train: optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            if is_train: loss.backward()
            if is_train: optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            y_true += labels.detach().cpu().numpy().flatten().tolist()
            y_pred += outputs.detach().cpu().numpy().flatten().tolist()
            
        
        ratio = sum(y_true) / (len(y_true) - sum(y_true))
        fps, tps, thrs = _binary_clf_curve(y_true, y_pred, pos_label=1)
        fps, tps, thrs = fps[::-1], tps[::-1], thrs[::-1]
        
        thrs = np.hstack([thrs, 1])
        precs = np.hstack([np.divide(tps, tps + fps, where=(tps + fps != 0)), 1])
        prec_unbs = np.hstack([np.divide(tps, tps + ratio * fps, where=(tps + ratio * fps != 0)), 1])
        recs = np.hstack([tps / tps[0], 0])
        f1s = 2 * precs * recs / (precs + recs + 1e-5)
        f1_unbs = 2 * prec_unbs * recs / (prec_unbs + recs + 1e-5)
        
        argmaxf1 = np.nanargmax(f1s)
        argmaxf1_unb = np.nanargmax(f1_unbs)
        bestf1, bestth = f1s[argmaxf1], thrs[argmaxf1]
        bestf1_unb, bestth_unb = f1_unbs[argmaxf1_unb], thrs[argmaxf1_unb]
        
        thr_i = np.searchsorted(thrs, threshold)
        prec, prec_unb, rec = precs[thr_i], prec_unbs[thr_i], recs[thr_i]
        f1, f1_unb = f1s[thr_i], f1_unbs[thr_i]
        
        avprec = -np.sum(np.diff(recs) * np.array(precs)[:-1])
        avprec_unb = -np.sum(np.diff(recs) * np.array(prec_unbs)[:-1])
        
        auroc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) == 2 else 0
        
        df = wandb.Table(dataframe=pd.DataFrame(np.stack([thrs, precs, prec_unbs, recs, f1s, f1_unbs], 1),
                                                columns=['thrs', 'precs', 'prec_unbs', 'recs', 'f1s', 'f1_unbs']))
    
    return {'loss': running_loss / len(dataloader.dataset), 'prec':prec, 'prec_unb':prec_unb, 'rec':rec,
            'f1':f1, 'f1_unb':f1_unb, 'auprc':avprec, 'auprc_unb':avprec_unb, 'auroc':auroc,
            'bestf1':bestf1, 'bestth':bestth, 'bestf1_unb':bestf1_unb, 'bestth_unb':bestth_unb, 'df':df}


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    tags = ['19subjects', 'synthmocap', 'synth_allP2P', 'singleP2P', 'auto_man_labels'] # 'lenMoCap_thr', 'no_nans_mocap', 'smooth2', 'sweep_real_size', 'sweep_synth_size', 'weighted_loss', 'f1loss'
    tags += ['resnet' + str(sweep_configuration['parameters']['resnet_type']['value'])]
    # if sweep_configuration['parameters']['stroke_size']['value'] > 0: tags += ['real_stroke']
    if sweep_configuration['parameters']['validate_thr']['value'] == 'bestth': tags += ['valF1']
    if sweep_configuration['parameters']['validate_thr']['value'] == 'bestth_unb': tags += ['valF1unb']
    run = wandb.init(config=sweep_configuration, group='train-real_val_test_real-LOSO_fold-resnet18', tags=tags)
    
    # Define hyperparameters and other settings
    batch_size = wandb.config.batch_size
    num_epochs = wandb.config.epochs
    learning_rate = wandb.config.lr
    seed = wandb.config.random_seed
    dropout_prob = wandb.config.dropout_prob
    synth_size = wandb.config.synth_size
    # real_size = wandb.config.real_size
    # stroke_size = wandb.config.stroke_size
    synth_splits = wandb.config.synth_splits
    real_splits = wandb.config.real_splits
    stroke_splits = wandb.config.stroke_splits
    win_th = wandb.config.windows_thresholds
    per_subj = bool(wandb.config.per_subj)
    with_aug = bool(wandb.config.with_aug)
    with_gyro = bool(wandb.config.with_gyro)
    with_acc = bool(wandb.config.with_acc)
    validate_thr = wandb.config.validate_thr
    compressed = wandb.config.compressed
    resnet_type = wandb.config.resnet_type
    pretrained = wandb.config.pretrained
    posttrained = wandb.config.posttrained
    pretrained_model = wandb.config.pretrained_model
    pretrained_biobank = wandb.config.pretrained_biobank
    finetune_val = wandb.config.finetune_val
    save_model = bool(wandb.config.save_model)
    
    dataset = {}
    
    # Create real dataset if necessary
    if sum(real_splits) > 0:
        print(f'Creating real dataset: {real_splits}')
        rng_real = np.random.default_rng(seed)
        perm = rng_real.permutation([1,2,3,4,5,6,8,10,11,12,13,14,15,16,17,18,19,20]) # removing 7
        splits = [np.split(perm, [real_splits[0], 18-sum(real_splits[1:]),
                                  18-real_splits[2]])[s] for s in [0,2,3]]
        print(f'Subject splits (healthy): {splits}')
        
        dataset['real'] = {}
        for name, split in zip(['tr','va','te'], splits):
            file_names = sum([
                sorted(
                    glob.glob(f'{root_dir}/data_real_th{win_th:0.1f}_singleP2P_man/S{s}/*_*_*.npy'),
                    key=lambda x: int(os.path.basename(x).split('_')[1])
                ) for s in split], [])
            dataset['real'][name] = CustomDataset(file_names, is_synth=False, augment=with_aug,
                                                  with_gyro=with_gyro, with_acc=with_acc)
    
    # Create stroke dataset if necessary
    if sum(stroke_splits) > 0:
        print(f'Creating stroke dataset: {stroke_splits}')
        rng_stroke = np.random.default_rng(seed)
        perm = rng_stroke.permutation([1,2,3,4,5,7,9,10,11,12,13,14,16,17,18,19,20])
        splits = [np.split(perm, [stroke_splits[0], 17-sum(stroke_splits[1:]),
                                  17-stroke_splits[2]])[s] for s in [0,2,3]]
        print(f'Subject splits (stroke): {splits}')
        
        dataset['stroke'] = {}
        for name, split in zip(['tr','va','te'], splits):
            file_names = sum([
                sorted(
                    glob.glob(f'{root_dir}/data_stroke_th{win_th:0.1f}_singleP2P_man/S{s}/*_*_*.npy'),
                    key=lambda x: int(os.path.basename(x).split('_')[1])
                ) for s in split], [])
            dataset['stroke'][name] = CustomDataset(file_names, is_synth=False, is_stroke=True, augment=with_aug,
                                                    with_gyro=with_gyro, with_acc=with_acc)
    
    # Create synthetic dataset if necessary
    if sum(synth_splits) > 0:
        print(f'Creating synthetic dataset: {synth_splits}')
        rng_synth = np.random.default_rng(seed)
        perm = rng_synth.permutation([1,2,3,4,5,6,8,10,11,12,13,14,15,16,17,18,19,20]) # removing 7
        splits = [np.split(perm, [synth_splits[0], 18-sum(synth_splits[1:]),
                                  18-synth_splits[2]])[s] for s in [0,2,3]]
        print(f'Subject splits (synthetic): {splits}')
        
        dataset['synth'] = {}
        for name, split in zip(['tr','va','te'], splits):
            file_names = sum([
                list(rng_synth.choice(np.array(sorted(
                    glob.glob('/work/pi_sunghoonlee_umass_edu/Ignacio/P2P-classifier/synthetic_dataset'
                              + f'/S{s}_3s50Hz/*_*_*.npy'),
                    key=lambda x: int(os.path.basename(x).split('_')[1])
                )).reshape(-1,4), int(synth_size * 2e4 / len(split)), False).reshape(-1)) for s in split], []) # 1->len(split)
            dataset['synth'][name] = CustomDataset(file_names, is_synth=True, augment=with_aug,
                                                   with_gyro=with_gyro, with_acc=with_acc)
    
    train_dataset = ConcatDataset([d['tr'] for d in dataset.values() if 'tr' in d])
    val_dataset = ConcatDataset([d['va'] for d in dataset.values() if 'va' in d])
    test_dataset = ConcatDataset([d['te'] for d in dataset.values() if 'te' in d])
    
    # Weights for imbalanced training
    global WEIGHTS
    positive_rate = np.mean([s['label'] for s in train_dataset])
    WEIGHTS = [1 / (1 - positive_rate), 1 / positive_rate]
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    if posttrained:
        test_trft_loader = DataLoader(test_trft_dataset, batch_size=batch_size, shuffle=True)
        test_teft_loader = DataLoader(test_teft_dataset, batch_size=batch_size)

    # model = CNNClassifier(compressed, dropout_prob).to(device)
    if resnet_type == 18: layers = [2, 2, 2, 2]; block = BasicBlock
    elif resnet_type == 34: layers = [3, 4, 6, 3]; block = BasicBlock
    elif resnet_type == 50: layers = [3, 4, 6, 3]; block = Bottleneck
    else: raise NotImplementedError
    ########################################################################################
    model = ResNet(block, layers, dropout_prob, with_gyro=with_gyro, with_acc=with_acc).to(device)
    # model = bbms.setup_model(bbms.config, device)
    # model = BioBank(model, outputs=1).to(device)
    ########################################################################################
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if posttrained: optimizer_ft = optim.Adam(model.fc.parameters(), lr=learning_rate/10)
    if finetune_val: optimizer_ft = optim.Adam(model.fc.parameters(), lr=learning_rate/10)
    if pretrained:
        pm_seed, pm_split, pm_issynth = tuple(pretrained_model)
        pm_dom = 'synth' if pm_issynth else 'real'
        model.load_state_dict(torch.load(f'{root_dir}/models/train_{pm_split}{pm_dom}_seed{pm_seed}.pth'))
        optimizer = optim.Adam([
            {'params': model.conv1.parameters(), 'lr': learning_rate/6},
            {'params': model.layer1.parameters(), 'lr': learning_rate/5},
            {'params': model.layer2.parameters(), 'lr': learning_rate/4},
            {'params': model.layer3.parameters(), 'lr': learning_rate/3},
            {'params': model.layer4.parameters(), 'lr': learning_rate/2},
            {'params': model.fc.parameters(), 'lr': learning_rate}
        ])
        
    elif pretrained_biobank:
        bbms.config['load_weights'] = True
        model = bbms.setup_model(bbms.config, device)
        optimizer = optim.Adam([
            {'params': model.feature_extractor.layer1.parameters(), 'lr': learning_rate/6},
            {'params': model.feature_extractor.layer2.parameters(), 'lr': learning_rate/5},
            {'params': model.feature_extractor.layer3.parameters(), 'lr': learning_rate/4},
            {'params': model.feature_extractor.layer4.parameters(), 'lr': learning_rate/3},
            {'params': model.feature_extractor.layer5.parameters(), 'lr': learning_rate/2},
            {'params': model.classifier.parameters(), 'lr': learning_rate}
        ])

    best_metric = 0
    
    # Epoch loop
    for epoch in range(num_epochs):
        # Training
        train_metrics = run_epoch(model, train_loader, criterion, optimizer)
        
        # Validation
        val_metrics = run_epoch(model, val_loader, criterion, split='val')
        thr = val_metrics[validate_thr]
        val_metrics = run_epoch(model, val_loader, criterion, split='val', threshold=thr)
        
        # Testing
        test_metrics = run_epoch(model, test_loader, criterion, split='test', threshold=thr)
        
        # Log metrics
        if epoch < num_epochs - 1 and 'df' in train_metrics.keys():
            train_metrics.pop('df')
            val_metrics.pop('df')
            test_metrics.pop('df')
        all_metrics = {'train': train_metrics, 'val': val_metrics, 'test': test_metrics}
        metrics_names = train_metrics.keys()
        wandb.log(dict([(f'{split}_{k}', all_metrics[split][k]) for k in metrics_names
                        for split in all_metrics.keys()]))
        
        ########################################################################################
        if val_metrics['auprc_unb'] > best_metric:
            best_metric = val_metrics['auprc_unb']
            if save_model: torch.save(model.state_dict(),
                                      f'{root_dir}/models/train_{synth_splits[0]}synth_seed{seed}.pth')
        ########################################################################################
        
        print(f'################ Epoch {epoch:02d} ################')
        print(train_metrics['f1'], train_metrics['f1_unb'])
        print(val_metrics['f1'], val_metrics['f1_unb'])
        print(test_metrics['f1'], test_metrics['f1_unb'])
        print('##########################################')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--random_seed', type=int, default=1, help='Random seed')
    parser.add_argument('--synth_size', type=float, default=1.0, help='Synthetic size')
    # parser.add_argument('--real_size', type=float, default=0.3, help='Real size')
    # parser.add_argument('--stroke_size', type=float, default=0.0, help='Stroke size')
    parser.add_argument('--synth_splits', nargs='+', type=int, default=[0,0,0], help='Synth splits train/val/test')
    parser.add_argument('--real_splits', nargs='+', type=int, default=[0,0,0], help='Real splits train/val/test')
    parser.add_argument('--stroke_splits', nargs='+', type=int, default=[0,0,0], help='Stroke splits train/val/test')
    parser.add_argument('--pretr_model', nargs=3, type=int, default=[0,0,0],
                        help='Pretrained model seed/num_subj/is_synth')
    args = parser.parse_args()
    
    root_dir = '/home/igavier_umass_edu/Documents/P2P-classifier'

    # Define sweep config
    sweep_configuration = {
        'method': 'grid',
        'name': 'train_real',
        'metric': {'goal': 'maximize', 'name': 'val_f1'},
        'parameters': {
            'batch_size': {'values': [64]},
            'epochs': {'values': [30]},
            'lr': {'values': [1e-4]},
            'random_seed': {'value': args.random_seed},
            'dropout_prob': {'value': 0.2},
            'synth_size': {'value': args.synth_size},
            # 'real_size': {'value': args.real_size},
            # 'stroke_size': {'value': args.stroke_size},
            'synth_splits': {'value': args.synth_splits},
            'real_splits': {'value': args.real_splits},
            'stroke_splits': {'value': args.stroke_splits},
            'windows_thresholds': {'value': 0.4},
            'per_subj': {'value': 1},
            'len_threshold': {'values': [0]},
            'ener_threshold': {'values': [0]},
            'with_aug': {'value': 0},
            'with_gyro': {'value': 0},
            'with_acc': {'value': 1},
            'validate_thr': {'value': 'bestth'},
            'compressed': {'value': 0},
            'resnet_type': {'value': 34},
            'pretrained': {'value': 1},
            'pretrained_model': {'value': args.pretr_model},
            'pretrained_biobank': {'value': 0},
            'posttrained': {'value': 0},
            'finetune_val': {'value': 0},
            'save_model': {'value': 0}
        },
    }

    # Initialize sweep by passing in config.
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='P2P-classifier')
    
    # Start sweep job.
    wandb.agent(sweep_id, function=main, count=1)
    
    # main()