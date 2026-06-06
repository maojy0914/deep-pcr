
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import MultiModalDataset
from model.model_concat import MultiModalConcatModel
from model.model_cross_attention import MultiModalCrossAttentionModel
from model.model_coattention import MultiModalCoAttentionModel
from model.model_deeppcr import DeepPCRModel
from model.clinical_model import ClinicalMLP
from utils import (
    compute_metrics,
    save_results_fold,
    save_metrics_csv,
    set_seed
)


def get_model(model_name, args):
    if model_name == 'concat':
        model = MultiModalConcatModel(
            path_embedding_dim=args.path_embedding_dim,
            text_embedding_dim=args.text_embedding_dim,
            hidden_dim=args.projection_dim,
            dropout=args.dropout,
            n_classes=args.n_classes
        )
    elif model_name == 'cross_attention':
        model = MultiModalCrossAttentionModel(
            path_embedding_dim=args.path_embedding_dim,
            text_embedding_dim=args.text_embedding_dim,
            hidden_dim=args.projection_dim,
            num_heads=args.num_heads,
            dropout=args.dropout,
            n_classes=args.n_classes
        )
    elif model_name == 'co_attention':
        model = MultiModalCoAttentionModel(
            path_embedding_dim=args.path_embedding_dim,
            text_embedding_dim=args.text_embedding_dim,
            hidden_dim=args.projection_dim,
            num_heads=args.num_heads,
            dropout=args.dropout,
            n_classes=args.n_classes
        )
    elif model_name == 'deeppcr':
        model = DeepPCRModel(
            path_embedding_dim=args.path_embedding_dim,
            text_embedding_dim=args.text_embedding_dim,
            hidden_dim=args.projection_dim,
            dropout=args.dropout,
            n_classes=args.n_classes
        )
    elif model_name == 'clinical':
        model = ClinicalMLP(
            input_dim=args.text_embedding_dim,
            hidden_dim=args.clinical_hidden_dim,
            n_classes=args.n_classes,
            dropout=args.dropout if hasattr(args, 'dropout') else 0.25
        )
    else:
        raise ValueError('Unknown model: {}'.format(model_name))

    return model


def train_epoch(model, train_loader, criterion, optimizer, device, model_name):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in train_loader:
        text_features = batch['embedding'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        if model_name == 'clinical':
            logits = model(text_features)
        else:
            path_features = batch['features'].to(device)
            logits = model(path_features, text_features)

        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def test_epoch(model, test_loader, criterion, device, model_name):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    slide_ids = []
    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for batch in test_loader:
            text_features = batch['embedding'].to(device)
            labels = batch['label'].to(device)

            if model_name == 'clinical':
                logits = model(text_features)
            else:
                path_features = batch['features'].to(device)
                logits = model(path_features, text_features)

            loss = criterion(logits, labels)

            total_loss += loss.item()
            num_batches += 1

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            slide_ids.extend(batch['slide_id'])
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs[:, 1].cpu().numpy())

    return total_loss / num_batches, slide_ids, np.array(y_true), np.array(y_pred), np.array(y_prob)


def train_fold(args, fold, device):
    print('=' * 50)
    print('Fold {}'.format(fold))

    split_path = os.path.join(args.splits_dir, 'split{}.csv'.format(fold))

    train_dataset = MultiModalDataset(
        split_csv=split_path,
        features_dir=args.features_dir,
        embedding_dir=args.embedding_dir,
        clinical_csv=args.clinical_csv,
        mode='train'
    )

    test_dataset = MultiModalDataset(
        split_csv=split_path,
        features_dir=args.features_dir,
        embedding_dir=args.embedding_dir,
        clinical_csv=args.clinical_csv,
        mode='test'
    )

    print('Train: {} samples, Test: {} samples'.format(len(train_dataset), len(test_dataset)))

    # DataLoader (batch_size=1)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = get_model(args.model_name, args)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_auroc = 0.0
    best_metrics = None
    best_model_state = None

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, args.model_name)
        test_loss, slide_ids, y_true, y_pred, y_prob = test_epoch(model, test_loader, criterion, device, args.model_name)

        metrics = compute_metrics(y_true, y_pred, y_prob)
        metrics['loss'] = test_loss

        print('Epoch {}/{} - Train Loss: {:.4f}, Test Loss: {:.4f}, Acc: {:.4f}, AUROC: {:.4f}'.format(
            epoch + 1, args.epochs, train_loss, test_loss, metrics['accuracy'], metrics['auroc']))

        if metrics['auroc'] > best_auroc:
            best_auroc = metrics['auroc']
            best_metrics = metrics.copy()
            best_model_state = model.state_dict().copy()
            print('  -> New best AUROC: {:.4f}'.format(best_auroc))

    best_model_path = os.path.join(args.exp_dir, 'best_model_fold{}.pth'.format(fold))
    torch.save(best_model_state, best_model_path)
    print('Best model saved to {}'.format(best_model_path))

    results_path = os.path.join(args.exp_dir, 'results_fold{}.npy'.format(fold))
    save_results_fold(results_path, slide_ids, y_true, y_pred)
    print('Results saved to {}'.format(results_path))

    return best_metrics


def main():
    parser = argparse.ArgumentParser(description='Train multimodal model')

    parser.add_argument('--clinical_csv', type=str)
    parser.add_argument('--features_dir', type=str)
    parser.add_argument('--embedding_dir', type=str)
    parser.add_argument('--splits_dir', type=str)


    parser.add_argument('--model_name', type=str, default='concat', choices=['concat', 'cross_attention', 'co_attention', 'deeppcr', 'clinical'])
    parser.add_argument('--path_embedding_dim', type=int, default=1536)
    parser.add_argument('--text_embedding_dim', type=int, default=512)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--projection_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--gate', type=bool, default=True)
    parser.add_argument('--size_arg', type=str, default='big')
    parser.add_argument('--dropout', type=float, default=0.25)


    parser.add_argument('--clinical_hidden_dim', type=int, default=256)


    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_folds', type=int, default=5)

    parser.add_argument('--results_dir', type=str)
    parser.add_argument('--exp_name', type=str, default='')

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: {}'.format(device))

    if args.exp_name:
        args.exp_dir = os.path.join(args.results_dir, args.exp_name)
    else:
        args.exp_dir = args.results_dir
    os.makedirs(args.exp_dir, exist_ok=True)
    print('Experiment directory: {}'.format(args.exp_dir))

    all_metrics = []
    for fold in range(1, args.num_folds + 1):
        best_metrics = train_fold(args, fold, device)

        best_metrics['fold'] = fold
        all_metrics.append(best_metrics)

    metrics_path = os.path.join(args.exp_dir, 'metrics.csv')
    save_metrics_csv(metrics_path, all_metrics)
    print('Metrics saved to {}'.format(metrics_path))

    print('\n' + '=' * 50)
    print('Summary:')
    for m in all_metrics:
        print('Fold {}: Acc={:.4f}, AUROC={:.4f}, AUPRC={:.4f}'.format(
            m['fold'], m['accuracy'], m['auroc'], m['auprc']))

    avg_acc = np.mean([m['accuracy'] for m in all_metrics])
    avg_auroc = np.mean([m['auroc'] for m in all_metrics])
    avg_auprc = np.mean([m['auprc'] for m in all_metrics])
    print('Average: Acc={:.4f}, AUROC={:.4f}, AUPRC={:.4f}'.format(avg_acc, avg_auroc, avg_auprc))


if __name__ == '__main__':
    main()