#!/usr/bin/env python

# PhysioNet Challenge 2026 - Cognitive Impairment Prediction from PSG
# Model: Raw PSG epoch encoder (1D CNN per modality) + BiGRU + attention pooling
#        + handcrafted features + CAISR algorithmic features + MLP fusion
# Team: Rishabh Jha, University of Victoria MCV Lab
#
# Key properties for the official harness:
#   * Required signatures unchanged: train_model / load_model / run_model
#   * Memory-safe training: raw epoch tensors are cached to disk (float16,
#     compressed) and loaded lazily per-batch, so RAM ~= batch_size x tensor
#     size rather than (num_records x tensor size).
#   * Internal stratified val split for model selection (falls back to train
#     loss when there are too few records).

import os
import sys
import shutil
import tempfile
import random
import joblib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from tqdm import tqdm
from scipy import signal as scipy_signal
from scipy.stats import skew, kurtosis, entropy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

from helper_code import *

################################################################################
# Paths and constants  (tune the KNOBS block if the harness is time-limited)
################################################################################

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, 'channel_table.csv')

TARGET_FS = 64
MAX_SECONDS = 3600
EPOCH_SECONDS = 30
EPOCH_SAMPLES = TARGET_FS * EPOCH_SECONDS      # 1920
MAX_EPOCHS = MAX_SECONDS // EPOCH_SECONDS       # 120

NUM_SIGNAL_GROUPS = 7
NUM_FEATURES_PER_CHANNEL = 20
NUM_ALGO_FEATURES = 12
NUM_HANDCRAFTED_FEATURES = NUM_SIGNAL_GROUPS * NUM_FEATURES_PER_CHANNEL + NUM_ALGO_FEATURES  # 152

# ---- KNOBS: reduce these first if the harness times out on CPU ----
NUM_EPOCHS = 20
BATCH_SIZE = 8
CACHE_DTYPE = np.float16   # on-disk dtype for cached raw epochs (float32 if you hit precision issues)
VAL_FRACTION = 0.15        # internal holdout for model selection
MIN_RECORDS_FOR_VAL = 40   # below this, select on train loss instead
# -------------------------------------------------------------------

MODALITY_ORDER = ['eeg', 'eog', 'chin', 'leg', 'ecg', 'resp', 'spo2']
MODALITY_TO_INDEX = {k: i for i, k in enumerate(MODALITY_ORDER)}

LEADS_TO_CHECK = {
    'eeg':  ['c3-m2', 'c4-m1', 'f3-m2', 'f4-m1', 'o1-m2', 'o2-m1'],
    'eog':  ['e1-m2', 'e2-m1'],
    'chin': ['chin1-chin2', 'chin'],
    'leg':  ['lat', 'rat'],
    'ecg':  ['ecg', 'ekg'],
    'resp': ['airflow', 'ptaf', 'abd', 'chest'],
    'spo2': ['spo2', 'sao2'],
}

BIPOLAR_CONFIGS = [
    ('f3-m2', 'f3', ['m2']), ('f4-m1', 'f4', ['m1']),
    ('c3-m2', 'c3', ['m2']), ('c4-m1', 'c4', ['m1']),
    ('o1-m2', 'o1', ['m2']), ('o2-m1', 'o2', ['m1']),
    ('e1-m2', 'e1', ['m2']), ('e2-m1', 'e2', ['m1']),
    ('chin1-chin2', 'chin 1', ['chin 2']),
    ('lat', 'lleg+', ['lleg-']), ('rat', 'rleg+', ['rleg-'])
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

################################################################################
# Utilities
################################################################################

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def safe_np(x, dtype=np.float32):
    x = np.asarray(x, dtype=dtype)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x

def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))

class Standardizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x):
        x = np.asarray(x, dtype=np.float32)
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
        self.std[self.std < 1e-6] = 1.0

    def transform(self, x):
        x = np.asarray(x, dtype=np.float32)
        return (x - self.mean) / self.std

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

def resample_signal(sig, original_fs, target_fs):
    """Resample a 1D signal via linear interpolation."""
    sig = np.asarray(sig, dtype=np.float32)
    if abs(original_fs - target_fs) < 0.01:
        return sig
    original_length = len(sig)
    if original_length < 2:
        return sig
    target_length = int(original_length * target_fs / original_fs)
    if target_length <= 0:
        return np.zeros(1, dtype=np.float32)
    x_original = np.linspace(0.0, 1.0, original_length, endpoint=False)
    x_target = np.linspace(0.0, 1.0, target_length, endpoint=False)
    return np.interp(x_target, x_original, sig).astype(np.float32)

################################################################################
# Neural model
################################################################################

class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.proj = None
        if in_ch != out_ch or stride != 1:
            self.proj = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )

    def forward(self, x):
        identity = x if self.proj is None else self.proj(x)
        out = self.block(x)
        return out + identity

class ModalityEpochEncoder(nn.Module):
    def __init__(self, in_ch=1, base=24, emb_dim=64, dropout=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base),
            nn.GELU()
        )
        self.layer1 = ConvBlock1D(base, base, kernel_size=7, stride=2, dropout=dropout)
        self.layer2 = ConvBlock1D(base, base * 2, kernel_size=7, stride=2, dropout=dropout)
        self.layer3 = ConvBlock1D(base * 2, base * 2, kernel_size=5, stride=2, dropout=dropout)
        self.layer4 = ConvBlock1D(base * 2, emb_dim, kernel_size=5, stride=2, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).squeeze(-1)
        return x

class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1)
        )

    def forward(self, x, mask):
        scores = self.attn(x).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled, weights

class PSGHybridNet(nn.Module):
    def __init__(self, handcrafted_dim, emb_dim=64, seq_hidden=128, dropout=0.2):
        super().__init__()
        self.modality_encoders = nn.ModuleList([
            ModalityEpochEncoder(in_ch=1, base=24, emb_dim=emb_dim, dropout=0.1)
            for _ in range(NUM_SIGNAL_GROUPS)
        ])

        self.modality_gate = nn.Sequential(
            nn.Linear(NUM_SIGNAL_GROUPS, NUM_SIGNAL_GROUPS),
            nn.Sigmoid()
        )

        self.sequence_model = nn.GRU(
            input_size=NUM_SIGNAL_GROUPS * emb_dim + NUM_SIGNAL_GROUPS,
            hidden_size=seq_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        self.attn_pool = AttentionPooling(seq_hidden * 2)

        self.handcrafted_branch = nn.Sequential(
            nn.Linear(handcrafted_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.fusion = nn.Sequential(
            nn.Linear(seq_hidden * 2 + 64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x_raw, x_hand, epoch_mask, modality_mask):
        b, t, m, l = x_raw.shape
        modality_embs = []
        for k in range(m):
            xk = x_raw[:, :, k, :].reshape(b * t, 1, l)
            ek = self.modality_encoders[k](xk).reshape(b, t, -1)
            modality_embs.append(ek)

        modality_embs = torch.stack(modality_embs, dim=2)          # (b, t, m, emb)
        gate = self.modality_gate(modality_mask.float())           # (b, m)
        gated = modality_embs * gate.unsqueeze(1).unsqueeze(-1)     # (b, t, m, emb)

        seq_feat = gated.reshape(b, t, -1)                         # (b, t, m*emb)
        seq_input = torch.cat([seq_feat, modality_mask.unsqueeze(1).repeat(1, t, 1)], dim=-1)

        seq_out, _ = self.sequence_model(seq_input)
        pooled, _ = self.attn_pool(seq_out, epoch_mask)

        hand = self.handcrafted_branch(x_hand)
        fused = torch.cat([pooled, hand], dim=-1)
        logit = self.fusion(fused).squeeze(-1)
        return logit

################################################################################
# On-disk record cache (keeps training RAM bounded)
################################################################################

def cache_record(cache_dir, idx, epoch_sample):
    """Persist one record's raw-epoch tensor + masks to a compressed npz."""
    path = os.path.join(cache_dir, f"rec_{idx:07d}.npz")
    np.savez_compressed(
        path,
        x_raw=epoch_sample['x_raw'].astype(CACHE_DTYPE),
        epoch_mask=epoch_sample['epoch_mask'].astype(np.float32),
        modality_mask=epoch_sample['modality_mask'].astype(np.float32),
    )
    return path

class PSGDataset(Dataset):
    """Loads raw-epoch tensors lazily from disk; handcrafted feats stay in RAM."""
    def __init__(self, cache_paths, handcrafted, labels):
        self.cache_paths = cache_paths
        self.handcrafted = handcrafted
        self.labels = labels

    def __len__(self):
        return len(self.cache_paths)

    def __getitem__(self, idx):
        with np.load(self.cache_paths[idx]) as d:
            x_raw = d['x_raw'].astype(np.float32)
            epoch_mask = d['epoch_mask'].astype(np.float32)
            modality_mask = d['modality_mask'].astype(np.float32)
        return (
            torch.from_numpy(x_raw),
            torch.from_numpy(self.handcrafted[idx].astype(np.float32)),
            torch.from_numpy(epoch_mask),
            torch.from_numpy(modality_mask),
            torch.tensor(float(self.labels[idx]), dtype=torch.float32),
        )

################################################################################
# Required functions
################################################################################

def train_model(data_folder, model_folder, verbose, csv_path=DEFAULT_CSV_PATH):
    seed_everything(42)
    os.makedirs(model_folder, exist_ok=True)

    if verbose:
        print('Finding the Challenge data...')

    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_metadata_list = find_patients(patient_data_file)
    num_records = len(patient_metadata_list)
    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    if verbose:
        print(f'Found {num_records} records. Extracting features and caching epochs...')

    cache_dir = tempfile.mkdtemp(prefix='psg_cache_')
    try:
        cache_paths = []
        handcrafted_features = []
        labels = []

        pbar = tqdm(range(num_records), desc='Preparing data', unit='rec', disable=not verbose)
        for i in pbar:
            try:
                record = patient_metadata_list[i]
                patient_id = record[HEADERS['bids_folder']]
                site_id = record[HEADERS['site_id']]
                session_id = record[HEADERS['session_id']]

                label = load_diagnoses(patient_data_file, patient_id)
                if label not in (0, 1):
                    continue

                phys_file = os.path.join(
                    data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER, site_id,
                    f"{patient_id}_ses-{session_id}.edf"
                )
                if not os.path.exists(phys_file):
                    continue

                physiological_data, physiological_fs = load_signal_data(phys_file)

                algo_file = os.path.join(
                    data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER, site_id,
                    f"{patient_id}_ses-{session_id}_caisr_annotations.edf"
                )
                if os.path.exists(algo_file):
                    algo_data, _ = load_signal_data(algo_file)
                    algo_features = extract_algorithmic_annotations_features(algo_data)
                else:
                    algo_features = np.zeros(NUM_ALGO_FEATURES, dtype=np.float32)

                selected = build_selected_modalities(physiological_data, physiological_fs, csv_path)
                epoch_sample = create_epoch_tensor(selected)
                phys_features = extract_physiological_features_from_selected(selected)

                hand = np.hstack([phys_features, algo_features]).astype(np.float32)
                hand = np.nan_to_num(hand, nan=0.0, posinf=0.0, neginf=0.0)

                path = cache_record(cache_dir, len(cache_paths), epoch_sample)
                cache_paths.append(path)
                handcrafted_features.append(hand)
                labels.append(float(label))

                del physiological_data, selected, epoch_sample

            except Exception as e:
                if verbose:
                    tqdm.write(f'Error processing record {i+1}: {e}')
                continue
        pbar.close()

        if len(labels) == 0:
            raise ValueError('No valid records found for training.')

        handcrafted_features = safe_np(handcrafted_features)
        if handcrafted_features.ndim == 1:
            handcrafted_features = handcrafted_features.reshape(1, -1)
        labels = safe_np(labels)

        if verbose:
            pos = int(labels.sum())
            neg = len(labels) - pos
            print(f'Training set: {len(labels)} records ({pos} positive, {neg} negative)')
            print(f'Handcrafted feature dimension: {handcrafted_features.shape[1]}')

        scaler = Standardizer()
        handcrafted_scaled = scaler.fit_transform(handcrafted_features)

        full_dataset = PSGDataset(cache_paths, handcrafted_scaled, labels)

        # ---- Internal stratified split for model selection ----
        use_val = len(labels) >= MIN_RECORDS_FOR_VAL and 0 < labels.sum() < len(labels)
        if use_val:
            try:
                from sklearn.model_selection import train_test_split
                idx_all = np.arange(len(labels))
                train_idx, val_idx = train_test_split(
                    idx_all, test_size=VAL_FRACTION, random_state=42, stratify=labels
                )
                train_loader = DataLoader(Subset(full_dataset, train_idx.tolist()),
                                          batch_size=BATCH_SIZE, shuffle=True,
                                          num_workers=0, drop_last=False)
                val_loader = DataLoader(Subset(full_dataset, val_idx.tolist()),
                                        batch_size=BATCH_SIZE, shuffle=False,
                                        num_workers=0, drop_last=False)
                pos_count = float(labels[train_idx].sum())
                neg_count = float(len(train_idx) - labels[train_idx].sum())
            except Exception:
                use_val = False

        if not use_val:
            train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                      num_workers=0, drop_last=False)
            val_loader = None
            pos_count = float(labels.sum())
            neg_count = float(len(labels) - labels.sum())

        model = PSGHybridNet(handcrafted_dim=handcrafted_features.shape[1],
                             emb_dim=64, seq_hidden=128, dropout=0.2).to(DEVICE)

        pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)], dtype=torch.float32, device=DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        best_state = None
        best_metric = float('inf')
        patience, patience_ctr = 6, 0

        if verbose:
            print(f'Training neural model on {DEVICE} '
                  f'({"train/val" if val_loader else "train-loss"} selection)...')

        for epoch in range(NUM_EPOCHS):
            model.train()
            running_loss, num_seen = 0.0, 0
            for x_raw, x_hand, epoch_mask, modality_mask, y in train_loader:
                x_raw = x_raw.to(DEVICE); x_hand = x_hand.to(DEVICE)
                epoch_mask = epoch_mask.to(DEVICE); modality_mask = modality_mask.to(DEVICE)
                y = y.to(DEVICE)

                optimizer.zero_grad()
                logits = model(x_raw, x_hand, epoch_mask, modality_mask)
                loss = criterion(logits, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                optimizer.step()

                running_loss += loss.item() * y.size(0)
                num_seen += y.size(0)
            scheduler.step()
            train_loss = running_loss / max(num_seen, 1)

            if val_loader is not None:
                model.eval()
                v_loss, v_seen = 0.0, 0
                with torch.no_grad():
                    for x_raw, x_hand, epoch_mask, modality_mask, y in val_loader:
                        x_raw = x_raw.to(DEVICE); x_hand = x_hand.to(DEVICE)
                        epoch_mask = epoch_mask.to(DEVICE); modality_mask = modality_mask.to(DEVICE)
                        y = y.to(DEVICE)
                        logits = model(x_raw, x_hand, epoch_mask, modality_mask)
                        v_loss += criterion(logits, y).item() * y.size(0)
                        v_seen += y.size(0)
                sel_metric = v_loss / max(v_seen, 1)
                if verbose:
                    print(f'Epoch {epoch+1:02d}/{NUM_EPOCHS} - train {train_loss:.5f} - val {sel_metric:.5f}')
            else:
                sel_metric = train_loss
                if verbose:
                    print(f'Epoch {epoch+1:02d}/{NUM_EPOCHS} - loss {sel_metric:.5f}')

            if sel_metric < best_metric:
                best_metric = sel_metric
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    if verbose:
                        print('Early stopping.')
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        save_neural_model(model_folder, model, scaler, handcrafted_features.shape[1], csv_path)
        if verbose:
            print('Done. Model saved.')
            print()

    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def load_model(model_folder, verbose):
    model_file = os.path.join(model_folder, 'model.sav')
    payload = joblib.load(model_file)

    model = PSGHybridNet(
        handcrafted_dim=payload['handcrafted_dim'],
        emb_dim=payload['config']['emb_dim'],
        seq_hidden=payload['config']['seq_hidden'],
        dropout=payload['config']['dropout'],
    )
    model.load_state_dict(payload['state_dict'])
    model.to(DEVICE)
    model.eval()

    payload['torch_model'] = model
    return payload


def run_model(model, record, data_folder, verbose):
    torch_model = model['torch_model']
    scaler = model['scaler']
    csv_path = model.get('csv_path', DEFAULT_CSV_PATH)

    patient_id = record[HEADERS['bids_folder']]
    site_id = record[HEADERS['site_id']]
    session_id = record[HEADERS['session_id']]

    phys_file = os.path.join(
        data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER, site_id,
        f"{patient_id}_ses-{session_id}.edf"
    )
    if os.path.exists(phys_file):
        physiological_data, physiological_fs = load_signal_data(phys_file)
        selected = build_selected_modalities(physiological_data, physiological_fs, csv_path)
        epoch_sample = create_epoch_tensor(selected)
        phys_features = extract_physiological_features_from_selected(selected)
    else:
        epoch_sample = {
            'x_raw': np.zeros((MAX_EPOCHS, NUM_SIGNAL_GROUPS, EPOCH_SAMPLES), dtype=np.float32),
            'epoch_mask': np.zeros((MAX_EPOCHS,), dtype=np.float32),
            'modality_mask': np.zeros((NUM_SIGNAL_GROUPS,), dtype=np.float32),
        }
        phys_features = np.zeros(NUM_SIGNAL_GROUPS * NUM_FEATURES_PER_CHANNEL, dtype=np.float32)

    algo_file = os.path.join(
        data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER, site_id,
        f"{patient_id}_ses-{session_id}_caisr_annotations.edf"
    )
    if os.path.exists(algo_file):
        algo_data, _ = load_signal_data(algo_file)
        algo_features = extract_algorithmic_annotations_features(algo_data)
    else:
        algo_features = np.zeros(NUM_ALGO_FEATURES, dtype=np.float32)

    hand = np.hstack([phys_features, algo_features]).astype(np.float32)
    hand = np.nan_to_num(hand, nan=0.0, posinf=0.0, neginf=0.0)
    hand = scaler.transform(hand.reshape(1, -1))[0]

    x_raw = torch.tensor(epoch_sample['x_raw'][None, ...], dtype=torch.float32, device=DEVICE)
    x_hand = torch.tensor(hand[None, ...], dtype=torch.float32, device=DEVICE)
    epoch_mask = torch.tensor(epoch_sample['epoch_mask'][None, ...], dtype=torch.float32, device=DEVICE)
    modality_mask = torch.tensor(epoch_sample['modality_mask'][None, ...], dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        logit = torch_model(x_raw, x_hand, epoch_mask, modality_mask).cpu().numpy().reshape(-1)[0]
        prob = float(sigmoid_np(logit))
        binary = int(prob >= 0.5)

    return binary, prob

################################################################################
# Signal preprocessing and modality construction
################################################################################

def build_selected_modalities(physiological_data, physiological_fs, csv_path=DEFAULT_CSV_PATH):
    original_labels = list(physiological_data.keys())
    rename_rules = load_rename_rules(os.path.abspath(csv_path))
    rename_map, cols_to_drop = standardize_channel_names_rename_only(original_labels, rename_rules)

    processed_channels = {}
    processed_fs = {}
    for old_label, data in physiological_data.items():
        if old_label in cols_to_drop:
            continue
        new_label = rename_map.get(old_label, old_label.lower())
        processed_channels[new_label] = safe_np(data)
        if old_label in physiological_fs:
            processed_fs[new_label] = physiological_fs[old_label]
        else:
            raise KeyError(f"Sampling frequency not found for channel '{old_label}'")

    for target, pos, neg_list in BIPOLAR_CONFIGS:
        if target in processed_channels or pos not in processed_channels:
            continue
        if not all(n in processed_channels for n in neg_list):
            continue
        involved = [pos] + neg_list
        fs_values = [processed_fs[ch] for ch in involved]
        if len(set(fs_values)) > 1:
            continue
        ref_sig = processed_channels[neg_list[0]] if len(neg_list) == 1 \
            else tuple(processed_channels[n] for n in neg_list)
        derived = derive_bipolar_signal(processed_channels[pos], ref_sig)
        if derived is not None:
            processed_channels[target] = safe_np(derived)
            processed_fs[target] = processed_fs[pos]

    selected = {}
    for lead_type, candidates in LEADS_TO_CHECK.items():
        sig, fs = None, None
        for candidate in candidates:
            if candidate in processed_channels and processed_channels[candidate] is not None:
                sig = processed_channels[candidate]
                fs = processed_fs.get(candidate, None)
                break
        if sig is not None and fs is not None and len(sig) > 1 and fs > 0:
            resampled = resample_signal(sig, fs, TARGET_FS)
            max_samples = TARGET_FS * MAX_SECONDS
            if len(resampled) > max_samples:
                resampled = resampled[:max_samples]
            selected[lead_type] = safe_np(resampled)
        else:
            selected[lead_type] = None
    return selected


def create_epoch_tensor(selected_modalities):
    x = np.zeros((MAX_EPOCHS, NUM_SIGNAL_GROUPS, EPOCH_SAMPLES), dtype=np.float32)
    epoch_mask = np.zeros((MAX_EPOCHS,), dtype=np.float32)
    modality_mask = np.zeros((NUM_SIGNAL_GROUPS,), dtype=np.float32)

    per_modality_epochs = np.zeros((NUM_SIGNAL_GROUPS,), dtype=np.int32)

    for mod_name, mod_idx in MODALITY_TO_INDEX.items():
        sig = selected_modalities.get(mod_name, None)
        if sig is None or len(sig) < EPOCH_SAMPLES:
            continue

        modality_mask[mod_idx] = 1.0
        n_epochs = min(len(sig) // EPOCH_SAMPLES, MAX_EPOCHS)
        per_modality_epochs[mod_idx] = n_epochs

        sig = sig[:n_epochs * EPOCH_SAMPLES]
        epochs = sig.reshape(n_epochs, EPOCH_SAMPLES)

        med = np.median(epochs, axis=1, keepdims=True)
        iqr = (np.percentile(epochs, 75, axis=1, keepdims=True)
               - np.percentile(epochs, 25, axis=1, keepdims=True))
        iqr[iqr < 1e-6] = 1.0
        epochs = np.clip((epochs - med) / iqr, -20.0, 20.0)
        x[:n_epochs, mod_idx, :] = epochs.astype(np.float32)

    # An epoch t is valid if at least one present modality actually has data at t.
    if modality_mask.sum() > 0:
        common_epochs = int(per_modality_epochs.max())
        common_epochs = max(1, min(common_epochs, MAX_EPOCHS))
        epoch_mask[:common_epochs] = 1.0

    return {'x_raw': x, 'epoch_mask': epoch_mask, 'modality_mask': modality_mask}

################################################################################
# Handcrafted feature extraction
################################################################################

def extract_physiological_features_from_selected(selected_modalities):
    final_features = []
    for lead_type in MODALITY_ORDER:
        sig = selected_modalities.get(lead_type, None)
        if sig is not None and len(sig) > 1:
            final_features.extend(compute_channel_features(sig, TARGET_FS))
        else:
            final_features.extend([0.0] * NUM_FEATURES_PER_CHANNEL)
    return np.array(final_features, dtype=np.float32)


def compute_channel_features(sig, fs):
    features = []
    n = len(sig)

    std_val = np.std(sig)
    mav_val = np.mean(np.abs(sig))
    rms_val = np.sqrt(np.mean(sig ** 2))
    zcr_val = np.mean(np.diff(np.sign(sig)) != 0)
    skew_val = skew(sig) if n > 2 else 0.0
    kurt_val = kurtosis(sig) if n > 2 else 0.0

    var_sig = np.var(sig)
    diff1 = np.diff(sig)
    var_d1 = np.var(diff1) if len(diff1) > 0 else 0.0
    diff2 = np.diff(diff1)
    var_d2 = np.var(diff2) if len(diff2) > 0 else 0.0
    mobility = np.sqrt(var_d1 / var_sig) if var_sig > 1e-12 else 0.0
    complexity = (np.sqrt(var_d2 / var_d1) / mobility) if (var_d1 > 1e-12 and mobility > 1e-12) else 0.0

    features.extend([std_val, mav_val, rms_val, zcr_val, skew_val, kurt_val, mobility, complexity])

    nperseg = min(4 * fs, n)
    if nperseg < 4:
        features.extend([0.0] * 10)
    else:
        freqs, psd = scipy_signal.welch(sig, fs=fs, nperseg=int(nperseg), noverlap=int(nperseg // 2))

        def band_power(f_low, f_high):
            mask = (freqs >= f_low) & (freqs <= f_high)
            return np.trapz(psd[mask], freqs[mask]) if np.any(mask) else 0.0

        delta_p = band_power(0.5, 4.0)
        theta_p = band_power(4.0, 8.0)
        alpha_p = band_power(8.0, 12.0)
        sigma_p = band_power(12.0, 15.0)
        beta_p = band_power(15.0, 30.0)

        dt_ratio = delta_p / theta_p if theta_p > 1e-12 else 0.0
        ta_ratio = theta_p / alpha_p if alpha_p > 1e-12 else 0.0

        total_power = np.trapz(psd, freqs) if len(psd) > 0 else 1e-12
        cumulative = np.cumsum(psd) * (freqs[1] - freqs[0]) if len(freqs) > 1 else np.array([0.0])
        cumulative_norm = cumulative / (total_power + 1e-12)
        se50_idx = np.searchsorted(cumulative_norm, 0.50)
        se95_idx = np.searchsorted(cumulative_norm, 0.95)
        se50 = freqs[min(se50_idx, len(freqs) - 1)]
        se95 = freqs[min(se95_idx, len(freqs) - 1)]

        psd_norm = psd / (psd.sum() + 1e-12)
        spec_entropy = entropy(psd_norm + 1e-12)

        features.extend([delta_p, theta_p, alpha_p, sigma_p, beta_p,
                         dt_ratio, ta_ratio, se50, se95, spec_entropy])

    features.extend([np.percentile(sig, 5), np.percentile(sig, 95)])
    return features


def extract_algorithmic_annotations_features(algo_data):
    if not algo_data:
        return np.zeros(NUM_ALGO_FEATURES, dtype=np.float32)

    features = []
    total_hours = len(algo_data.get('resp_caisr', [])) / 3600.0

    def count_discrete_events(key):
        if key not in algo_data or total_hours <= 0:
            return 0.0
        sig = np.asarray(algo_data[key]).astype(float)
        binary_sig = (sig > 0).astype(int)
        diff = np.diff(binary_sig, prepend=0)
        return np.count_nonzero(diff == 1) / total_hours

    features.extend([
        count_discrete_events('resp_caisr'),
        count_discrete_events('arousal_caisr'),
        count_discrete_events('limb_caisr'),
    ])

    stages = np.asarray(algo_data.get('stage_caisr', np.array([])))
    valid_stages = stages[stages < 9.0] if len(stages) > 0 else np.array([])
    if len(valid_stages) > 0:
        w_pct = np.mean(valid_stages == 5)
        r_pct = np.mean(valid_stages == 4)
        n1_pct = np.mean(valid_stages == 3)
        n2_pct = np.mean(valid_stages == 2)
        n3_pct = np.mean(valid_stages == 1)
        efficiency = np.mean((valid_stages >= 1) & (valid_stages <= 4))
    else:
        w_pct = n1_pct = n2_pct = n3_pct = r_pct = efficiency = 0.0
    features.extend([w_pct, n1_pct, n2_pct, n3_pct, r_pct, efficiency])

    prob_w = np.mean(algo_data.get('caisr_prob_w', [0]))
    prob_n3 = np.mean(algo_data.get('caisr_prob_n3', [0]))
    prob_arous = np.mean(algo_data.get('caisr_prob_arous', [0]))
    clean = lambda v: v if v < 1.0 else 0.0
    features.extend([clean(prob_w), clean(prob_n3), clean(prob_arous)])

    return np.array(features, dtype=np.float32)

################################################################################
# Save / load
################################################################################

def save_neural_model(model_folder, model, scaler, handcrafted_dim, csv_path):
    payload = {
        'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
        'handcrafted_dim': int(handcrafted_dim),
        'config': {'emb_dim': 64, 'seq_hidden': 128, 'dropout': 0.2},
        'scaler': scaler,
        'csv_path': csv_path,
    }
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(payload, filename, protocol=4)
