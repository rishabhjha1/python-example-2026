#!/usr/bin/env python

# George B. Moody PhysioNet Challenge 2026
# Screening for Cognitive Impairment During Sleep Studies
#
# Official-phase-compatible entry (revised).
#
# Key changes vs. the 0.48-AUC baseline:
#   1. Site transfer: robust per-channel normalization (median/IQR) + RELATIVE
#      spectral band powers, so features no longer encode amplifier gain / site.
#   2. Missing != zero: absent channels/blocks become NaN, not 0.0. The model
#      handles NaN natively and can learn "missingness" instead of confusing it
#      with a genuine zero value.
#   3. Model: HistGradientBoostingClassifier (regularized, early-stopped, native
#      NaN, balanced sample weights) replaces the un-regularized GBC + scaler.
#   4. Honest local signal: a SITE-GROUPED out-of-fold AUROC is printed at train
#      time; this mirrors the hidden cross-site test far better than random CV.
#   5. Decision threshold tuned on out-of-fold probabilities (helps Accuracy /
#      F-measure); AUROC/AUPRC are threshold-free and benefit from 1-3.
#
# The harness (train_model.py / run_model.py / helper_code.py) is UNCHANGED and
# only this file is edited. train/inference share ONE feature path
# (`assemble_features`) so the feature dimension can never silently drift.
#
# Requires scikit-learn >= 1.0 (HistGradientBoostingClassifier).
#
# Team: Rishabh Jha, University of Victoria MCV Lab

################################################################################
#
# Libraries
#
################################################################################

import joblib
import numpy as np
import os
import sys
from scipy import signal as scipy_signal
from scipy.stats import skew, kurtosis, entropy
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.base import clone
from sklearn.model_selection import (StratifiedKFold, StratifiedGroupKFold,
                                     cross_val_predict)
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from helper_code import *

################################################################################
# Path, toggle & dimension configuration
################################################################################

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, 'channel_table.csv')

# ---- Feature toggles -------------------------------------------------------
# Demographics are available on the hidden validation/test sets (safe at
# inference) and are clinically informative for cognitive impairment.
USE_DEMOGRAPHICS = True

# Human (expert) annotations are NOT available on the hidden sets. They must
# never enter the model feature vector, or train/inference dims will mismatch.
USE_HUMAN_ANNOTATIONS = False

# Set True while diagnosing to make the per-block try/except blocks LOUD instead
# of silently substituting NaN. Turn OFF for the real submission run.
DEBUG_EXTRACTION = False

# ---- Signal-processing constants ------------------------------------------
TARGET_FS   = 64     # Resample every channel to 64 Hz
MAX_SECONDS = 3600   # Use up to 1 hour of recording

# ---- Feature dimensions (single source of truth) --------------------------
NUM_FEATURES_PER_CHANNEL = 20
NUM_SIGNAL_GROUPS        = 7
NUM_PHYS_FEATURES        = NUM_SIGNAL_GROUPS * NUM_FEATURES_PER_CHANNEL   # 140
NUM_ALGO_FEATURES        = 12
NUM_DEMOGRAPHIC_FEATURES = 10    # age(1) + sex(3) + race(5) + bmi(1)


def _total_feature_dim():
    dim = NUM_PHYS_FEATURES + NUM_ALGO_FEATURES
    if USE_DEMOGRAPHICS:
        dim += NUM_DEMOGRAPHIC_FEATURES
    return dim


def _debug(msg):
    if DEBUG_EXTRACTION:
        tqdm.write(msg)


def _sanitize(x):
    """Map +/-inf to NaN, leave real values and NaN untouched (missing stays
    missing so the model can use it)."""
    x = np.asarray(x, dtype=np.float32)
    return np.where(np.isfinite(x), x, np.nan).astype(np.float32)


def _build_model():
    """Regularized, early-stopped, NaN-native gradient boosting."""
    return HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.05,
        max_depth=3,               # shallow trees -> less memorization
        max_leaf_nodes=15,
        l2_regularization=1.0,
        min_samples_leaf=20,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42,
    )


################################################################################
#
# Required functions. Do NOT change the arguments of these functions.
#
################################################################################

def train_model(data_folder, model_folder, verbose, csv_path=DEFAULT_CSV_PATH):
    """Train a HistGradientBoosting model on PSG (+ demographic) features."""

    if verbose:
        print('Finding the Challenge data...')

    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_metadata_list = find_patients(patient_data_file)
    num_records = len(patient_metadata_list)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    if verbose:
        print(f'Found {num_records} records. Extracting features and labels...')

    diagnosis_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)

    features = []
    labels = []
    site_groups = []   # for site-grouped out-of-fold estimation

    pbar = tqdm(range(num_records), desc="Extracting Features", unit="rec",
                disable=not verbose)
    for i in pbar:
        patient_id = None
        try:
            record = patient_metadata_list[i]
            patient_id = record[HEADERS['bids_folder']]
            site_id    = record[HEADERS['site_id']]
            session_id = record[HEADERS['session_id']]

            if verbose:
                pbar.set_postfix({"patient": patient_id})

            # --- Label (training only; withheld on hidden sets) ---
            label = load_diagnoses(diagnosis_file, patient_id)
            if label != 0 and label != 1:
                continue  # skip unlabeled / ambiguous records

            # --- Require the physiological signal to exist ---
            phys_file = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER,
                                     site_id, f"{patient_id}_ses-{session_id}.edf")
            if not os.path.exists(phys_file):
                _debug(f"  ! Missing physiological data for {patient_id}. Skipping...")
                continue

            # --- Single, shared feature-assembly path ---
            feature_vec = assemble_features(record, data_folder, csv_path=csv_path)

            # Optional: human expert annotations (train-only). Computed for
            # experimentation but NOT appended to the model vector by default.
            if USE_HUMAN_ANNOTATIONS:
                human_file = os.path.join(data_folder, HUMAN_ANNOTATIONS_SUBFOLDER,
                                          site_id, f"{patient_id}_ses-{session_id}_expert_annotations.edf")
                if os.path.exists(human_file):
                    human_data, _ = load_signal_data(human_file)
                    _ = extract_human_annotations_features(human_data)  # noqa: F841

            features.append(feature_vec)
            labels.append(int(label))
            site_groups.append(str(site_id))

        except Exception as e:
            tqdm.write(f"  !!! Error processing record {i + 1} ({patient_id}): {e}")
            continue

    pbar.close()

    if len(labels) == 0:
        raise ValueError("No valid labeled records found for training.")

    features = _sanitize(np.asarray(features, dtype=np.float32))   # inf -> NaN, keep NaN
    labels   = np.asarray(labels, dtype=np.int32)
    groups   = np.asarray(site_groups)

    if verbose:
        n_pos = int(labels.sum())
        n_neg = len(labels) - n_pos
        n_allzero = int(np.sum(np.all((features == 0) | np.isnan(features), axis=1)))
        # feature 0 is age when USE_DEMOGRAPHICS; guard against all-NaN col
        age_col = features[:, 0]
        age_mean = np.nanmean(age_col) if np.any(np.isfinite(age_col)) else float('nan')
        print(f'\nTraining set: {len(labels)} records ({n_pos} positive, {n_neg} negative)')
        print(f'Feature dimension: {features.shape[1]} (expected {_total_feature_dim()})')
        print(f'Sanity -> empty/degenerate vectors: {n_allzero} | '
              f'mean(feature[0], age): {age_mean:.2f} | sites: {len(set(groups))}')

    # ---- Balanced sample weights (helps Accuracy / F-measure under imbalance) ----
    sample_weight = compute_sample_weight('balanced', labels)

    # ---- Honest local estimate + threshold selection via out-of-fold probs ----
    threshold = 0.5
    try:
        n_sites = len(set(groups))
        n_splits = 5
        if n_sites >= n_splits:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                            random_state=42)
            split_iter = splitter.split(features, labels, groups)
            cv_kind = f'site-grouped {n_splits}-fold'
        else:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                       random_state=42)
            split_iter = splitter.split(features, labels)
            cv_kind = f'stratified {n_splits}-fold (too few sites for grouping)'

        oof = np.full(len(labels), np.nan, dtype=np.float64)
        for tr, va in split_iter:
            m = clone(_build_model())
            m.fit(features[tr], labels[tr],
                  sample_weight=compute_sample_weight('balanced', labels[tr]))
            oof[va] = m.predict_proba(features[va])[:, 1]

        mask = np.isfinite(oof)
        if mask.sum() > 0 and len(set(labels[mask])) == 2:
            oof_auc = roc_auc_score(labels[mask], oof[mask])
            # sweep threshold for best F1 on OOF predictions
            best_t, best_f = 0.5, -1.0
            for t in np.linspace(0.05, 0.95, 91):
                f = f1_score(labels[mask], (oof[mask] >= t).astype(int),
                             zero_division=0)
                if f > best_f:
                    best_f, best_t = f, t
            threshold = float(best_t)
            if verbose:
                print(f'Local OOF AUROC ({cv_kind}): {oof_auc:.3f}  '
                      f'<-- this should track the leaderboard')
                print(f'Chosen decision threshold: {threshold:.3f} '
                      f'(OOF F1 {best_f:.3f})')
    except Exception as e:
        if verbose:
            print(f'  (OOF estimation skipped: {e})')

    # ---- Fit final model on ALL data ----
    if verbose:
        print('Training the final model on all data...')

    model = _build_model()
    model.fit(features, labels, sample_weight=sample_weight)

    # ---- Save model bundle ----
    os.makedirs(model_folder, exist_ok=True)
    save_model(model_folder, model, scaler=None, threshold=threshold)

    if verbose:
        print('Done.')
        print()


def load_model(model_folder, verbose):
    """Load the trained model bundle (classifier + metadata)."""
    model_filename = os.path.join(model_folder, 'model.sav')
    bundle = joblib.load(model_filename)
    return bundle


def run_model(model, record, data_folder, verbose):
    """Run inference on a single record. Returns (binary_label, probability)."""

    clf       = model['model']
    scaler    = model.get('scaler', None)
    threshold = float(model.get('threshold', 0.5))

    feature_vec = assemble_features(record, data_folder).reshape(1, -1)
    if scaler is not None:
        feature_vec = scaler.transform(feature_vec)

    probability_output = clf.predict_proba(feature_vec)[0][1]
    binary_output = int(probability_output >= threshold)

    return binary_output, probability_output


################################################################################
#
# Shared feature assembly (used by BOTH train_model and run_model)
#
################################################################################

def assemble_features(record, data_folder, csv_path=DEFAULT_CSV_PATH):
    """
    Build the complete, fixed-length feature vector for one record.

    Order (identical at train and inference):
        [ demographics? ] + physiological(140) + algorithmic(12)

    A missing/unreadable block falls back to a NaN vector of the correct length
    (NOT zeros), so dimension stays constant AND the model can distinguish
    "missing" from a genuine zero measurement.
    """
    patient_id = record[HEADERS['bids_folder']]
    site_id    = record[HEADERS['site_id']]
    session_id = record[HEADERS['session_id']]

    blocks = []

    # ---- Demographics (available on train AND test) ----
    if USE_DEMOGRAPHICS:
        try:
            demo_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
            patient_data = load_demographics(demo_file, patient_id, session_id)
            demographic_features = extract_demographic_features(patient_data)
        except Exception as e:
            _debug(f"[demo fail] {patient_id}: {e}")
            demographic_features = np.full(NUM_DEMOGRAPHIC_FEATURES, np.nan, dtype=np.float32)
        blocks.append(np.asarray(demographic_features, dtype=np.float32))

    # ---- Physiological signal features ----
    phys_file = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER,
                             site_id, f"{patient_id}_ses-{session_id}.edf")
    if os.path.exists(phys_file):
        try:
            phys_data, phys_fs = load_signal_data(phys_file)
            phys_features = extract_physiological_features(phys_data, phys_fs, csv_path=csv_path)
            del phys_data
        except Exception as e:
            _debug(f"[phys fail] {patient_id}: {e}")
            phys_features = np.full(NUM_PHYS_FEATURES, np.nan, dtype=np.float32)
    else:
        phys_features = np.full(NUM_PHYS_FEATURES, np.nan, dtype=np.float32)
    blocks.append(np.asarray(phys_features, dtype=np.float32))

    # ---- Algorithmic (CAISR) annotation features ----
    algo_file = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
                             site_id, f"{patient_id}_ses-{session_id}_caisr_annotations.edf")
    if os.path.exists(algo_file):
        try:
            algo_data, _ = load_signal_data(algo_file)
            algo_features = extract_algorithmic_annotations_features(algo_data)
        except Exception as e:
            _debug(f"[algo fail] {patient_id}: {e}")
            algo_features = np.full(NUM_ALGO_FEATURES, np.nan, dtype=np.float32)
    else:
        algo_features = np.full(NUM_ALGO_FEATURES, np.nan, dtype=np.float32)
    blocks.append(np.asarray(algo_features, dtype=np.float32))

    feature_vec = np.hstack(blocks).astype(np.float32)
    return _sanitize(feature_vec)   # inf -> NaN, keep NaN (do NOT zero-fill)


################################################################################
#
# Feature extraction — Demographics
#
################################################################################

def extract_demographic_features(data):
    """
    Encode demographics into a length-10 vector:
        [0]    Age (continuous)
        [1:4]  Sex one-hot   (Female, Male, Other/Unknown)
        [4:9]  Race one-hot  (Asian, Black, Other, Unavailable, White)
        [9]    BMI (continuous)
    """
    age = np.array([load_age(data)], dtype=np.float32)

    sex = load_sex(data)
    sex_vec = np.zeros(3, dtype=np.float32)
    if sex == 'Female':
        sex_vec[0] = 1
    elif sex == 'Male':
        sex_vec[1] = 1
    else:
        sex_vec[2] = 1

    race_category = get_standardized_race(data).lower()
    race_vec = np.zeros(5, dtype=np.float32)
    race_mapping = {'asian': 0, 'black': 1, 'others': 2, 'unavailable': 3, 'white': 4}
    race_vec[race_mapping.get(race_category, 2)] = 1

    bmi = np.array([load_bmi(data)], dtype=np.float32)

    return np.concatenate([age, sex_vec, race_vec, bmi]).astype(np.float32)


################################################################################
#
# Feature extraction — Physiological signals (20 features x 7 groups = 140)
#
################################################################################

def extract_physiological_features(physiological_data, physiological_fs, csv_path=DEFAULT_CSV_PATH):
    """
    Standardize channels, build bipolar derivations, select one channel per
    signal group, then compute 20 (scale-free) features per group.
    """
    original_labels = list(physiological_data.keys())

    # Step 1: Standardize channel names
    rename_rules = load_rename_rules(os.path.abspath(csv_path))
    rename_map, cols_to_drop = standardize_channel_names_rename_only(original_labels, rename_rules)

    processed_channels = {}
    processed_fs = {}
    for old_label, data in physiological_data.items():
        if old_label in cols_to_drop:
            continue
        new_label = rename_map.get(old_label, old_label.lower())
        processed_channels[new_label] = data
        if old_label in physiological_fs:
            processed_fs[new_label] = physiological_fs[old_label]
        else:
            raise KeyError(f"Sampling frequency not found for channel '{old_label}'")

    # Step 2: Bipolar derivations
    bipolar_configs = [
        ('f3-m2', 'f3', ['m2']), ('f4-m1', 'f4', ['m1']),
        ('c3-m2', 'c3', ['m2']), ('c4-m1', 'c4', ['m1']),
        ('o1-m2', 'o1', ['m2']), ('o2-m1', 'o2', ['m1']),
        ('e1-m2', 'e1', ['m2']), ('e2-m1', 'e2', ['m1']),
        ('chin1-chin2', 'chin 1', ['chin 2']),
        ('lat', 'lleg+', ['lleg-']), ('rat', 'rleg+', ['rleg-'])
    ]

    for target, pos, neg_list in bipolar_configs:
        if target in processed_channels or pos not in processed_channels:
            continue
        if not all(n in processed_channels for n in neg_list):
            continue
        all_involved = [pos] + neg_list
        fs_values = [processed_fs[ch] for ch in all_involved]
        if len(set(fs_values)) > 1:
            continue
        ref_sig = processed_channels[neg_list[0]] if len(neg_list) == 1 \
            else tuple(processed_channels[n] for n in neg_list)
        derived = derive_bipolar_signal(processed_channels[pos], ref_sig)
        if derived is not None:
            processed_channels[target] = derived
            processed_fs[target] = processed_fs[pos]

    # Step 3: Select one channel per signal group
    leads_to_check = {
        'eeg':  ['c3-m2', 'c4-m1', 'f3-m2', 'f4-m1', 'o1-m2', 'o2-m1'],
        'eog':  ['e1-m2', 'e2-m1'],
        'chin': ['chin1-chin2', 'chin'],
        'leg':  ['lat', 'rat'],
        'ecg':  ['ecg', 'ekg'],
        'resp': ['airflow', 'ptaf', 'abd', 'chest'],
        'spo2': ['spo2', 'sao2'],
    }

    final_features = []
    for lead_type, candidates in leads_to_check.items():
        sig = None
        fs = None
        for candidate in candidates:
            if candidate in processed_channels and processed_channels[candidate] is not None:
                sig = processed_channels[candidate]
                fs = processed_fs.get(candidate)
                break

        if sig is not None and len(sig) > 1 and fs is not None and fs > 0:
            resampled = resample_signal(sig, fs, TARGET_FS)
            max_samples = TARGET_FS * MAX_SECONDS
            if len(resampled) > max_samples:
                resampled = resampled[:max_samples]
            final_features.extend(compute_channel_features(resampled, TARGET_FS))
        else:
            # channel genuinely absent -> NaN (missing), not zeros
            final_features.extend([np.nan] * NUM_FEATURES_PER_CHANNEL)

    del processed_channels
    return np.array(final_features, dtype=np.float32)


def compute_channel_features(sig, fs):
    """
    Compute 20 SCALE-FREE features for a single channel.

    Robust per-channel normalization (subtract median, divide by IQR) removes
    site/amplifier gain + DC offset -- the dominant confound that made absolute
    amplitude features fail to transfer across sites. Spectral features use
    RELATIVE band power (fraction of total power) for the same reason.

    Per channel (20):
        Time-domain (8):  std, MAV, RMS, ZCR, skewness, kurtosis,
                          Hjorth mobility, Hjorth complexity
        Spectral (10):    delta, theta, alpha, sigma, beta RELATIVE power;
                          delta/theta ratio, theta/alpha ratio,
                          spectral edge 50%, spectral edge 95%, spectral entropy
        Percentile (2):   5th percentile, 95th percentile (in IQR units)
    """
    sig = np.asarray(sig, dtype=np.float64)
    n = len(sig)

    # --- robust per-channel normalization (site/gain/offset invariant) ---
    med = np.median(sig)
    iqr = np.subtract(*np.percentile(sig, [75, 25]))
    if iqr > 1e-9:
        sig = (sig - med) / iqr

    features = []

    # ===== TIME DOMAIN (8) =====
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

    # ===== SPECTRAL DOMAIN (10) =====
    nperseg = min(4 * fs, n)
    if nperseg < 4:
        features.extend([0.0] * 10)
    else:
        freqs, psd = scipy_signal.welch(sig, fs=fs, nperseg=int(nperseg),
                                        noverlap=int(nperseg // 2))

        def band_power(f_low, f_high):
            mask = (freqs >= f_low) & (freqs <= f_high)
            return np.trapz(psd[mask], freqs[mask]) if np.any(mask) else 0.0

        delta_p = band_power(0.5, 4.0)
        theta_p = band_power(4.0, 8.0)
        alpha_p = band_power(8.0, 12.0)
        sigma_p = band_power(12.0, 15.0)
        beta_p  = band_power(15.0, 30.0)

        dt_ratio = delta_p / theta_p if theta_p > 1e-12 else 0.0
        ta_ratio = theta_p / alpha_p if alpha_p > 1e-12 else 0.0

        total_power = np.trapz(psd, freqs) if len(psd) > 0 else 0.0
        denom = total_power + 1e-12

        # RELATIVE band powers (fraction of total power) -> site-transferable
        rel_delta = delta_p / denom
        rel_theta = theta_p / denom
        rel_alpha = alpha_p / denom
        rel_sigma = sigma_p / denom
        rel_beta  = beta_p  / denom

        cumulative = np.cumsum(psd) * (freqs[1] - freqs[0]) if len(freqs) > 1 else np.array([0])
        cumulative_norm = cumulative / denom

        se50_idx = np.searchsorted(cumulative_norm, 0.50)
        se95_idx = np.searchsorted(cumulative_norm, 0.95)
        se50 = freqs[min(se50_idx, len(freqs) - 1)]
        se95 = freqs[min(se95_idx, len(freqs) - 1)]

        psd_norm = psd / (psd.sum() + 1e-12)
        spec_entropy = entropy(psd_norm + 1e-12)

        features.extend([rel_delta, rel_theta, rel_alpha, rel_sigma, rel_beta,
                         dt_ratio, ta_ratio, se50, se95, spec_entropy])

    # ===== PERCENTILES (2) — now in IQR units =====
    features.extend([np.percentile(sig, 5), np.percentile(sig, 95)])

    return features


def resample_signal(signal, original_fs, target_fs):
    """Resample a 1D signal using linear interpolation."""
    if abs(original_fs - target_fs) < 0.01:
        return signal.astype(np.float32)

    original_length = len(signal)
    target_length = int(original_length * target_fs / original_fs)
    if target_length <= 0:
        return np.zeros(1, dtype=np.float32)

    x_original = np.linspace(0, 1, original_length, endpoint=False)
    x_target   = np.linspace(0, 1, target_length, endpoint=False)
    return np.interp(x_target, x_original, signal).astype(np.float32)


################################################################################
#
# Feature extraction — Algorithmic (CAISR) annotations (length 12)
#
################################################################################

def extract_algorithmic_annotations_features(algo_data):
    """Sleep architecture and event-density features from CAISR outputs.

    Note: a computed value of 0 here is a REAL measurement (e.g. 0 arousals/hr),
    so we keep zeros. Only a missing/unreadable file becomes NaN (handled by the
    caller in assemble_features)."""
    if not algo_data:
        return np.full(NUM_ALGO_FEATURES, np.nan, dtype=np.float32)

    features = []

    total_hours = len(algo_data.get('resp_caisr', [])) / 3600.0

    def count_discrete_events(key):
        if key not in algo_data or total_hours <= 0:
            return 0.0
        sig = algo_data[key].astype(float)
        binary_sig = (sig > 0).astype(int)
        diff = np.diff(binary_sig, prepend=0)
        return np.count_nonzero(diff == 1) / total_hours

    features.extend([
        count_discrete_events('resp_caisr'),     # AHI
        count_discrete_events('arousal_caisr'),  # Arousal index
        count_discrete_events('limb_caisr'),     # Limb movement index
    ])

    stages = algo_data.get('stage_caisr', np.array([]))
    valid_stages = stages[stages < 9.0] if len(stages) > 0 else np.array([])

    if len(valid_stages) > 0:
        w_pct  = np.mean(valid_stages == 5)
        r_pct  = np.mean(valid_stages == 4)
        n1_pct = np.mean(valid_stages == 3)
        n2_pct = np.mean(valid_stages == 2)
        n3_pct = np.mean(valid_stages == 1)
        efficiency = np.mean((valid_stages >= 1) & (valid_stages <= 4))
    else:
        w_pct = n1_pct = n2_pct = n3_pct = r_pct = efficiency = 0.0

    features.extend([w_pct, n1_pct, n2_pct, n3_pct, r_pct, efficiency])

    prob_w     = np.mean(algo_data.get('caisr_prob_w', [0]))
    prob_n3    = np.mean(algo_data.get('caisr_prob_n3', [0]))
    prob_arous = np.mean(algo_data.get('caisr_prob_arous', [0]))
    clean = lambda x: x if x < 1.0 else 0.0
    features.extend([clean(prob_w), clean(prob_n3), clean(prob_arous)])

    return np.array(features, dtype=np.float32)


################################################################################
#
# Feature extraction — Human (expert) annotations  [TRAIN-ONLY, OPTIONAL]
#
################################################################################

def extract_human_annotations_features(human_data):
    """Expert-scored event indices and sleep architecture (length 12)."""
    if not human_data or 'resp_expert' not in human_data:
        return np.zeros(12, dtype=np.float32)

    features = []
    total_hours = len(human_data.get('resp_expert', [])) / 3600.0

    def count_discrete_events(key):
        if key not in human_data or total_hours <= 0:
            return 0.0
        sig = (human_data[key] > 0).astype(int)
        diff = np.diff(sig, prepend=0)
        return np.count_nonzero(diff == 1) / total_hours

    features.extend([
        count_discrete_events('resp_expert'),     # Human AHI
        count_discrete_events('arousal_expert'),  # Human arousal index
        count_discrete_events('limb_expert'),     # Human PLMI
    ])

    stages = human_data.get('stage_expert', np.array([]))
    valid_stages = stages[stages < 9.0] if len(stages) > 0 else np.array([])

    if len(valid_stages) > 0:
        w_pct  = np.mean(valid_stages == 5)
        r_pct  = np.mean(valid_stages == 4)
        n1_pct = np.mean(valid_stages == 3)
        n2_pct = np.mean(valid_stages == 2)
        n3_pct = np.mean(valid_stages == 1)
        efficiency = np.mean(valid_stages > 0)
    else:
        w_pct = n1_pct = n2_pct = n3_pct = r_pct = efficiency = 0.0

    features.extend([w_pct, n1_pct, n2_pct, n3_pct, r_pct, efficiency])

    if len(valid_stages) > 1 and total_hours > 0:
        transitions = np.count_nonzero(np.diff(valid_stages)) / total_hours
        waso_minutes = (np.count_nonzero(valid_stages == 0) * 30) / 60.0
        rem_indices = np.where(valid_stages == 4)[0]
        rem_latency = float(rem_indices[0]) if len(rem_indices) > 0 else 0.0
    else:
        transitions = waso_minutes = rem_latency = 0.0

    features.extend([transitions, waso_minutes, rem_latency])
    return np.array(features, dtype=np.float32)


################################################################################
#
# Save / load utilities
#
################################################################################

def save_model(model_folder, model, scaler=None, threshold=0.5):
    """Persist classifier + the config needed to reproduce features/decisions."""
    bundle = {
        'model': model,
        'scaler': scaler,                 # None for HistGB (no scaling needed)
        'threshold': float(threshold),    # tuned decision threshold
        'use_demographics': USE_DEMOGRAPHICS,
        'n_features': _total_feature_dim(),
    }
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(bundle, filename, protocol=0)
