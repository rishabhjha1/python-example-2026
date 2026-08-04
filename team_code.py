#!/usr/bin/env python
"""
George B. Moody PhysioNet Challenge 2026 -- Screening for Cognitive Impairment
During Sleep Studies.

Design notes (what changed and why)
-----------------------------------
1.  The Challenge is decided by AGE-CONDITIONED AUROC (positives compared only
    against negatives within +/- 2 years). Plain AUROC rewards the age ->
    dementia prevalence gradient, which is exactly the shortcut we must not
    take. Every offline number printed here is the conditioned one.

2.  Age is kept as a feature (it enables genuine interactions) but the training
    objective is re-weighted so that within each age bin the positive and
    negative classes carry equal total weight. That removes the marginal
    age -> label association the model would otherwise learn for free.

3.  A brain-age model is fit on NEGATIVE patients only (PSG features -> age),
    cross-fitted so training rows get out-of-fold predictions. The residual
    (predicted age - chronological age) is an explicitly age-orthogonal
    quantity and is the single most promising feature for a conditioned metric.

4.  The binary output uses the reward-optimal rule. For the Challenge reward
        TP: 1/p_a - 1     FP: -1     FN: -1     TN: 1/(1 - p_a) - 1
    predicting positive beats predicting negative exactly when
        q / p_a > (1 - q) / (1 - p_a)   <=>   q > p_a
    i.e. threshold at the AGE-SPECIFIC PREVALENCE, not at 0.5. A single global
    scale factor alpha on that curve is tuned on out-of-fold predictions
    against the actual reward metric.

5.  Threshold selection is NEVER gated behind `verbose`. In the previous
    revision the diagnostic (and therefore the threshold) only ran with -v, so
    a submission trained without it silently shipped threshold = 0.5 and
    predicted almost nothing positive.

6.  Annotation channels are read at their OWN sampling rate from the EDF
    header. CAISR arousals are 0.5 s resolution while respiratory and limb
    events are 1 s; assuming 1 Hz for everything corrupted the arousal
    inter-event intervals, the first-half / second-half split, and every
    per-stage arousal index.

7.  A dedicated SpO2 block (mean, 5th percentile, time below 90%, ODI) replaces
    the generic spectral treatment, which is meaningless on an oximetry trace.

8.  `audit_folder()` at the bottom reports per-block NaN rates. Run it on the
    supplementary set (10 records from each hidden source) BEFORE spending a
    submission. If the physiological block is mostly NaN there and populated in
    training, nothing else in this file matters.
"""

import os
import sys
import joblib
import warnings

import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import kurtosis as _kurtosis

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from tqdm import tqdm

warnings.filterwarnings('ignore')

from helper_code import *

try:
    from sklearn.model_selection import StratifiedGroupKFold as _SGKFold
except Exception:
    _SGKFold = None
try:
    from sklearn.model_selection import GroupKFold as _GKFold
except Exception:
    _GKFold = None


################################################################################
#
# Configuration
#
################################################################################

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, 'channel_table.csv')

# ---- Feature toggles -------------------------------------------------------
USE_DEMOGRAPHICS = True      # age/sex/race/BMI are present on the hidden sets
USE_BRAIN_AGE = True         # add predicted-age minus actual-age residual
AGE_BALANCED_WEIGHTS = True  # equalise class weight within each age bin
AGE_BIN_YEARS = 5            # width of the re-weighting bin

# ---- Scoring ---------------------------------------------------------------
AGE_DELTA = 2.0              # the Challenge's +/- 2 year conditioning window

# ---- Signal processing -----------------------------------------------------
TARGET_FS = 64
EPOCH_SECONDS = 30
MAX_HOURS = 12
MAX_EEG_CHANNELS = 6
MIN_VALID_EPOCHS = 10

# ---- Model -----------------------------------------------------------------
N_ENSEMBLE_FOLDS = 5
RANDOM_STATE = 42
RUN_ABLATIONS = True         # demographics-only vs signal-only conditioned AUROC


################################################################################
#
# Feature registry. Single source of truth for names and dimensions.
#
################################################################################

EPOCH_FEATURE_NAMES = [
    'rel_amp', 'rel_delta', 'rel_theta', 'rel_alpha', 'rel_sigma', 'rel_beta',
    'slowing', 'spec_ent', 'sef95', 'hj_mob', 'hj_comp', 'kurt',
]
N_EPOCH_FEATURES = len(EPOCH_FEATURE_NAMES)          # 12
IDX_SIGMA_PEAK = N_EPOCH_FEATURES                    # trailing helper column

OTHER_FEATURE_IDX = [0, 7, 8, 9, 10, 11]

EEG_STATS = ['mean', 'std', 'p10', 'p90']
OTHER_STATS = ['mean', 'std']

# SpO2 is handled by its own block, so it leaves the generic list.
OTHER_GROUPS = ['eog', 'chin', 'leg', 'ecg', 'resp']

STAGE_CODE = {'n3': 1, 'n2': 2, 'n1': 3, 'rem': 4, 'wake': 5}
SLEEP_CODES = [STAGE_CODE['n3'], STAGE_CODE['n2'], STAGE_CODE['n1'], STAGE_CODE['rem']]
NREM_CODES = [STAGE_CODE['n1'], STAGE_CODE['n2'], STAGE_CODE['n3']]
STAGE_BLOCKS = ['n3', 'n2', 'rem', 'wake']
CONTRAST_IDX = [1, 2, 4, 6]

SPINDLE_FEATURE_NAMES = [
    'sigma_peak_hz_sleep', 'sigma_peak_hz_n2', 'sigma_peak_hz_std_n2',
    'spindle_density_n2', 'spindle_density_sleep', 'spindle_mean_dur',
    'spindle_amp_ratio', 'sigma_env_cv_n2',
]

HRV_FEATURE_NAMES = [
    'rr_mean', 'sdnn', 'rmssd', 'pnn50', 'lf_rel', 'hf_rel', 'lf_hf',
    'hr_mean', 'hr_cv', 'sdnn_nrem', 'sdnn_rem', 'sdnn_rem_nrem_ratio',
    'hr_nrem', 'hr_rem',
]

SPO2_FEATURE_NAMES = [
    'spo2_mean', 'spo2_p5', 'spo2_min', 'spo2_std',
    'spo2_pct_below_90', 'spo2_pct_below_88', 'odi3', 'odi4',
]

ANNOT_FEATURE_NAMES = [
    'ahi', 'arousal_index', 'limb_index',
    'pct_wake', 'pct_n1', 'pct_n2', 'pct_n3', 'pct_rem',
    'sleep_efficiency', 'record_hours', 'sleep_hours', 'waso_min',
    'sleep_latency_min', 'rem_latency_min', 'stage_transitions_per_hr',
    'arousal_iei_mean_s', 'arousal_iei_cv',
    'arousal_index_h1', 'arousal_index_h2', 'arousal_h1_h2_ratio',
    'ahi_h1', 'ahi_h2',
    'arousal_index_n2', 'arousal_index_n3', 'arousal_index_rem',
    'caisr_prob_w', 'caisr_prob_n3', 'caisr_prob_arousal',
    'rem_fragmentation', 'n3_bout_mean_min', 'longest_sleep_bout_min',
]

DEMO_FEATURE_NAMES = [
    'age', 'sex_f', 'sex_m', 'sex_other',
    'race_asian', 'race_black', 'race_other', 'race_unavailable', 'race_white',
    'bmi',
]

BRAIN_AGE_FEATURE_NAMES = ['brain_age_pred', 'brain_age_gap']

DIM_EEG_GLOBAL = N_EPOCH_FEATURES * len(EEG_STATS)                                 # 48
DIM_OTHER_GLOBAL = len(OTHER_GROUPS) * len(OTHER_FEATURE_IDX) * len(OTHER_STATS)   # 60
DIM_EEG_STAGE = N_EPOCH_FEATURES * len(STAGE_BLOCKS)                               # 48
DIM_CONTRAST = len(CONTRAST_IDX) * 2                                               # 8
DIM_SPINDLE = len(SPINDLE_FEATURE_NAMES)                                           # 8
DIM_HRV = len(HRV_FEATURE_NAMES)                                                   # 14
DIM_SPO2 = len(SPO2_FEATURE_NAMES)                                                 # 8
DIM_ANNOT = len(ANNOT_FEATURE_NAMES)                                               # 31
DIM_DEMO = len(DEMO_FEATURE_NAMES)                                                 # 10
DIM_BRAIN_AGE = len(BRAIN_AGE_FEATURE_NAMES)                                       # 2

DIM_PHYS = (DIM_EEG_GLOBAL + DIM_OTHER_GLOBAL + DIM_EEG_STAGE
            + DIM_CONTRAST + DIM_SPINDLE + DIM_HRV + DIM_SPO2)                     # 194


def _raw_feature_dim():
    """Dimension of the vector produced by `assemble_features` (pre brain-age)."""
    dim = DIM_PHYS + DIM_ANNOT
    if USE_DEMOGRAPHICS:
        dim += DIM_DEMO
    return dim


def _total_feature_dim():
    """Dimension of the matrix the classifier actually sees."""
    dim = _raw_feature_dim()
    if USE_BRAIN_AGE:
        dim += DIM_BRAIN_AGE
    return dim


def _demo_slice():
    return slice(0, DIM_DEMO) if USE_DEMOGRAPHICS else slice(0, 0)


def _phys_slice():
    start = DIM_DEMO if USE_DEMOGRAPHICS else 0
    return slice(start, start + DIM_PHYS)


def _annot_slice():
    start = (DIM_DEMO if USE_DEMOGRAPHICS else 0) + DIM_PHYS
    return slice(start, start + DIM_ANNOT)


def _signal_cols(n_raw):
    """Columns the brain-age regressor may use: everything except demographics."""
    start = DIM_DEMO if USE_DEMOGRAPHICS else 0
    return np.arange(start, n_raw)


def raw_feature_names():
    names = []
    if USE_DEMOGRAPHICS:
        names += ['demo:' + n for n in DEMO_FEATURE_NAMES]
    for stat in EEG_STATS:
        names += [f'eeg:{n}:{stat}' for n in EPOCH_FEATURE_NAMES]
    for group in OTHER_GROUPS:
        for stat in OTHER_STATS:
            names += [f'{group}:{EPOCH_FEATURE_NAMES[i]}:{stat}' for i in OTHER_FEATURE_IDX]
    for stage in STAGE_BLOCKS:
        names += [f'eeg:{n}:{stage}' for n in EPOCH_FEATURE_NAMES]
    for i in CONTRAST_IDX:
        names.append(f'eeg:{EPOCH_FEATURE_NAMES[i]}:n3_minus_rem')
        names.append(f'eeg:{EPOCH_FEATURE_NAMES[i]}:nrem_minus_wake')
    names += ['spindle:' + n for n in SPINDLE_FEATURE_NAMES]
    names += ['hrv:' + n for n in HRV_FEATURE_NAMES]
    names += ['spo2:' + n for n in SPO2_FEATURE_NAMES]
    names += ['annot:' + n for n in ANNOT_FEATURE_NAMES]
    return names


def feature_names():
    names = raw_feature_names()
    if USE_BRAIN_AGE:
        names += ['brainage:' + n for n in BRAIN_AGE_FEATURE_NAMES]
    return names


_FALLBACK_COUNTS = {'demo': 0, 'phys': 0, 'annot': 0, 'eeg': 0, 'ecg': 0, 'spo2': 0}


def _nan(n):
    return np.full(int(n), np.nan, dtype=np.float32)


################################################################################
#
# Challenge metrics, reimplemented so we optimise the right thing offline.
#
################################################################################

def age_conditioned_auroc(labels, scores, ages, delta=AGE_DELTA):
    """
    Pr(score_pos >= score_neg) over positive/negative pairs whose ages differ by
    at most `delta`. This is the metric that decides the Challenge.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    ages = np.asarray(ages, dtype=np.float64)

    ok = np.isfinite(scores) & np.isfinite(ages)
    labels, scores, ages = labels[ok], scores[ok], ages[ok]

    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    if pos.size == 0 or neg.size == 0:
        return np.nan

    neg_ages, neg_scores = ages[neg], scores[neg]
    order = np.argsort(neg_ages, kind='mergesort')
    neg_ages, neg_scores = neg_ages[order], neg_scores[order]

    num = 0.0
    den = 0.0
    lo_all = np.searchsorted(neg_ages, ages[pos] - delta, side='left')
    hi_all = np.searchsorted(neg_ages, ages[pos] + delta, side='right')
    for k, i in enumerate(pos):
        lo, hi = lo_all[k], hi_all[k]
        if hi <= lo:
            continue
        window = neg_scores[lo:hi]
        num += np.count_nonzero(window < scores[i]) + 0.5 * np.count_nonzero(window == scores[i])
        den += window.size
    return float(num / den) if den > 0 else np.nan


def age_prevalence_curve(ages, labels, delta=AGE_DELTA, grid=None):
    """
    p_a = Pr(positive | age within delta of a), estimated on the training set,
    matching how the Challenge scorer builds its prevalence table.
    """
    ages = np.asarray(ages, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    ok = np.isfinite(ages)
    a_ok, y_ok = ages[ok], labels[ok]
    global_p = float(np.mean(labels)) if labels.size else 0.1

    if grid is None:
        grid = np.arange(0.0, 121.0, 1.0)
    values = np.full(grid.shape, np.nan, dtype=np.float64)
    if a_ok.size:
        order = np.argsort(a_ok, kind='mergesort')
        a_sorted, y_sorted = a_ok[order], y_ok[order]
        csum = np.concatenate([[0.0], np.cumsum(y_sorted)])
        lo = np.searchsorted(a_sorted, grid - delta, side='left')
        hi = np.searchsorted(a_sorted, grid + delta, side='right')
        n = hi - lo
        with np.errstate(invalid='ignore', divide='ignore'):
            values = np.where(n >= 15, (csum[hi] - csum[lo]) / np.maximum(n, 1), np.nan)

    # Fill sparse ages by nearest available estimate, then by global prevalence.
    finite = np.isfinite(values)
    if finite.any():
        values = np.interp(grid, grid[finite], values[finite])
    else:
        values = np.full(grid.shape, global_p)
    return grid.astype(np.float64), np.clip(values, 1e-3, 1 - 1e-3), global_p


def prevalence_at_age(age, grid, values, global_p):
    if age is None or not np.isfinite(age):
        return float(global_p)
    return float(np.interp(float(age), grid, values))


def decision_threshold(p_a, alpha=1.0, balanced=AGE_BALANCED_WEIGHTS):
    """
    Threshold on the model's probability output.

    Unweighted training gives a posterior calibrated to the cohort prior, and
    the reward-optimal rule is then simply q > p_a.

    Age-balanced weighting makes the effective prior 0.5 WITHIN each age bin.
    Converting that balanced posterior q_b back to the p_a prior multiplies the
    odds by p_a / (1 - p_a), so the rule q_cal > p_a collapses exactly to
    q_b > 0.5. Failing to account for this is why a naive p_a threshold on a
    balanced model fires on nearly every patient.

    `alpha` is a single multiplicative correction in ODDS space, fitted
    out-of-fold against the reward itself.
    """
    base = 0.5 if balanced else float(np.clip(p_a, 1e-3, 1 - 1e-3))
    odds = (base / (1.0 - base)) * float(alpha)
    return float(np.clip(odds / (1.0 + odds), 0.005, 0.995))


def prevalence_reward(labels, binary, ages, grid, values, global_p):
    """The Challenge's prevalence-based reward, averaged over the cohort."""
    labels = np.asarray(labels).astype(int)
    binary = np.asarray(binary).astype(int)
    p = np.array([prevalence_at_age(a, grid, values, global_p) for a in np.asarray(ages)])
    p = np.clip(p, 1e-3, 1 - 1e-3)
    r = np.where((labels == 1) & (binary == 1), 1.0 / p - 1.0,
        np.where((labels == 0) & (binary == 0), 1.0 / (1.0 - p) - 1.0, -1.0))
    return float(np.mean(r))


def age_balanced_weights(ages, labels, bin_years=AGE_BIN_YEARS):
    """
    Equalise positive and negative total weight inside each age bin, so the
    model gains nothing from the marginal age -> label relationship and has to
    find within-age structure instead.
    """
    ages = np.asarray(ages, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    w = np.ones(len(labels), dtype=np.float64)
    if not AGE_BALANCED_WEIGHTS:
        return w

    bins = np.where(np.isfinite(ages), np.floor(ages / bin_years), -999)
    for b in np.unique(bins):
        m = bins == b
        for cls in (0, 1):
            mc = m & (labels == cls)
            n = mc.sum()
            if n > 0:
                w[mc] = 1.0 / n
        # A bin with only one class carries no within-age information at all.
        if len(np.unique(labels[m])) < 2:
            w[m] *= 0.25
    w *= len(w) / w.sum()
    return w


################################################################################
#
# Required functions. Do NOT change the arguments of these functions.
#
################################################################################

def train_model(data_folder, model_folder, verbose, csv_path=DEFAULT_CSV_PATH):
    _verify_channel_table(csv_path)

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
    features, labels, sites, ages = [], [], [], []

    pbar = tqdm(range(num_records), desc='Extracting features', unit='rec',
                disable=not verbose)
    for i in pbar:
        patient_id = None
        try:
            record = patient_metadata_list[i]
            patient_id = record[HEADERS['bids_folder']]
            site_id = record[HEADERS['site_id']]
            session_id = record[HEADERS['session_id']]

            if verbose:
                pbar.set_postfix({'patient': patient_id})

            label = load_diagnoses(diagnosis_file, patient_id)
            if label not in (0, 1):
                continue

            phys_file = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER,
                                     site_id, f'{patient_id}_ses-{session_id}.edf')
            if not os.path.exists(phys_file):
                if verbose:
                    tqdm.write(f'  ! Missing physiological data for {patient_id}. Skipping.')
                continue

            feature_vec, age = _assemble(record, data_folder, csv_path=csv_path)

            features.append(feature_vec)
            labels.append(int(label))
            sites.append(str(site_id))
            ages.append(age)

        except Exception as e:
            tqdm.write(f'  !!! Error processing record {i + 1} ({patient_id}): {e}')
            continue
    pbar.close()

    if len(labels) == 0:
        raise ValueError('No valid labeled records found for training.')

    X_raw = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)
    groups = np.asarray(sites)
    ages = np.asarray(ages, dtype=np.float64)

    X_raw[~np.isfinite(X_raw) & ~np.isnan(X_raw)] = np.nan

    if X_raw.shape[1] != _raw_feature_dim():
        raise ValueError(f'Raw feature dimension {X_raw.shape[1]} != expected '
                         f'{_raw_feature_dim()}. Aborting before train/inference diverge.')

    if verbose:
        n_pos = int(y.sum())
        print(f'\nTraining set: {len(y)} records ({n_pos} positive, {len(y) - n_pos} negative)')
        print(f'Raw feature dimension: {X_raw.shape[1]}')
        print(f'Age available for {np.isfinite(ages).mean():.3f} of records; '
              f'median {np.nanmedian(ages):.1f}')
        _print_block_nan_table(X_raw)
        print('Fallback counts (blocks that failed and were set to NaN): '
              f'{_FALLBACK_COUNTS}')
        _print_site_table(groups, y, ages)

    # ---- Age-conditioned prevalence, used for the binary decision rule -----
    grid, prev_values, global_p = age_prevalence_curve(ages, y)

    # ---- Honest offline estimate. ALWAYS runs; verbose only controls printing.
    alpha, oof_pred = _run_diagnostic(X_raw, y, ages, groups, grid, prev_values,
                                      global_p, verbose)

    # ---- Brain-age model on all negatives, for use at inference ------------
    brain_age_model = None
    if USE_BRAIN_AGE:
        brain_age_model = _fit_brain_age(X_raw, y, ages)
    X = _augment(X_raw, ages, brain_age_model)

    if X.shape[1] != _total_feature_dim():
        raise ValueError(f'Feature dimension {X.shape[1]} != expected {_total_feature_dim()}.')

    # ---- Final fold ensemble on all data ----------------------------------
    if verbose:
        print(f'\nTraining {N_ENSEMBLE_FOLDS}-fold ensemble on all data...')

    weights = age_balanced_weights(ages, y)
    models = []
    n_splits = min(N_ENSEMBLE_FOLDS, int(np.bincount(y).min()))
    if n_splits < 2:
        models.append(_build_model(RANDOM_STATE).fit(X, y, sample_weight=weights))
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        for k, (tr, _) in enumerate(skf.split(X, y)):
            m = _build_model(RANDOM_STATE + k).fit(X[tr], y[tr], sample_weight=weights[tr])
            models.append(m)

    os.makedirs(model_folder, exist_ok=True)
    save_model(model_folder, models, brain_age_model, alpha, grid, prev_values, global_p)

    if verbose:
        print('Done.')
        print()


def load_model(model_folder, verbose):
    _verify_channel_table(DEFAULT_CSV_PATH)
    bundle = joblib.load(os.path.join(model_folder, 'model.sav'))
    if bundle.get('n_features') != _total_feature_dim():
        raise ValueError(
            f"Model was trained with {bundle.get('n_features')} features but this "
            f"code produces {_total_feature_dim()}. Refusing to run.")
    return bundle


def run_model(model, record, data_folder, verbose):
    """
    Inference on one record.

    The binary output thresholds at alpha * p_a(age): the reward-optimal rule is
    "positive when the posterior exceeds the age-specific prevalence", and alpha
    is a single scale factor fitted out-of-fold against the reward itself.
    """
    models = model['models']
    brain_age_model = model.get('brain_age_model', None)
    alpha = float(model.get('alpha', 1.0))
    grid = np.asarray(model['prev_grid'], dtype=np.float64)
    prev_values = np.asarray(model['prev_values'], dtype=np.float64)
    global_p = float(model.get('global_prevalence', 0.1))

    x_raw, age = _assemble(record, data_folder, csv_path=DEFAULT_CSV_PATH)
    x_raw = x_raw.reshape(1, -1)
    x_raw[~np.isfinite(x_raw) & ~np.isnan(x_raw)] = np.nan

    x = _augment(x_raw, np.array([age], dtype=np.float64), brain_age_model)

    probs = [float(m.predict_proba(x)[0][1]) for m in models]
    probability_output = float(np.mean(probs))

    p_a = prevalence_at_age(age, grid, prev_values, global_p)
    balanced = bool(model.get('age_balanced_weights', AGE_BALANCED_WEIGHTS))
    threshold = decision_threshold(p_a, alpha, balanced)
    binary_output = int(probability_output >= threshold)

    return binary_output, probability_output


################################################################################
#
# Brain age: a deliberately age-orthogonal feature
#
################################################################################

def _fit_brain_age(X_raw, y, ages, seed=RANDOM_STATE):
    """
    Regress chronological age on PSG features using NEGATIVE patients only, so
    the model learns normal ageing rather than disease. The residual on any
    patient then answers "does this brain look older than it should?".
    """
    cols = _signal_cols(X_raw.shape[1])
    m = (np.asarray(y) == 0) & np.isfinite(ages)
    if m.sum() < 50:
        return None
    reg = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=20, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=25,
        random_state=seed,
    )
    try:
        reg.fit(X_raw[np.ix_(np.flatnonzero(m), cols)], ages[m])
    except Exception:
        return None
    return {'model': reg, 'cols': cols}


def _augment(X_raw, ages, brain_age_model):
    """Append [predicted_age, predicted_age - actual_age]."""
    if not USE_BRAIN_AGE:
        return np.asarray(X_raw, dtype=np.float32)
    n = X_raw.shape[0]
    extra = np.full((n, DIM_BRAIN_AGE), np.nan, dtype=np.float32)
    if brain_age_model is not None:
        try:
            pred = brain_age_model['model'].predict(X_raw[:, brain_age_model['cols']])
            extra[:, 0] = pred
            extra[:, 1] = pred - np.asarray(ages, dtype=np.float64)
        except Exception:
            pass
    out = np.hstack([np.asarray(X_raw, dtype=np.float32), extra]).astype(np.float32)
    out[~np.isfinite(out) & ~np.isnan(out)] = np.nan
    return out


################################################################################
#
# Shared feature assembly (used by BOTH train_model and run_model)
#
################################################################################

def assemble_features(record, data_folder, csv_path=DEFAULT_CSV_PATH):
    """Backwards-compatible wrapper returning only the vector."""
    return _assemble(record, data_folder, csv_path=csv_path)[0]


def _assemble(record, data_folder, csv_path=DEFAULT_CSV_PATH):
    """
    Fixed-length raw feature vector plus the patient's age.

    Order (identical at train and inference):
        [ demographics(10) ] + physiological(194) + annotations(31)
    """
    patient_id = record[HEADERS['bids_folder']]
    site_id = record[HEADERS['site_id']]
    session_id = record[HEADERS['session_id']]

    blocks = []
    age = np.nan

    # ---- Demographics ------------------------------------------------------
    demo_vec = _nan(DIM_DEMO)
    try:
        demo_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
        patient_data = load_demographics(demo_file, patient_id, session_id)
        demo_vec = extract_demographic_features(patient_data)
    except Exception:
        _FALLBACK_COUNTS['demo'] += 1
    age = float(demo_vec[0])
    if USE_DEMOGRAPHICS:
        blocks.append(demo_vec)

    # ---- Algorithmic (CAISR) annotations ----------------------------------
    algo_file = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
                             site_id, f'{patient_id}_ses-{session_id}_caisr_annotations.edf')
    algo_data, algo_fs = None, {}
    if os.path.exists(algo_file):
        try:
            algo_data, algo_fs = load_signal_data(algo_file)
        except Exception:
            algo_data, algo_fs = None, {}
    if algo_data is None:
        _FALLBACK_COUNTS['annot'] += 1
        annot_features = _nan(DIM_ANNOT)
        stage_sig, stage_fs = None, 1.0
    else:
        annot_features = extract_annotation_features(algo_data, algo_fs)
        stage_sig, stage_fs = _stage_series(algo_data, algo_fs)

    # ---- Physiological signals --------------------------------------------
    phys_file = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER,
                             site_id, f'{patient_id}_ses-{session_id}.edf')
    phys_features = _nan(DIM_PHYS)
    if os.path.exists(phys_file):
        try:
            phys_data, phys_fs = load_signal_data(phys_file)
            phys_features = extract_physiological_features(
                phys_data, phys_fs, stage_sig, stage_fs, csv_path=csv_path)
            del phys_data
        except Exception:
            _FALLBACK_COUNTS['phys'] += 1
            phys_features = _nan(DIM_PHYS)
    else:
        _FALLBACK_COUNTS['phys'] += 1

    blocks.append(np.asarray(phys_features, dtype=np.float32))
    blocks.append(np.asarray(annot_features, dtype=np.float32))

    vec = np.hstack(blocks).astype(np.float32)
    vec[~np.isfinite(vec) & ~np.isnan(vec)] = np.nan
    return vec, age


################################################################################
#
# Demographics
#
################################################################################

def extract_demographic_features(data):
    """age, sex one-hot(3), race one-hot(5), bmi."""
    try:
        age = float(load_age(data))
    except Exception:
        age = np.nan
    try:
        bmi = float(load_bmi(data))
    except Exception:
        bmi = np.nan

    if not np.isfinite(age) or not (0 < age < 120):
        age = np.nan
    if not np.isfinite(bmi) or not (8 < bmi < 100):
        bmi = np.nan

    sex_vec = np.zeros(3, dtype=np.float32)
    try:
        sex = load_sex(data)
    except Exception:
        sex = None
    if sex == 'Female':
        sex_vec[0] = 1
    elif sex == 'Male':
        sex_vec[1] = 1
    else:
        sex_vec[2] = 1

    race_vec = np.zeros(5, dtype=np.float32)
    try:
        race_category = get_standardized_race(data).lower()
    except Exception:
        race_category = 'unavailable'
    race_mapping = {'asian': 0, 'black': 1, 'others': 2, 'unavailable': 3, 'white': 4}
    race_vec[race_mapping.get(race_category, 2)] = 1

    return np.concatenate([[age], sex_vec, race_vec, [bmi]]).astype(np.float32)


################################################################################
#
# Physiological signals
#
################################################################################

def extract_physiological_features(physiological_data, physiological_fs,
                                   stage_sig, stage_fs, csv_path=DEFAULT_CSV_PATH):
    """
    Length-DIM_PHYS vector:
        EEG global (48) + other-group global (60) + EEG per-stage (48)
        + stage contrasts (8) + spindle/sigma (8) + HRV (14) + SpO2 (8)
    """
    channels, fs_map = _standardize_and_derive(physiological_data, physiological_fs, csv_path)

    eeg_candidates = ['c3-m2', 'c4-m1', 'f3-m2', 'f4-m1', 'o1-m2', 'o2-m1']
    other_candidates = {
        'eog':  ['e1-m2', 'e2-m1'],
        'chin': ['chin1-chin2', 'chin'],
        'leg':  ['lat', 'rat'],
        'ecg':  ['ecg', 'ekg'],
        'resp': ['airflow', 'ptaf', 'abd', 'chest'],
    }
    spo2_candidates = ['spo2', 'sao2']

    max_epochs = int(MAX_HOURS * 3600 // EPOCH_SECONDS)

    # ---- EEG ---------------------------------------------------------------
    eeg_mats = []
    for name in eeg_candidates[:MAX_EEG_CHANNELS]:
        sig = channels.get(name)
        fs = fs_map.get(name)
        if sig is None or fs is None or fs <= 0 or len(sig) < 2:
            continue
        mat = _epoch_feature_matrix(_resample(sig, fs, TARGET_FS), TARGET_FS, max_epochs)
        if mat is not None:
            eeg_mats.append(mat)

    if eeg_mats:
        n_ep = min(m.shape[0] for m in eeg_mats)
        eeg_mat = np.nanmean(np.stack([m[:n_ep] for m in eeg_mats], axis=0), axis=0)
    else:
        _FALLBACK_COUNTS['eeg'] += 1
        eeg_mat = None

    n_epochs = eeg_mat.shape[0] if eeg_mat is not None else 0
    stage_per_epoch = _stage_per_epoch(stage_sig, stage_fs, n_epochs)

    if eeg_mat is not None:
        eeg_global = _aggregate(eeg_mat[:, :N_EPOCH_FEATURES], EEG_STATS)
    else:
        eeg_global = _nan(DIM_EEG_GLOBAL)

    # ---- Other groups ------------------------------------------------------
    other_global = []
    ecg_raw, ecg_fs = None, None
    for group in OTHER_GROUPS:
        sig, fs = None, None
        for name in other_candidates[group]:
            if channels.get(name) is not None and fs_map.get(name):
                sig, fs = channels[name], fs_map[name]
                break
        if group == 'ecg' and sig is not None:
            ecg_raw, ecg_fs = sig, fs
        if sig is None or fs is None or fs <= 0 or len(sig) < 2:
            other_global.append(_nan(len(OTHER_FEATURE_IDX) * len(OTHER_STATS)))
            continue
        mat = _epoch_feature_matrix(_resample(sig, fs, TARGET_FS), TARGET_FS, max_epochs)
        if mat is None:
            other_global.append(_nan(len(OTHER_FEATURE_IDX) * len(OTHER_STATS)))
        else:
            other_global.append(_aggregate(mat[:, OTHER_FEATURE_IDX], OTHER_STATS))
    other_global = np.hstack(other_global)

    # ---- Per-stage means and contrasts ------------------------------------
    if eeg_mat is not None and stage_per_epoch is not None:
        stage_means, contrasts = _stage_blocks(eeg_mat[:, :N_EPOCH_FEATURES], stage_per_epoch)
    else:
        stage_means, contrasts = _nan(DIM_EEG_STAGE), _nan(DIM_CONTRAST)

    # ---- Spindles ----------------------------------------------------------
    primary_eeg, primary_fs = None, None
    for name in eeg_candidates:
        if channels.get(name) is not None and fs_map.get(name):
            primary_eeg, primary_fs = channels[name], fs_map[name]
            break
    if primary_eeg is not None and eeg_mat is not None:
        spindle = _spindle_features(_resample(primary_eeg, primary_fs, TARGET_FS),
                                    TARGET_FS, eeg_mat, stage_per_epoch)
    else:
        spindle = _nan(DIM_SPINDLE)

    # ---- HRV ---------------------------------------------------------------
    if ecg_raw is not None and ecg_fs:
        hrv = _hrv_features(np.asarray(ecg_raw, dtype=np.float64), float(ecg_fs),
                            stage_per_epoch)
    else:
        _FALLBACK_COUNTS['ecg'] += 1
        hrv = _nan(DIM_HRV)

    # ---- SpO2 --------------------------------------------------------------
    spo2_sig, spo2_fs = None, None
    for name in spo2_candidates:
        if channels.get(name) is not None and fs_map.get(name):
            spo2_sig, spo2_fs = channels[name], fs_map[name]
            break
    if spo2_sig is not None:
        spo2 = _spo2_features(np.asarray(spo2_sig, dtype=np.float64), float(spo2_fs),
                              stage_per_epoch)
    else:
        _FALLBACK_COUNTS['spo2'] += 1
        spo2 = _nan(DIM_SPO2)

    del channels
    out = np.hstack([eeg_global, other_global, stage_means, contrasts, spindle, hrv, spo2])
    return out.astype(np.float32)


def _standardize_and_derive(physiological_data, physiological_fs, csv_path):
    rename_rules = load_rename_rules(os.path.abspath(csv_path))
    rename_map, cols_to_drop = standardize_channel_names_rename_only(
        list(physiological_data.keys()), rename_rules)

    channels, fs_map = {}, {}
    for old_label, data in physiological_data.items():
        if old_label in cols_to_drop:
            continue
        new_label = rename_map.get(old_label, old_label.lower())
        channels[new_label] = data
        if old_label in physiological_fs:
            fs_map[new_label] = physiological_fs[old_label]

    bipolar_configs = [
        ('f3-m2', 'f3', ['m2']), ('f4-m1', 'f4', ['m1']),
        ('c3-m2', 'c3', ['m2']), ('c4-m1', 'c4', ['m1']),
        ('o1-m2', 'o1', ['m2']), ('o2-m1', 'o2', ['m1']),
        ('e1-m2', 'e1', ['m2']), ('e2-m1', 'e2', ['m1']),
        ('chin1-chin2', 'chin 1', ['chin 2']),
        ('lat', 'lleg+', ['lleg-']), ('rat', 'rleg+', ['rleg-']),
    ]
    for target, pos, neg_list in bipolar_configs:
        if target in channels or pos not in channels:
            continue
        if not all(n in channels for n in neg_list):
            continue
        fs_values = [fs_map.get(ch) for ch in [pos] + neg_list]
        if None in fs_values or len(set(fs_values)) > 1:
            continue
        ref = channels[neg_list[0]] if len(neg_list) == 1 else tuple(
            channels[n] for n in neg_list)
        derived = derive_bipolar_signal(channels[pos], ref)
        if derived is not None:
            channels[target] = derived
            fs_map[target] = fs_map[pos]

    # Cross-mastoid fallback: some sites ship only one mastoid, or an already
    # bipolar montage under a different name. Accept a contralateral partner
    # rather than losing the whole EEG block.
    fallbacks = [('c3-m2', 'c3', 'm1'), ('c4-m1', 'c4', 'm2'),
                 ('f3-m2', 'f3', 'm1'), ('f4-m1', 'f4', 'm2'),
                 ('o1-m2', 'o1', 'm1'), ('o2-m1', 'o2', 'm2')]
    for target, pos, alt_ref in fallbacks:
        if target in channels or pos not in channels or alt_ref not in channels:
            continue
        if fs_map.get(pos) != fs_map.get(alt_ref):
            continue
        derived = derive_bipolar_signal(channels[pos], channels[alt_ref])
        if derived is not None:
            channels[target] = derived
            fs_map[target] = fs_map[pos]

    return channels, fs_map


def _resample(sig, original_fs, target_fs):
    sig = np.asarray(sig, dtype=np.float64).ravel()
    if abs(original_fs - target_fs) < 0.01:
        return sig
    n = len(sig)
    m = int(n * target_fs / float(original_fs))
    if m <= 1:
        return np.zeros(1, dtype=np.float64)
    x_old = np.linspace(0.0, 1.0, n, endpoint=False)
    x_new = np.linspace(0.0, 1.0, m, endpoint=False)
    return np.interp(x_new, x_old, sig)


def _epoch_feature_matrix(sig, fs, max_epochs):
    """
    Vectorised per-epoch features. Returns (n_epochs, N_EPOCH_FEATURES + 1);
    the trailing column is the sigma peak frequency used by the spindle block.
    """
    fs = int(round(fs))
    spe = EPOCH_SECONDS * fs
    n_ep = int(len(sig) // spe)
    if n_ep < 1:
        return None
    n_ep = min(n_ep, max_epochs)
    X = np.asarray(sig[:n_ep * spe], dtype=np.float64).reshape(n_ep, spe)

    out = np.full((n_ep, N_EPOCH_FEATURES + 1), np.nan, dtype=np.float64)
    eps = 1e-12

    q75, q25 = np.percentile(X, 75, axis=1), np.percentile(X, 25, axis=1)
    epoch_iqr = q75 - q25
    night_iqr = float(np.median(epoch_iqr[np.isfinite(epoch_iqr)])) if np.any(
        np.isfinite(epoch_iqr)) else 0.0
    if night_iqr <= eps:
        return None
    out[:, 0] = np.log((epoch_iqr + eps) / (night_iqr + eps))

    nperseg = int(min(4 * fs, spe))
    if nperseg < 8:
        return None
    freqs, psd = scipy_signal.welch(X, fs=fs, nperseg=nperseg,
                                    noverlap=nperseg // 2, axis=-1)

    tmask = (freqs >= 0.5) & (freqs <= 30.0)
    if not np.any(tmask):
        return None

    def band(lo, hi):
        m = (freqs >= lo) & (freqs <= hi)
        if not np.any(m):
            return np.zeros(n_ep)
        return np.trapezoid(psd[:, m], freqs[m], axis=-1) if hasattr(np, 'trapezoid') \
            else np.trapz(psd[:, m], freqs[m], axis=-1)

    total = band(0.5, 30.0)
    delta, theta = band(0.5, 4.0), band(4.0, 8.0)
    alpha, sigma_b, beta = band(8.0, 12.0), band(12.0, 15.0), band(15.0, 30.0)

    out[:, 1] = delta / (total + eps)
    out[:, 2] = theta / (total + eps)
    out[:, 3] = alpha / (total + eps)
    out[:, 4] = sigma_b / (total + eps)
    out[:, 5] = beta / (total + eps)
    out[:, 6] = np.log((delta + theta + eps) / (alpha + beta + eps))

    p = psd[:, tmask]
    p = p / (p.sum(axis=1, keepdims=True) + eps)
    out[:, 7] = -(p * np.log(p + eps)).sum(axis=1)

    cum = np.cumsum(p, axis=1)
    idx95 = np.argmax(cum >= 0.95, axis=1)
    out[:, 8] = freqs[tmask][np.clip(idx95, 0, tmask.sum() - 1)]

    var0 = np.var(X, axis=1)
    d1 = np.diff(X, axis=1)
    var1 = np.var(d1, axis=1)
    d2 = np.diff(d1, axis=1)
    var2 = np.var(d2, axis=1)
    mob = np.sqrt(np.where(var0 > eps, var1 / (var0 + eps), np.nan))
    out[:, 9] = mob
    out[:, 10] = np.sqrt(np.where(var1 > eps, var2 / (var1 + eps), np.nan)) / (mob + eps)

    out[:, 11] = _kurtosis(X, axis=1)

    smask = (freqs >= 12.0) & (freqs <= 15.0)
    if np.any(smask):
        out[:, IDX_SIGMA_PEAK] = freqs[smask][np.argmax(psd[:, smask], axis=1)]

    dead = (var0 <= eps) | (epoch_iqr <= eps)
    out[dead, :] = np.nan

    if np.isfinite(out[:, :N_EPOCH_FEATURES]).any(axis=1).sum() < MIN_VALID_EPOCHS:
        return None
    return out


def _aggregate(mat, stats):
    parts = []
    for stat in stats:
        if stat == 'mean':
            parts.append(np.nanmean(mat, axis=0))
        elif stat == 'std':
            parts.append(np.nanstd(mat, axis=0))
        elif stat == 'p10':
            parts.append(np.nanpercentile(mat, 10, axis=0))
        elif stat == 'p90':
            parts.append(np.nanpercentile(mat, 90, axis=0))
    return np.hstack(parts).astype(np.float32)


def _stage_blocks(mat, stage_per_epoch):
    n = min(mat.shape[0], len(stage_per_epoch))
    mat, stage = mat[:n], stage_per_epoch[:n]

    means = {}
    for name in STAGE_BLOCKS:
        m = stage == STAGE_CODE[name]
        means[name] = np.nanmean(mat[m], axis=0) if m.sum() >= 3 else _nan(N_EPOCH_FEATURES)

    nrem_mask = np.isin(stage, NREM_CODES)
    nrem = np.nanmean(mat[nrem_mask], axis=0) if nrem_mask.sum() >= 3 else _nan(N_EPOCH_FEATURES)

    stage_means = np.hstack([means[name] for name in STAGE_BLOCKS]).astype(np.float32)

    contrasts = []
    for i in CONTRAST_IDX:
        contrasts.append(means['n3'][i] - means['rem'][i])
        contrasts.append(nrem[i] - means['wake'][i])
    return stage_means, np.asarray(contrasts, dtype=np.float32)


def _spindle_features(sig, fs, eeg_mat, stage_per_epoch):
    """Sigma peak frequency and spindle density; both drop with cortical ageing."""
    out = _nan(DIM_SPINDLE).astype(np.float64)
    eps = 1e-12
    fs = int(round(fs))

    sigma_peak = eeg_mat[:, IDX_SIGMA_PEAK]
    if stage_per_epoch is not None:
        n = min(len(sigma_peak), len(stage_per_epoch))
        stage = stage_per_epoch[:n]
        sp = sigma_peak[:n]
        sleep_mask = np.isin(stage, SLEEP_CODES)
        n2_mask = stage == STAGE_CODE['n2']
    else:
        n = len(sigma_peak)
        sp = sigma_peak
        sleep_mask = np.ones(n, dtype=bool)
        n2_mask = np.zeros(n, dtype=bool)

    if sleep_mask.sum() >= 3:
        out[0] = np.nanmean(sp[sleep_mask])
    if n2_mask.sum() >= 3:
        out[1] = np.nanmean(sp[n2_mask])
        out[2] = np.nanstd(sp[n2_mask])

    try:
        nyq = fs / 2.0
        b, a = scipy_signal.butter(4, [12.0 / nyq, 15.0 / nyq], btype='band')
        filt = scipy_signal.filtfilt(b, a, sig)
        env = np.abs(scipy_signal.hilbert(filt))
        w = max(1, int(0.2 * fs))
        env = np.convolve(env, np.ones(w) / w, mode='same')
    except Exception:
        return out.astype(np.float32)

    spe = EPOCH_SECONDS * fs
    n_ep_sig = int(len(env) // spe)
    n_use = min(n_ep_sig, n)
    if n_use < 1:
        return out.astype(np.float32)

    sample_sleep = np.repeat(sleep_mask[:n_use], spe)
    sample_n2 = np.repeat(n2_mask[:n_use], spe)

    env_use = env[:n_use * spe]
    ref = env_use[sample_sleep] if sample_sleep.any() else env_use
    if ref.size < fs:
        return out.astype(np.float32)
    baseline = float(np.median(ref))
    if baseline <= eps:
        return out.astype(np.float32)
    thr = 1.5 * baseline

    above = (env_use > thr).astype(np.int8)
    edges = np.diff(above, prepend=0, append=0)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    durs = (ends - starts) / float(fs)
    keep = (durs >= 0.4) & (durs <= 2.0)
    starts, ends, durs = starts[keep], ends[keep], durs[keep]

    sleep_min = sample_sleep.sum() / float(fs) / 60.0
    n2_min = sample_n2.sum() / float(fs) / 60.0

    if len(starts):
        in_sleep = sample_sleep[starts]
        in_n2 = sample_n2[starts]
        if n2_min > 0:
            out[3] = float(in_n2.sum()) / n2_min
        if sleep_min > 0:
            out[4] = float(in_sleep.sum()) / sleep_min
        out[5] = float(np.mean(durs))
        peaks = np.array([env_use[s:e].max() for s, e in zip(starts, ends)])
        out[6] = float(np.mean(peaks) / (baseline + eps))
    else:
        if n2_min > 0:
            out[3] = 0.0
        if sleep_min > 0:
            out[4] = 0.0

    if sample_n2.any():
        e_n2 = env_use[sample_n2]
        m = float(np.mean(e_n2))
        if m > eps:
            out[7] = float(np.std(e_n2) / m)

    return out.astype(np.float32)


def _hrv_features(ecg, fs, stage_per_epoch):
    """HRV from detected R peaks, resolved by sleep stage."""
    out = _nan(DIM_HRV).astype(np.float64)
    eps = 1e-12
    if ecg is None or len(ecg) < fs * 60:
        return out.astype(np.float32)

    try:
        nyq = fs / 2.0
        hi = min(25.0, nyq * 0.9)
        lo = 5.0
        if hi <= lo:
            return out.astype(np.float32)
        b, a = scipy_signal.butter(3, [lo / nyq, hi / nyq], btype='band')
        x = scipy_signal.filtfilt(b, a, ecg)
        x = np.abs(np.diff(x, prepend=x[0]))
        w = max(1, int(0.08 * fs))
        x = np.convolve(x, np.ones(w) / w, mode='same')
    except Exception:
        return out.astype(np.float32)

    height = float(np.percentile(x, 98)) * 0.4
    if not np.isfinite(height) or height <= eps:
        return out.astype(np.float32)
    peaks, _ = scipy_signal.find_peaks(x, distance=max(1, int(0.3 * fs)), height=height)
    if len(peaks) < 60:
        return out.astype(np.float32)

    t_peaks = peaks / float(fs)
    rr = np.diff(t_peaks)
    ok = (rr >= 0.3) & (rr <= 2.0)
    rr_clean = rr[ok]
    if len(rr_clean) < 50:
        return out.astype(np.float32)

    out[0] = np.mean(rr_clean)
    out[1] = np.std(rr_clean)
    drr = np.diff(rr_clean)
    out[2] = np.sqrt(np.mean(drr ** 2)) if len(drr) else np.nan
    out[3] = np.mean(np.abs(drr) > 0.05) if len(drr) else np.nan
    out[7] = 60.0 / (out[0] + eps)
    out[8] = out[1] / (out[0] + eps)

    try:
        t_rr = t_peaks[1:][ok]
        grid = np.arange(t_rr[0], t_rr[-1], 0.25)
        if len(grid) > 64:
            tach = np.interp(grid, t_rr, rr_clean)
            tach = tach - np.mean(tach)
            f, p = scipy_signal.welch(tach, fs=4.0, nperseg=min(256, len(tach)))
            lf = p[(f >= 0.04) & (f < 0.15)].sum()
            hf = p[(f >= 0.15) & (f < 0.40)].sum()
            tot = lf + hf
            if tot > eps:
                out[4] = lf / tot
                out[5] = hf / tot
                out[6] = lf / (hf + eps)
    except Exception:
        pass

    if stage_per_epoch is not None and len(stage_per_epoch):
        beat_epoch = (t_peaks[1:][ok] // EPOCH_SECONDS).astype(int)
        valid = beat_epoch < len(stage_per_epoch)
        beat_epoch, rr_s = beat_epoch[valid], rr_clean[valid]
        if len(rr_s) > 30:
            st = stage_per_epoch[beat_epoch]
            nrem = np.isin(st, NREM_CODES)
            rem = st == STAGE_CODE['rem']
            if nrem.sum() > 30:
                out[9] = np.std(rr_s[nrem])
                out[12] = 60.0 / (np.mean(rr_s[nrem]) + eps)
            if rem.sum() > 30:
                out[10] = np.std(rr_s[rem])
                out[13] = 60.0 / (np.mean(rr_s[rem]) + eps)
            if np.isfinite(out[9]) and np.isfinite(out[10]) and out[9] > eps:
                out[11] = out[10] / out[9]

    return out.astype(np.float32)


def _spo2_features(spo2, fs, stage_per_epoch):
    """
    Oximetry deserves its own treatment: hypoxic burden, not spectral entropy.
    Desaturation indices are computed against a 120 s moving baseline, over
    sleep epochs when staging is available.
    """
    out = _nan(DIM_SPO2).astype(np.float64)
    if spo2 is None or fs is None or fs <= 0 or len(spo2) < 60:
        return out.astype(np.float32)

    x = _resample(np.asarray(spo2, dtype=np.float64), fs, 1.0)   # 1 Hz
    if x.size < 300:
        return out.astype(np.float32)

    # Some sites store a fraction rather than a percentage.
    finite = x[np.isfinite(x)]
    if finite.size and np.nanmedian(finite) <= 1.5:
        x = x * 100.0
    x = np.where((x >= 50.0) & (x <= 100.0), x, np.nan)

    if stage_per_epoch is not None and len(stage_per_epoch):
        sleep_sec = np.repeat(np.isin(stage_per_epoch, SLEEP_CODES), EPOCH_SECONDS)
        n = min(len(sleep_sec), len(x))
        x_use = np.where(sleep_sec[:n], x[:n], np.nan)
        if np.isfinite(x_use).sum() < 300:
            x_use = x[:n]
    else:
        x_use = x

    valid = x_use[np.isfinite(x_use)]
    if valid.size < 300:
        return out.astype(np.float32)

    hours = valid.size / 3600.0
    out[0] = float(np.mean(valid))
    out[1] = float(np.percentile(valid, 5))
    out[2] = float(np.min(valid))
    out[3] = float(np.std(valid))
    out[4] = float(np.mean(valid < 90.0))
    out[5] = float(np.mean(valid < 88.0))

    # Desaturation events against a 2-minute moving baseline.
    filled = np.copy(x_use)
    idx = np.arange(filled.size)
    good = np.isfinite(filled)
    if good.sum() < 300:
        return out.astype(np.float32)
    filled = np.interp(idx, idx[good], filled[good])
    w = 120
    kernel = np.ones(w) / w
    baseline = np.convolve(filled, kernel, mode='same')

    for slot, drop in ((6, 3.0), (7, 4.0)):
        below = (baseline - filled) >= drop
        edges = np.diff(below.astype(np.int8), prepend=0)
        starts = np.flatnonzero(edges == 1)
        if hours > 0.1:
            out[slot] = float(len(starts)) / hours

    return out.astype(np.float32)


################################################################################
#
# Annotation features (CAISR) -- sampling-rate aware
#
################################################################################

def _channel_lookup(data, fs_map, candidates, default_fs=1.0):
    """Return (signal, fs) for the first candidate key that exists."""
    for key in candidates:
        if key in data and data[key] is not None and len(data[key]):
            fs = fs_map.get(key, default_fs) if isinstance(fs_map, dict) else default_fs
            try:
                fs = float(fs)
            except Exception:
                fs = default_fs
            if not np.isfinite(fs) or fs <= 0:
                fs = default_fs
            return np.asarray(data[key], dtype=np.float64), fs
    return None, None


def _stage_series(algo_data, algo_fs):
    """CAISR stage vector plus its own sampling rate."""
    sig, fs = _channel_lookup(algo_data, algo_fs, ['stage_caisr', 'stage'], 1.0)
    if sig is None:
        return None, 1.0
    return sig, fs


def _stage_per_epoch(stage_sig, stage_fs, n_epochs):
    """Majority stage code per 30 s epoch. 0 means unknown."""
    if stage_sig is None or n_epochs <= 0:
        return None
    spe = max(1, int(round(EPOCH_SECONDS * float(stage_fs))))
    out = np.zeros(n_epochs, dtype=np.int32)
    for i in range(n_epochs):
        seg = stage_sig[i * spe:(i + 1) * spe]
        if seg.size == 0:
            continue
        seg = seg[np.isfinite(seg)]
        seg = seg[seg < 9.0]
        if seg.size == 0:
            continue
        vals, counts = np.unique(seg.astype(np.int32), return_counts=True)
        out[i] = int(vals[np.argmax(counts)])
    return out


def _event_start_seconds(algo_data, algo_fs, candidates):
    """
    Event onsets in SECONDS. The previous revision returned sample indices and
    assumed 1 Hz for every channel, which silently mangled arousal features
    because CAISR arousals are stored at 0.5 s resolution.
    """
    sig, fs = _channel_lookup(algo_data, algo_fs, candidates, 1.0)
    if sig is None:
        return None, None
    binary = (np.nan_to_num(sig, nan=0.0) > 0).astype(np.int8)
    starts = np.flatnonzero(np.diff(binary, prepend=0) == 1)
    duration = len(binary) / fs
    return starts / fs, duration


def _bout_lengths(mask):
    """Lengths (in samples) of contiguous True runs."""
    if mask.size == 0:
        return np.array([])
    d = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return (ends - starts).astype(np.float64)


def extract_annotation_features(algo_data, algo_fs=None):
    """Sleep architecture, event indices, and arousal temporal structure."""
    out = _nan(DIM_ANNOT).astype(np.float64)
    if not algo_data:
        return out.astype(np.float32)
    if algo_fs is None:
        algo_fs = {}

    stage_sig, stage_fs = _stage_series(algo_data, algo_fs)

    resp_starts, resp_dur = _event_start_seconds(algo_data, algo_fs, ['resp_caisr', 'resp'])
    arous_starts, arous_dur = _event_start_seconds(
        algo_data, algo_fs, ['arousal_caisr', 'arousal'])
    limb_starts, limb_dur = _event_start_seconds(algo_data, algo_fs, ['limb_caisr', 'limb'])

    durations = [d for d in (resp_dur, arous_dur, limb_dur) if d]
    if stage_sig is not None:
        durations.append(len(stage_sig) / stage_fs)
    if not durations:
        return out.astype(np.float32)
    total_seconds = float(max(durations))
    total_hours = total_seconds / 3600.0
    if total_hours <= 0:
        return out.astype(np.float32)

    # ---- Sleep architecture (needed first: AHI is per hour of SLEEP) -------
    sleep_hours = np.nan
    stage_sec = None
    if stage_sig is not None and len(stage_sig):
        valid_mask = np.isfinite(stage_sig) & (stage_sig < 9.0)
        valid = stage_sig[valid_mask]
        if valid.size:
            out[3] = np.mean(valid == STAGE_CODE['wake'])
            out[4] = np.mean(valid == STAGE_CODE['n1'])
            out[5] = np.mean(valid == STAGE_CODE['n2'])
            out[6] = np.mean(valid == STAGE_CODE['n3'])
            out[7] = np.mean(valid == STAGE_CODE['rem'])
            sleep_mask = np.isin(valid, SLEEP_CODES)
            out[8] = np.mean(sleep_mask)
            out[9] = total_hours
            sleep_hours = sleep_mask.sum() / stage_fs / 3600.0
            out[10] = sleep_hours
            out[11] = (valid.size - sleep_mask.sum()) / stage_fs / 60.0

            sleep_idx = np.flatnonzero(sleep_mask)
            if sleep_idx.size:
                out[12] = sleep_idx[0] / stage_fs / 60.0
                rem_idx = np.flatnonzero(valid == STAGE_CODE['rem'])
                if rem_idx.size:
                    out[13] = max(0.0, (rem_idx[0] - sleep_idx[0]) / stage_fs / 60.0)
            # Count transitions on epoch-resolution stages, not raw samples.
            epoch_stage = valid[::max(1, int(round(EPOCH_SECONDS * stage_fs)))]
            out[14] = np.count_nonzero(np.diff(epoch_stage)) / total_hours

            # Bout structure: fragmentation is the phenotype, not just percentages.
            rem_bouts = _bout_lengths(valid == STAGE_CODE['rem'])
            if rem_bouts.size:
                out[28] = rem_bouts.size / max(rem_bouts.sum() / stage_fs / 3600.0, 1e-6)
            n3_bouts = _bout_lengths(valid == STAGE_CODE['n3'])
            if n3_bouts.size:
                out[29] = float(np.mean(n3_bouts)) / stage_fs / 60.0
            sleep_bouts = _bout_lengths(sleep_mask)
            if sleep_bouts.size:
                out[30] = float(np.max(sleep_bouts)) / stage_fs / 60.0

            stage_sec = valid_mask  # sample-level validity mask at stage_fs

    denom_hours = sleep_hours if np.isfinite(sleep_hours) and sleep_hours > 0.5 else total_hours

    if resp_starts is not None:
        out[0] = len(resp_starts) / denom_hours
    if arous_starts is not None:
        out[1] = len(arous_starts) / denom_hours
    if limb_starts is not None:
        out[2] = len(limb_starts) / denom_hours

    # ---- Arousal temporal structure (seconds, not samples) ----------------
    if arous_starts is not None and len(arous_starts) >= 3:
        iei = np.diff(arous_starts).astype(np.float64)
        out[15] = float(np.mean(iei))
        if out[15] > 1e-12:
            out[16] = float(np.std(iei) / out[15])

    half_seconds = total_seconds / 2.0
    half_hours = total_hours / 2.0
    if half_hours > 0:
        if arous_starts is not None:
            a1 = np.sum(arous_starts < half_seconds) / half_hours
            a2 = np.sum(arous_starts >= half_seconds) / half_hours
            out[17], out[18] = a1, a2
            out[19] = a1 / (a2 + 1e-6)
        if resp_starts is not None:
            out[20] = np.sum(resp_starts < half_seconds) / half_hours
            out[21] = np.sum(resp_starts >= half_seconds) / half_hours

    # ---- Arousal index per stage ------------------------------------------
    if arous_starts is not None and stage_sig is not None and len(stage_sig):
        arous_idx = np.clip((arous_starts * stage_fs).astype(int), 0, len(stage_sig) - 1)
        for slot, name in zip([22, 23, 24], ['n2', 'n3', 'rem']):
            mask = stage_sig == STAGE_CODE[name]
            hours = mask.sum() / stage_fs / 3600.0
            if hours > 0.05:
                out[slot] = float(np.sum(mask[arous_idx])) / hours

    # ---- CAISR confidence --------------------------------------------------
    prob_keys = [
        (25, ['caisr_prob_w', 'caisr_prob_wake']),
        (26, ['caisr_prob_n3']),
        (27, ['caisr_prob_arousal', 'caisr_prob_arous']),
    ]
    for slot, keys in prob_keys:
        v, _ = _channel_lookup(algo_data, algo_fs, keys, 1.0)
        if v is None:
            continue
        m = float(np.nanmean(v))
        out[slot] = m if 0.0 <= m <= 1.0 else np.nan

    return out.astype(np.float32)


def extract_human_annotations_features(human_data, human_fs=None):
    """Expert-scored equivalents. TRAIN-ONLY; never enters the feature vector."""
    if not human_data or 'resp' not in human_data:
        return _nan(DIM_ANNOT)
    renamed = {k.replace('_expert', '_caisr'): v for k, v in human_data.items()}
    renamed_fs = {}
    if human_fs:
        renamed_fs = {k.replace('_expert', '_caisr'): v for k, v in human_fs.items()}
    return extract_annotation_features(renamed, renamed_fs)


################################################################################
#
# Model, diagnostics, persistence
#
################################################################################

def _build_model(seed):
    """NaN-native gradient boosting. No scaler: trees do not need one, and
    scaling zero-filled missingness turns 'absent channel' into an outlier."""
    return HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        random_state=seed,
    )


def _verify_channel_table(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f'channel_table.csv not found at {csv_path}. Without it every '
            'physiological feature would be silently missing at inference, '
            'producing a chance-level score from a model that trained fine. '
            'Ship it alongside team_code.py.')
    try:
        load_rename_rules(os.path.abspath(csv_path))
    except Exception as e:
        raise RuntimeError(f'channel_table.csv at {csv_path} is unreadable: {e}')


def _print_site_table(groups, y, ages):
    print('\n--- label prevalence by site (swings here mean confound risk) ---')
    for s in np.unique(groups):
        m = groups == s
        print(f'  {s}: n={m.sum():5d}  prevalence={y[m].mean():.3f}  '
              f'median_age={np.nanmedian(ages[m]):.1f}')


def _print_block_nan_table(X_raw):
    print('\n--- NaN fraction by block ---')
    blocks = [('demographics', _demo_slice()), ('physiological', _phys_slice()),
              ('annotations', _annot_slice())]
    for name, sl in blocks:
        if sl.stop - sl.start <= 0:
            continue
        sub = X_raw[:, sl]
        print(f'  {name:14s}: mean NaN {np.isnan(sub).mean():.3f}, '
              f'all-NaN records {int(np.isnan(sub).all(axis=1).sum())}')


def _make_splits(X, y, groups):
    n_sites = len(np.unique(groups))
    if n_sites >= 2 and _SGKFold is not None:
        return list(_SGKFold(n_splits=min(n_sites, 5)).split(X, y, groups)), True
    if n_sites >= 2 and _GKFold is not None:
        return list(_GKFold(n_splits=min(n_sites, 5)).split(X, y, groups)), True
    return list(StratifiedKFold(n_splits=5, shuffle=True,
                                random_state=RANDOM_STATE).split(X, y)), False


def _oof_predictions(X_raw, y, ages, splits, cols=None, use_brain_age=USE_BRAIN_AGE):
    """
    Out-of-fold probabilities. The brain-age regressor is refit inside every
    fold on training-fold negatives only, so the residual feature carries no
    leakage into the held-out site.
    """
    pred = np.full(len(y), np.nan)
    for tr, va in splits:
        if len(np.unique(y[tr])) < 2:
            continue
        ba = _fit_brain_age(X_raw[tr], y[tr], ages[tr]) if use_brain_age else None
        Xtr = _augment(X_raw[tr], ages[tr], ba) if use_brain_age else X_raw[tr]
        Xva = _augment(X_raw[va], ages[va], ba) if use_brain_age else X_raw[va]
        if cols is not None:
            Xtr, Xva = Xtr[:, cols], Xva[:, cols]
        w = age_balanced_weights(ages[tr], y[tr])
        m = _build_model(RANDOM_STATE)
        m.fit(Xtr, y[tr], sample_weight=w)
        pred[va] = m.predict_proba(Xva)[:, 1]
    return pred


def _run_diagnostic(X_raw, y, ages, groups, grid, prev_values, global_p, verbose):
    """
    Site-grouped out-of-fold evaluation on the Challenge's own metrics, and
    selection of the threshold scale factor. Always runs -- a submission must
    never depend on whether logging was switched on.
    """
    splits, grouped = _make_splits(X_raw, y, groups)

    pred = _oof_predictions(X_raw, y, ages, splits)
    ok = np.isfinite(pred)

    ac = age_conditioned_auroc(y[ok], pred[ok], ages[ok])
    plain = roc_auc_score(y[ok], pred[ok]) if len(np.unique(y[ok])) == 2 else np.nan

    if verbose:
        print('\n--- out-of-fold evaluation '
              f'({"site-grouped" if grouped else "stratified"} folds) ---')
        print(f'  age-conditioned AUROC (THE metric): {ac:.4f}')
        print(f'  plain AUROC (confounded, for reference): {plain:.4f}')
        if np.isfinite(ac) and np.isfinite(plain) and plain - ac > 0.08:
            print('  ! large gap: most of the plain AUROC is the age gradient.')
        if np.isfinite(ac) and ac < 0.52:
            print('  ! conditioned OOF is at chance. Do not spend a submission on this.')

    if verbose and RUN_ABLATIONS and USE_DEMOGRAPHICS:
        n_raw = X_raw.shape[1]
        demo_cols = np.arange(DIM_DEMO)
        sig_cols = _signal_cols(n_raw)
        pd_ = _oof_predictions(X_raw, y, ages, splits, cols=demo_cols, use_brain_age=False)
        ps_ = _oof_predictions(X_raw, y, ages, splits, cols=sig_cols, use_brain_age=False)
        okd, oks = np.isfinite(pd_), np.isfinite(ps_)
        print(f'  demographics only : conditioned {age_conditioned_auroc(y[okd], pd_[okd], ages[okd]):.4f}')
        print(f'  signal only       : conditioned {age_conditioned_auroc(y[oks], ps_[oks], ages[oks]):.4f}')
        if USE_BRAIN_AGE:
            ba_only = _oof_predictions(X_raw, y, ages, splits,
                                       cols=np.arange(X_raw.shape[1], X_raw.shape[1] + DIM_BRAIN_AGE))
            okb = np.isfinite(ba_only)
            if okb.any():
                print(f'  brain-age gap only: conditioned '
                      f'{age_conditioned_auroc(y[okb], ba_only[okb], ages[okb]):.4f}')

    # ---- Threshold: odds-space correction on the prior-implied cut ---------
    alpha, best_reward = 1.0, -np.inf
    if ok.sum() >= 20 and len(np.unique(y[ok])) == 2:
        p_a = np.array([prevalence_at_age(a, grid, prev_values, global_p) for a in ages[ok]])
        baseline = prevalence_reward(y[ok], np.zeros(ok.sum(), int), ages[ok],
                                     grid, prev_values, global_p)
        for cand in np.logspace(-1.0, 1.7, 79):
            thr = np.array([decision_threshold(p, cand) for p in p_a])
            binary = (pred[ok] >= thr).astype(int)
            r = prevalence_reward(y[ok], binary, ages[ok], grid, prev_values, global_p)
            if r > best_reward:
                best_reward, alpha = r, float(cand)

        # A rule that cannot beat "predict nothing" out-of-fold should not be
        # shipped; fall back to the most conservative cut on the grid.
        if best_reward < baseline:
            alpha = float(np.logspace(-1.0, 1.7, 79)[-1])
            thr = np.array([decision_threshold(p, alpha) for p in p_a])
            best_reward = prevalence_reward(y[ok], (pred[ok] >= thr).astype(int),
                                            ages[ok], grid, prev_values, global_p)

        if verbose:
            thr = np.array([decision_threshold(p, alpha) for p in p_a])
            rate = float(np.mean(pred[ok] >= thr))
            base_desc = '0.5 (age-balanced training)' if AGE_BALANCED_WEIGHTS else 'p_a(age)'
            print(f'  threshold rule: cut at {base_desc} with odds factor {alpha:.2f} '
                  f'-> median threshold {np.median(thr):.3f}')
            print(f'    out-of-fold reward {best_reward:.4f}, '
                  f'positive rate {rate:.3f}, cohort prevalence {y[ok].mean():.3f}')
            print(f'    (an all-negative classifier would score {baseline:.4f})')

    return alpha, pred


def save_model(model_folder, models, brain_age_model, alpha, grid, prev_values, global_p):
    bundle = {
        'models': models,
        'brain_age_model': brain_age_model,
        'alpha': float(alpha),
        'prev_grid': np.asarray(grid, dtype=np.float64),
        'prev_values': np.asarray(prev_values, dtype=np.float64),
        'global_prevalence': float(global_p),
        'use_demographics': USE_DEMOGRAPHICS,
        'use_brain_age': USE_BRAIN_AGE,
        'age_balanced_weights': bool(AGE_BALANCED_WEIGHTS),
        'n_features': _total_feature_dim(),
        'feature_names': feature_names(),
        'target_fs': TARGET_FS,
        'epoch_seconds': EPOCH_SECONDS,
        'age_delta': AGE_DELTA,
    }
    joblib.dump(bundle, os.path.join(model_folder, 'model.sav'))


################################################################################
#
# Offline audit. Run this on the supplementary set before any submission:
#
#     python team_code.py audit /path/to/supplementary_set
#
# It answers the only question that matters first: are the features that exist
# in training also populated on the hidden sources?
#
################################################################################

def audit_folder(data_folder, csv_path=DEFAULT_CSV_PATH, limit=None):
    _verify_channel_table(csv_path)
    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    records = find_patients(patient_data_file)
    if limit:
        records = records[:limit]

    rows, ages = [], []
    for record in records:
        try:
            vec, age = _assemble(record, data_folder, csv_path=csv_path)
            rows.append(vec)
            ages.append(age)
        except Exception as e:
            print(f'  !!! {record.get(HEADERS["bids_folder"], "?")}: {e}')

    if not rows:
        print('No records could be processed.')
        return

    X = np.asarray(rows, dtype=np.float32)
    print(f'\nAudited {X.shape[0]} records from {data_folder}')
    print(f'Age available for {np.mean(np.isfinite(ages)):.2f} of records')
    _print_block_nan_table(X)
    print(f'Fallback counts: {_FALLBACK_COUNTS}')

    names = raw_feature_names()
    frac = np.isnan(X).mean(axis=0)
    dead = [n for n, f in zip(names, frac) if f > 0.9]
    print(f'\nFeatures missing in >90% of these records ({len(dead)} of {len(names)}):')
    for n in dead:
        print(f'  {n}')
    if len(dead) > 0.3 * len(names):
        print('\n! A large share of the feature vector is unavailable here. '
              'Fix channel mapping before submitting; nothing downstream can '
              'compensate for features that are simply absent at inference.')


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == 'audit':
        audit_folder(sys.argv[2])
    else:
        print(__doc__)
