

import os
import sys
import joblib
import warnings

import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import kurtosis as _kurtosis

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from tqdm import tqdm

warnings.filterwarnings('ignore')

from helper_code import *

# Optional imports used only by the offline diagnostic.
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
# Demographics are present on the hidden sets, so they are safe at inference.
# They are also the most obvious route to a site/age confound, so the offline
# diagnostic reports demographics-only and signal-only AUROC separately. If
# demographics-only >> signal-only on grouped CV, set this to False.
USE_DEMOGRAPHICS = True

# Human expert annotations are NOT on the hidden sets. Never in the vector.
USE_HUMAN_ANNOTATIONS = False

# ---- Signal processing -----------------------------------------------------
TARGET_FS = 64           # resample every analysed channel to 64 Hz
EPOCH_SECONDS = 30       # standard PSG scoring epoch
MAX_HOURS = 12           # sanity cap on recording length
MAX_EEG_CHANNELS = 6     # cap EEG channels averaged, for runtime
MIN_VALID_EPOCHS = 10    # below this a channel is treated as unusable

# ---- Model -----------------------------------------------------------------
N_ENSEMBLE_FOLDS = 5
RANDOM_STATE = 42
RUN_DIAGNOSTIC = True    # print site-grouped OOF AUROC during training


################################################################################
#
# Feature registry. Single source of truth for names and dimensions.
#
################################################################################

# Per-epoch, per-channel features. All scale-invariant.
EPOCH_FEATURE_NAMES = [
    'rel_amp',        # log(epoch IQR / night IQR)
    'rel_delta',      # 0.5-4 Hz power / total 0.5-30 Hz power
    'rel_theta',      # 4-8
    'rel_alpha',      # 8-12
    'rel_sigma',      # 12-15
    'rel_beta',       # 15-30
    'slowing',        # log((delta+theta)/(alpha+beta))
    'spec_ent',       # spectral entropy over 0.5-30
    'sef95',          # 95% spectral edge frequency
    'hj_mob',         # Hjorth mobility
    'hj_comp',        # Hjorth complexity
    'kurt',           # kurtosis
]
N_EPOCH_FEATURES = len(EPOCH_FEATURE_NAMES)          # 12
# index 12 of the internal epoch matrix is the sigma peak frequency, used by
# the spindle block rather than the generic aggregation.
IDX_SIGMA_PEAK = N_EPOCH_FEATURES

# Non-EEG groups get a reduced subset (band ratios are less meaningful there).
OTHER_FEATURE_IDX = [0, 7, 8, 9, 10, 11]             # amp, ent, sef95, mob, comp, kurt

EEG_STATS = ['mean', 'std', 'p10', 'p90']
OTHER_STATS = ['mean', 'std']

OTHER_GROUPS = ['eog', 'chin', 'leg', 'ecg', 'resp', 'spo2']

# CAISR stage codes, per the challenge annotation convention.
STAGE_CODE = {'n3': 1, 'n2': 2, 'n1': 3, 'rem': 4, 'wake': 5}
STAGE_BLOCKS = ['n3', 'n2', 'rem', 'wake']
CONTRAST_IDX = [1, 2, 4, 6]                          # delta, theta, sigma, slowing

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

ANNOT_FEATURE_NAMES = [
    'ahi', 'arousal_index', 'limb_index',
    'pct_wake', 'pct_n1', 'pct_n2', 'pct_n3', 'pct_rem',
    'sleep_efficiency', 'record_hours', 'sleep_hours', 'waso_min',
    'sleep_latency_min', 'rem_latency_min', 'stage_transitions_per_hr',
    'arousal_iei_mean', 'arousal_iei_cv',
    'arousal_index_h1', 'arousal_index_h2', 'arousal_h1_h2_ratio',
    'ahi_h1', 'ahi_h2',
    'arousal_index_n2', 'arousal_index_n3', 'arousal_index_rem',
    'caisr_prob_w', 'caisr_prob_n3', 'caisr_prob_arous',
]

DEMO_FEATURE_NAMES = [
    'age', 'sex_f', 'sex_m', 'sex_other',
    'race_asian', 'race_black', 'race_other', 'race_unavailable', 'race_white',
    'bmi',
]

DIM_EEG_GLOBAL = N_EPOCH_FEATURES * len(EEG_STATS)                       # 48
DIM_OTHER_GLOBAL = len(OTHER_GROUPS) * len(OTHER_FEATURE_IDX) * len(OTHER_STATS)  # 72
DIM_EEG_STAGE = N_EPOCH_FEATURES * len(STAGE_BLOCKS)                     # 48
DIM_CONTRAST = len(CONTRAST_IDX) * 2                                     # 8
DIM_SPINDLE = len(SPINDLE_FEATURE_NAMES)                                 # 8
DIM_HRV = len(HRV_FEATURE_NAMES)                                         # 14
DIM_ANNOT = len(ANNOT_FEATURE_NAMES)                                     # 28
DIM_DEMO = len(DEMO_FEATURE_NAMES)                                       # 10

DIM_PHYS = (DIM_EEG_GLOBAL + DIM_OTHER_GLOBAL + DIM_EEG_STAGE
            + DIM_CONTRAST + DIM_SPINDLE + DIM_HRV)                      # 198


def _total_feature_dim():
    dim = DIM_PHYS + DIM_ANNOT
    if USE_DEMOGRAPHICS:
        dim += DIM_DEMO
    return dim


def feature_names():
    """Ordered feature names, same order as `assemble_features` output."""
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
    names += ['annot:' + n for n in ANNOT_FEATURE_NAMES]
    return names


# Counters so silent fallbacks become visible instead of being swallowed.
_FALLBACK_COUNTS = {'demo': 0, 'phys': 0, 'annot': 0, 'eeg': 0, 'ecg': 0}


def _nan(n):
    return np.full(int(n), np.nan, dtype=np.float32)


################################################################################
#
# Required functions. Do NOT change the arguments of these functions.
#
################################################################################

def train_model(data_folder, model_folder, verbose, csv_path=DEFAULT_CSV_PATH):
    """Extract features, run an offline diagnostic, train a fold ensemble."""

    # ---- Fail loud on a missing channel table -----------------------------
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

    features, labels, sites = [], [], []

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

            feature_vec = assemble_features(record, data_folder, csv_path=csv_path)

            features.append(feature_vec)
            labels.append(int(label))
            sites.append(str(site_id))

        except Exception as e:
            tqdm.write(f'  !!! Error processing record {i + 1} ({patient_id}): {e}')
            continue

    pbar.close()

    if len(labels) == 0:
        raise ValueError('No valid labeled records found for training.')

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)
    groups = np.asarray(sites)

    # HistGB accepts NaN but not inf.
    X[~np.isfinite(X) & ~np.isnan(X)] = np.nan

    expected = _total_feature_dim()
    if X.shape[1] != expected:
        raise ValueError(f'Feature dimension {X.shape[1]} != expected {expected}. '
                         'Train and inference would disagree; aborting.')

    if verbose:
        n_pos = int(y.sum())
        print(f'\nTraining set: {len(y)} records ({n_pos} positive, {len(y) - n_pos} negative)')
        print(f'Feature vector dimension: {X.shape[1]}')
        print(f'NaN fraction overall: {np.isnan(X).mean():.3f}')
        print(f'Records with an all-NaN physiological block: '
              f'{int(np.isnan(X[:, _phys_slice()]).all(axis=1).sum())}')
        print('Fallback counts (blocks that failed and were set to NaN): '
              f'{_FALLBACK_COUNTS}')
        _print_site_table(groups, y)

    # ---- Honest offline estimate before spending a submission -------------
    threshold = 0.5
    if RUN_DIAGNOSTIC and verbose:
        threshold = _run_diagnostic(X, y, groups)

    # ---- Final fold ensemble on all data ----------------------------------
    if verbose:
        print(f'\nTraining {N_ENSEMBLE_FOLDS}-fold ensemble on all data...')

    models = []
    n_splits = min(N_ENSEMBLE_FOLDS, int(np.bincount(y).min()))
    if n_splits < 2:
        models.append(_build_model(RANDOM_STATE).fit(X, y))
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        for k, (tr, _) in enumerate(skf.split(X, y)):
            m = _build_model(RANDOM_STATE + k).fit(X[tr], y[tr])
            models.append(m)

    os.makedirs(model_folder, exist_ok=True)
    save_model(model_folder, models, threshold)

    if verbose:
        print('Done.')
        print()


def load_model(model_folder, verbose):
    """Load the trained bundle. Fails loudly if the channel table is absent."""
    _verify_channel_table(DEFAULT_CSV_PATH)
    bundle = joblib.load(os.path.join(model_folder, 'model.sav'))
    if bundle.get('n_features') != _total_feature_dim():
        raise ValueError(
            f"Model was trained with {bundle.get('n_features')} features but this "
            f"code produces {_total_feature_dim()}. Refusing to run.")
    return bundle


def run_model(model, record, data_folder, verbose):
    """Inference on one record. Returns (binary_label, probability)."""
    models = model['models']
    threshold = float(model.get('threshold', 0.5))

    x = assemble_features(record, data_folder, csv_path=DEFAULT_CSV_PATH).reshape(1, -1)
    x[~np.isfinite(x) & ~np.isnan(x)] = np.nan

    probs = [float(m.predict_proba(x)[0][1]) for m in models]
    probability_output = float(np.mean(probs))
    binary_output = int(probability_output >= threshold)

    return binary_output, probability_output


################################################################################
#
# Shared feature assembly (used by BOTH train_model and run_model)
#
################################################################################

def _phys_slice():
    start = DIM_DEMO if USE_DEMOGRAPHICS else 0
    return slice(start, start + DIM_PHYS)


def assemble_features(record, data_folder, csv_path=DEFAULT_CSV_PATH):
    """
    Fixed-length feature vector for one record.

    Order (identical at train and inference):
        [ demographics(10) ] + physiological(198) + annotations(28)

    Any block whose source is missing or unreadable becomes NaN of the correct
    length, so the dimension is constant and "missing" stays distinguishable
    from "measured zero".
    """
    patient_id = record[HEADERS['bids_folder']]
    site_id = record[HEADERS['site_id']]
    session_id = record[HEADERS['session_id']]

    blocks = []

    # ---- Demographics ------------------------------------------------------
    if USE_DEMOGRAPHICS:
        try:
            demo_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
            patient_data = load_demographics(demo_file, patient_id, session_id)
            blocks.append(extract_demographic_features(patient_data))
        except Exception:
            _FALLBACK_COUNTS['demo'] += 1
            blocks.append(_nan(DIM_DEMO))

    # ---- Algorithmic (CAISR) annotations, loaded first for the hypnogram ---
    algo_file = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
                             site_id, f'{patient_id}_ses-{session_id}_caisr_annotations.edf')
    algo_data = None
    if os.path.exists(algo_file):
        try:
            algo_data, _ = load_signal_data(algo_file)
        except Exception:
            algo_data = None
    if algo_data is None:
        _FALLBACK_COUNTS['annot'] += 1
        annot_features = _nan(DIM_ANNOT)
        stage_per_second = None
    else:
        annot_features = extract_annotation_features(algo_data)
        stage_per_second = _stage_series(algo_data)

    # ---- Physiological signals --------------------------------------------
    phys_file = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER,
                             site_id, f'{patient_id}_ses-{session_id}.edf')
    phys_features = _nan(DIM_PHYS)
    if os.path.exists(phys_file):
        try:
            phys_data, phys_fs = load_signal_data(phys_file)
            phys_features = extract_physiological_features(
                phys_data, phys_fs, stage_per_second, csv_path=csv_path)
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
    return vec


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

    # Implausible values are missingness, not measurements.
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
                                   stage_per_second, csv_path=DEFAULT_CSV_PATH):
    """
    Returns a length-DIM_PHYS vector:
        EEG global stats (48) + other-group global stats (72)
        + EEG per-stage means (48) + EEG stage contrasts (8)
        + spindle/sigma block (8) + HRV block (14)
    """
    channels, fs_map = _standardize_and_derive(physiological_data, physiological_fs, csv_path)

    # ---- Channel group candidates -----------------------------------------
    eeg_candidates = ['c3-m2', 'c4-m1', 'f3-m2', 'f4-m1', 'o1-m2', 'o2-m1']
    other_candidates = {
        'eog':  ['e1-m2', 'e2-m1'],
        'chin': ['chin1-chin2', 'chin'],
        'leg':  ['lat', 'rat'],
        'ecg':  ['ecg', 'ekg'],
        'resp': ['airflow', 'ptaf', 'abd', 'chest'],
        'spo2': ['spo2', 'sao2'],
    }

    max_epochs = int(MAX_HOURS * 3600 // EPOCH_SECONDS)

    # ---- EEG: average the epoch feature matrix across all present channels -
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

    # ---- Stage vector aligned to epochs -----------------------------------
    n_epochs = eeg_mat.shape[0] if eeg_mat is not None else 0
    stage_per_epoch = _stage_per_epoch(stage_per_second, n_epochs)

    # ---- Block 1: EEG global stats ----------------------------------------
    if eeg_mat is not None:
        eeg_global = _aggregate(eeg_mat[:, :N_EPOCH_FEATURES], EEG_STATS)
    else:
        eeg_global = _nan(DIM_EEG_GLOBAL)

    # ---- Block 2: other groups, global stats ------------------------------
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

    # ---- Blocks 3 and 4: per-stage means and contrasts --------------------
    if eeg_mat is not None and stage_per_epoch is not None:
        stage_means, contrasts = _stage_blocks(eeg_mat[:, :N_EPOCH_FEATURES], stage_per_epoch)
    else:
        stage_means, contrasts = _nan(DIM_EEG_STAGE), _nan(DIM_CONTRAST)

    # ---- Block 5: spindles / sigma ----------------------------------------
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

    # ---- Block 6: HRV ------------------------------------------------------
    if ecg_raw is not None and ecg_fs:
        hrv = _hrv_features(np.asarray(ecg_raw, dtype=np.float64), float(ecg_fs),
                            stage_per_epoch)
    else:
        _FALLBACK_COUNTS['ecg'] += 1
        hrv = _nan(DIM_HRV)

    del channels
    out = np.hstack([eeg_global, other_global, stage_means, contrasts, spindle, hrv])
    return out.astype(np.float32)


def _standardize_and_derive(physiological_data, physiological_fs, csv_path):
    """Rename channels to the standard vocabulary and build bipolar derivations."""
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

    return channels, fs_map


def _resample(sig, original_fs, target_fs):
    """Linear-interpolation resample of a 1D signal."""
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
    Vectorised per-epoch features.

    Returns (n_epochs, N_EPOCH_FEATURES + 1); the trailing column is the sigma
    peak frequency, consumed by the spindle block. Flat or degenerate epochs
    become all-NaN rows. Returns None if the channel is unusable.
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

    # Robust amplitude, expressed in units of the channel's own night-long IQR,
    # so amplifier gain cancels.
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

    # Flat / dead epochs carry no information; mark them missing.
    dead = (var0 <= eps) | (epoch_iqr <= eps)
    out[dead, :] = np.nan

    if np.isfinite(out[:, :N_EPOCH_FEATURES]).any(axis=1).sum() < MIN_VALID_EPOCHS:
        return None
    return out


def _aggregate(mat, stats):
    """Summarise an (n_epochs, n_features) matrix into a flat vector."""
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


def _stage_series(algo_data):
    """1 Hz CAISR stage vector, or None."""
    stages = algo_data.get('stage_caisr', None)
    if stages is None or len(stages) == 0:
        return None
    return np.asarray(stages, dtype=np.float64)


def _stage_per_epoch(stage_per_second, n_epochs):
    """Majority stage code per 30 s epoch. 0 means unknown."""
    if stage_per_second is None or n_epochs <= 0:
        return None
    out = np.zeros(n_epochs, dtype=np.int32)
    for i in range(n_epochs):
        seg = stage_per_second[i * EPOCH_SECONDS:(i + 1) * EPOCH_SECONDS]
        seg = seg[np.isfinite(seg)]
        seg = seg[seg < 9.0]
        if seg.size == 0:
            continue
        vals, counts = np.unique(seg.astype(np.int32), return_counts=True)
        out[i] = int(vals[np.argmax(counts)])
    return out


def _stage_blocks(mat, stage_per_epoch):
    """Per-stage means plus N3-vs-REM and NREM-vs-wake contrasts."""
    n = min(mat.shape[0], len(stage_per_epoch))
    mat, stage = mat[:n], stage_per_epoch[:n]

    means = {}
    for name in STAGE_BLOCKS:
        m = stage == STAGE_CODE[name]
        means[name] = np.nanmean(mat[m], axis=0) if m.sum() >= 3 else _nan(N_EPOCH_FEATURES)

    nrem_mask = np.isin(stage, [STAGE_CODE['n1'], STAGE_CODE['n2'], STAGE_CODE['n3']])
    nrem = np.nanmean(mat[nrem_mask], axis=0) if nrem_mask.sum() >= 3 else _nan(N_EPOCH_FEATURES)

    stage_means = np.hstack([means[name] for name in STAGE_BLOCKS]).astype(np.float32)

    contrasts = []
    for i in CONTRAST_IDX:
        contrasts.append(means['n3'][i] - means['rem'][i])
        contrasts.append(nrem[i] - means['wake'][i])
    return stage_means, np.asarray(contrasts, dtype=np.float32)


def _spindle_features(sig, fs, eeg_mat, stage_per_epoch):
    """
    Sigma peak frequency and spindle density.

    Dresden's SHAP analysis singled out mean sigma peak frequency, tied to
    thalamocortical loop degradation and reduced spindles. Absolute sigma band
    power is not the same quantity, so both are computed here explicitly.
    """
    out = _nan(DIM_SPINDLE).astype(np.float64)
    eps = 1e-12
    fs = int(round(fs))

    sigma_peak = eeg_mat[:, IDX_SIGMA_PEAK]
    if stage_per_epoch is not None:
        n = min(len(sigma_peak), len(stage_per_epoch))
        stage = stage_per_epoch[:n]
        sp = sigma_peak[:n]
        sleep_mask = np.isin(stage, [1, 2, 3, 4])
        n2_mask = stage == STAGE_CODE['n2']
    else:
        n = len(sigma_peak)
        stage = None
        sp = sigma_peak
        sleep_mask = np.ones(n, dtype=bool)
        n2_mask = np.zeros(n, dtype=bool)

    if sleep_mask.sum() >= 3:
        out[0] = np.nanmean(sp[sleep_mask])
    if n2_mask.sum() >= 3:
        out[1] = np.nanmean(sp[n2_mask])
        out[2] = np.nanstd(sp[n2_mask])

    # ---- Envelope-based spindle detection ---------------------------------
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

    sample_sleep = np.zeros(n_use * spe, dtype=bool)
    sample_n2 = np.zeros(n_use * spe, dtype=bool)
    for i in range(n_use):
        if sleep_mask[i]:
            sample_sleep[i * spe:(i + 1) * spe] = True
        if n2_mask[i]:
            sample_n2[i * spe:(i + 1) * spe] = True

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
    """
    Heart-rate variability from detected R peaks.

    The previous entry ran generic std / skew / band-power on the ECG trace,
    which is close to meaningless. HRV needs R-peak detection followed by
    RR-interval analysis, and the autonomic signal is stage-dependent.
    """
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

    # Frequency-domain HRV on a 4 Hz resampled tachogram.
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

    # Stage-resolved HRV.
    if stage_per_epoch is not None and len(stage_per_epoch):
        beat_epoch = (t_peaks[1:][ok] // EPOCH_SECONDS).astype(int)
        valid = beat_epoch < len(stage_per_epoch)
        beat_epoch, rr_s = beat_epoch[valid], rr_clean[valid]
        if len(rr_s) > 30:
            st = stage_per_epoch[beat_epoch]
            nrem = np.isin(st, [STAGE_CODE['n1'], STAGE_CODE['n2'], STAGE_CODE['n3']])
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


################################################################################
#
# Annotation features (CAISR)
#
################################################################################

def _event_starts(algo_data, key):
    """Sample indices (1 Hz) where a discrete event begins."""
    if key not in algo_data:
        return None
    sig = np.asarray(algo_data[key], dtype=np.float64)
    if sig.size == 0:
        return None
    binary = (sig > 0).astype(np.int8)
    return np.flatnonzero(np.diff(binary, prepend=0) == 1)


def extract_annotation_features(algo_data):
    """Sleep architecture, event indices, and arousal temporal structure."""
    out = _nan(DIM_ANNOT).astype(np.float64)
    if not algo_data:
        return out.astype(np.float32)

    resp = algo_data.get('resp_caisr', None)
    n_sec = len(resp) if resp is not None else 0
    stages = _stage_series(algo_data)
    if n_sec == 0 and stages is not None:
        n_sec = len(stages)
    total_hours = n_sec / 3600.0
    if total_hours <= 0:
        return out.astype(np.float32)

    resp_starts = _event_starts(algo_data, 'resp_caisr')
    arous_starts = _event_starts(algo_data, 'arousal_caisr')
    limb_starts = _event_starts(algo_data, 'limb_caisr')

    if resp_starts is not None:
        out[0] = len(resp_starts) / total_hours
    if arous_starts is not None:
        out[1] = len(arous_starts) / total_hours
    if limb_starts is not None:
        out[2] = len(limb_starts) / total_hours

    # ---- Sleep architecture ------------------------------------------------
    if stages is not None and len(stages):
        valid = stages[np.isfinite(stages) & (stages < 9.0)]
        if valid.size:
            out[3] = np.mean(valid == STAGE_CODE['wake'])
            out[4] = np.mean(valid == STAGE_CODE['n1'])
            out[5] = np.mean(valid == STAGE_CODE['n2'])
            out[6] = np.mean(valid == STAGE_CODE['n3'])
            out[7] = np.mean(valid == STAGE_CODE['rem'])
            sleep_mask = np.isin(valid, [1, 2, 3, 4])
            out[8] = np.mean(sleep_mask)
            out[9] = total_hours
            out[10] = sleep_mask.sum() / 3600.0
            out[11] = (valid.size - sleep_mask.sum()) / 60.0

            sleep_idx = np.flatnonzero(sleep_mask)
            if sleep_idx.size:
                out[12] = sleep_idx[0] / 60.0
                rem_idx = np.flatnonzero(valid == STAGE_CODE['rem'])
                if rem_idx.size:
                    out[13] = max(0.0, (rem_idx[0] - sleep_idx[0]) / 60.0)
            out[14] = np.count_nonzero(np.diff(valid)) / total_hours

    # ---- Arousal temporal structure ---------------------------------------
    if arous_starts is not None and len(arous_starts) >= 3:
        iei = np.diff(arous_starts).astype(np.float64)
        out[15] = np.mean(iei)
        if out[15] > 1e-12:
            out[16] = np.std(iei) / out[15]

    half = n_sec / 2.0
    half_hours = total_hours / 2.0
    if half_hours > 0:
        if arous_starts is not None:
            a1 = np.sum(arous_starts < half) / half_hours
            a2 = np.sum(arous_starts >= half) / half_hours
            out[17], out[18] = a1, a2
            out[19] = a1 / (a2 + 1e-6)
        if resp_starts is not None:
            out[20] = np.sum(resp_starts < half) / half_hours
            out[21] = np.sum(resp_starts >= half) / half_hours

    # ---- Arousal index per stage ------------------------------------------
    if arous_starts is not None and stages is not None and len(stages):
        for slot, name in zip([22, 23, 24], ['n2', 'n3', 'rem']):
            mask = stages == STAGE_CODE[name]
            hours = mask.sum() / 3600.0
            if hours > 0.05:
                idx = arous_starts[arous_starts < len(stages)]
                out[slot] = np.sum(mask[idx]) / hours

    # ---- CAISR confidence --------------------------------------------------
    for slot, key in zip([25, 26, 27], ['caisr_prob_w', 'caisr_prob_n3', 'caisr_prob_arous']):
        v = algo_data.get(key, None)
        if v is None or len(v) == 0:
            continue
        m = float(np.nanmean(np.asarray(v, dtype=np.float64)))
        # A probability outside [0, 1] is a broken channel, not a low value.
        out[slot] = m if 0.0 <= m <= 1.0 else np.nan

    return out.astype(np.float32)


################################################################################
#
# Human (expert) annotations. TRAIN-ONLY, never in the feature vector.
#
################################################################################

def extract_human_annotations_features(human_data):
    """Expert-scored equivalents. Unavailable on the hidden sets by design."""
    if not human_data or 'resp_expert' not in human_data:
        return _nan(DIM_ANNOT)
    renamed = {}
    for k, v in human_data.items():
        renamed[k.replace('_expert', '_caisr')] = v
    return extract_annotation_features(renamed)


################################################################################
#
# Model, diagnostics, persistence
#
################################################################################

def _build_model(seed):
    """
    NaN-native gradient boosting. No StandardScaler: trees do not need it, and
    scaling zero-filled missingness is what turned "absent channel" into an
    extreme outlier in the previous entry.
    """
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
    """A missing channel table must fail loudly, not silently zero the signal."""
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


def _print_site_table(groups, y):
    print('\n--- label prevalence by site (swings here mean confound risk) ---')
    for s in np.unique(groups):
        m = groups == s
        print(f'  {s}: n={m.sum():4d}  prevalence={y[m].mean():.3f}')


def _run_diagnostic(X, y, groups):
    """
    Site-grouped out-of-fold AUROC with ablations. Costs no submission.

    Read it as: grouped OOF is the number that should track the leaderboard.
    If demographics-only clearly beats signal-only, the model is an age
    classifier and USE_DEMOGRAPHICS should be reconsidered.
    """
    print('\n--- site-grouped out-of-fold AUROC (this tracks the leaderboard) ---')
    n_sites = len(np.unique(groups))
    if n_sites < 2:
        print('  only one site present, falling back to stratified CV')
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        split = list(splitter.split(X, y))
    elif _SGKFold is not None:
        splitter = _SGKFold(n_splits=min(n_sites, 5))
        split = list(splitter.split(X, y, groups))
    elif _GKFold is not None:
        split = list(_GKFold(n_splits=min(n_sites, 5)).split(X, y, groups))
    else:
        split = list(StratifiedKFold(n_splits=5, shuffle=True,
                                     random_state=RANDOM_STATE).split(X, y))

    demo_dim = DIM_DEMO if USE_DEMOGRAPHICS else 0

    def oof(cols):
        pred = np.full(len(y), np.nan)
        for tr, va in split:
            if len(np.unique(y[tr])) < 2:
                continue
            m = _build_model(RANDOM_STATE)
            m.fit(X[np.ix_(tr, cols)], y[tr])
            pred[va] = m.predict_proba(X[np.ix_(va, cols)])[:, 1]
        ok = np.isfinite(pred)
        if ok.sum() < 10 or len(np.unique(y[ok])) < 2:
            return np.nan, pred
        return roc_auc_score(y[ok], pred[ok]), pred

    all_cols = np.arange(X.shape[1])
    auc_full, pred_full = oof(all_cols)
    print(f'  full        ({X.shape[1]:3d} feats): {auc_full:.3f}')
    if demo_dim:
        auc_d, _ = oof(np.arange(demo_dim))
        auc_s, _ = oof(np.arange(demo_dim, X.shape[1]))
        print(f'  demographics ({demo_dim:3d} feats): {auc_d:.3f}')
        print(f'  signal only  ({X.shape[1] - demo_dim:3d} feats): {auc_s:.3f}')
        if np.isfinite(auc_d) and np.isfinite(auc_s) and auc_d > auc_s + 0.05:
            print('  ! demographics dominate. Consider USE_DEMOGRAPHICS = False.')
    if np.isfinite(auc_full) and auc_full < 0.52:
        print('  ! grouped OOF is at chance. Do not spend a submission on this.')

    # Threshold for the binary output, chosen on out-of-fold predictions.
    ok = np.isfinite(pred_full)
    best_thr, best_f1 = 0.5, -1.0
    if ok.sum() >= 10 and len(np.unique(y[ok])) == 2:
        for thr in np.linspace(0.05, 0.95, 91):
            pred_bin = (pred_full[ok] >= thr).astype(int)
            tp = np.sum((pred_bin == 1) & (y[ok] == 1))
            fp = np.sum((pred_bin == 1) & (y[ok] == 0))
            fn = np.sum((pred_bin == 0) & (y[ok] == 1))
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        print(f'  binary threshold {best_thr:.2f} (out-of-fold F1 {best_f1:.3f})')
    return float(best_thr)


def save_model(model_folder, models, threshold):
    """Persist the fold ensemble plus the config needed to reproduce features."""
    bundle = {
        'models': models,
        'threshold': float(threshold),
        'use_demographics': USE_DEMOGRAPHICS,
        'n_features': _total_feature_dim(),
        'feature_names': feature_names(),
        'target_fs': TARGET_FS,
        'epoch_seconds': EPOCH_SECONDS,
    }
    joblib.dump(bundle, os.path.join(model_folder, 'model.sav'))
