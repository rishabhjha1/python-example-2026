

import joblib
import numpy as np
import os
import re
import sys
import warnings

from scipy import signal as scipy_signal
from scipy.stats import skew, kurtosis, entropy
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve

warnings.filterwarnings('ignore')

from helper_code import *

# numpy 1.x / 2.x compatibility
_trapz = getattr(np, 'trapezoid', None) or np.trapz

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x=None, **k):
        return x if x is not None else None

try:
    import lightgbm as lgb
    HAVE_LGB = True
except Exception:
    HAVE_LGB = False

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAVE_HGB = True
except Exception:
    HAVE_HGB = False

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAVE_SGKF = True
except Exception:
    from sklearn.model_selection import StratifiedKFold
    HAVE_SGKF = False


################################################################################
#
# Configuration
#
################################################################################

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, 'channel_table.csv')

TARGET_FS = 64            # resample every channel to 64 Hz
MAX_SECONDS = 28800       # use up to 8 h of recording
EPOCH_SEC = 30            # standard sleep epoch
RANDOM_STATE = 42

# --- Signal groups -----------------------------------------------------------
LEADS_TO_CHECK = {
    'eeg_f': ['f3-m2', 'f4-m1', 'f3', 'f4'],
    'eeg_c': ['c3-m2', 'c4-m1', 'c3', 'c4'],
    'eeg_o': ['o1-m2', 'o2-m1', 'o1', 'o2'],
    'eog':   ['e1-m2', 'e2-m1', 'e1', 'e2'],
    'chin':  ['chin1-chin2', 'chin', 'chin 1'],
    'leg':   ['lat', 'rat', 'lleg+', 'rleg+'],
    'ecg':   ['ecg', 'ekg'],
    'resp':  ['airflow', 'ptaf', 'abd', 'chest'],
    'spo2':  ['spo2', 'sao2'],
}
SIGNAL_GROUPS = list(LEADS_TO_CHECK)
NUM_FEATURES_PER_CHANNEL = 20
NUM_BASE = len(SIGNAL_GROUPS) * NUM_FEATURES_PER_CHANNEL              # 180

STAGE_REGIONS = ['eeg_f', 'eeg_c', 'eeg_o']
STAGE_CODES = [('n3', 1), ('n2', 2), ('rem', 4), ('wake', 5)]         # CAISR coding
STAGE_FEATS = ['delta', 'theta', 'alpha', 'sigma', 'beta',
               'dt_ratio', 'ta_ratio', 'se95', 'spec_ent']
NUM_STAGE = len(STAGE_REGIONS) * len(STAGE_CODES) * len(STAGE_FEATS)  # 108

MICRO_FEATS = ([f'{b}_{s}' for b in ['delta', 'theta', 'alpha', 'sigma', 'beta']
                for s in ['std', 'iqr']] + ['spindle_density', 'so_density'])
NUM_MICRO = len(MICRO_FEATS)                                          # 12

ALGO_FEATS = ['ahi', 'arousal_idx', 'limb_idx', 'w_pct', 'n1_pct', 'n2_pct',
              'n3_pct', 'rem_pct', 'sleep_eff', 'prob_w', 'prob_n3', 'prob_arous']
NUM_ALGO_FEATURES = len(ALGO_FEATS)                                   # 12

NUM_FEATURES = NUM_BASE + NUM_STAGE + NUM_MICRO + NUM_ALGO_FEATURES   # 312

BANDS = [('delta', 0.5, 4.0), ('theta', 4.0, 8.0), ('alpha', 8.0, 12.0),
         ('sigma', 12.0, 15.0), ('beta', 15.0, 30.0)]

BIPOLAR_CONFIGS = [
    ('f3-m2', 'f3', ['m2']), ('f4-m1', 'f4', ['m1']),
    ('c3-m2', 'c3', ['m2']), ('c4-m1', 'c4', ['m1']),
    ('o1-m2', 'o1', ['m2']), ('o2-m1', 'o2', ['m1']),
    ('e1-m2', 'e1', ['m2']), ('e2-m1', 'e2', ['m1']),
    ('chin1-chin2', 'chin 1', ['chin 2']),
    ('lat', 'lleg+', ['lleg-']), ('rat', 'rleg+', ['rleg-']),
]


def get_feature_names():
    names = [f'{g}_{i:02d}' for g in SIGNAL_GROUPS
             for i in range(NUM_FEATURES_PER_CHANNEL)]
    names += [f'{r}_{s}_{f}' for r in STAGE_REGIONS
              for s, _ in STAGE_CODES for f in STAGE_FEATS]
    names += [f'micro_{m}' for m in MICRO_FEATS]
    names += list(ALGO_FEATS)
    return names


################################################################################
#
# Required functions
#
################################################################################

def train_model(data_folder, model_folder, verbose, csv_path=DEFAULT_CSV_PATH):
    """Extract features from every training record, fit the classifier, and
    tune the decision threshold by patient-grouped cross-validation."""

    if verbose:
        print('Finding the Challenge data...')

    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_metadata_list = find_patients(patient_data_file)
    num_records = len(patient_metadata_list)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    if verbose:
        print(f'Found {num_records} records. Extracting features...')
        print(f'Feature vector: {NUM_FEATURES} '
              f'({NUM_BASE} base + {NUM_STAGE} stage-conditional + '
              f'{NUM_MICRO} micro + {NUM_ALGO_FEATURES} algorithmic)')

    features, labels, groups = [], [], []

    pbar = tqdm(range(num_records), desc='Extracting features', unit='rec',
                disable=not verbose)
    for i in pbar:
        try:
            record = patient_metadata_list[i]
            patient_id = record[HEADERS['bids_folder']]
            site_id = record[HEADERS['site_id']]
            session_id = record[HEADERS['session_id']]

            # --- Label ---
            label = load_diagnoses(os.path.join(data_folder, DEMOGRAPHICS_FILE),
                                   patient_id)
            if label != 0 and label != 1:
                continue

            feature_vec = extract_all_features(data_folder, patient_id, site_id,
                                               session_id, csv_path=csv_path,
                                               verbose=verbose)
            if feature_vec is None:
                continue

            features.append(feature_vec)
            labels.append(label)
            groups.append(str(patient_id))

        except Exception as e:
            try:
                tqdm.write(f'  !!! Error processing record {i + 1}: {e}')
            except Exception:
                print(f'  !!! Error processing record {i + 1}: {e}')
            continue

    try:
        pbar.close()
    except Exception:
        pass

    if len(labels) == 0:
        raise ValueError('No valid records found for training.')

    features = np.nan_to_num(np.asarray(features, dtype=np.float32),
                             nan=0.0, posinf=0.0, neginf=0.0)
    labels = np.asarray(labels, dtype=np.float32)
    groups = np.asarray(groups)

    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if verbose:
        print(f'\nTraining set: {len(labels)} records '
              f'({n_pos} positive, {n_neg} negative)')
        print(f'Feature matrix: {features.shape}, '
              f'non-constant columns: {int((features.std(axis=0) > 0).sum())}')

    pos_weight = float(n_neg / max(n_pos, 1))

    # ---- Tune the decision threshold by cross-validation ------------------
    threshold = 0.5
    if n_pos >= 10 and n_neg >= 10:
        if verbose:
            print('Tuning decision threshold by cross-validation...')
        try:
            threshold = tune_threshold(features, labels, groups, pos_weight,
                                       verbose=verbose)
        except Exception as e:
            if verbose:
                print(f'  Threshold tuning failed ({e}); using 0.5')
    elif verbose:
        print('Too few positives for threshold tuning; using 0.5')

    if verbose:
        print(f'Decision threshold: {threshold:.4f}')

    # ---- Fit the final model on all data ----------------------------------
    if verbose:
        print(f'Training final model '
              f'({"LightGBM" if HAVE_LGB else "HistGB" if HAVE_HGB else "GB"})...')

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    model = fit_classifier(make_classifier(pos_weight), features_scaled,
                           labels, pos_weight)

    os.makedirs(model_folder, exist_ok=True)
    save_model(model_folder, model, scaler, threshold)

    if verbose:
        print('Done. Model saved.')
        print()


def load_model(model_folder, verbose):
    """Load the trained model, scaler and threshold."""
    model_filename = os.path.join(model_folder, 'model.sav')
    data = joblib.load(model_filename)
    return data


def run_model(model, record, data_folder, verbose):
    """Run inference on a single record."""

    clf = model['model']
    scaler = model['scaler']
    threshold = float(model.get('threshold', 0.5))

    patient_id = record[HEADERS['bids_folder']]
    site_id = record[HEADERS['site_id']]
    session_id = record[HEADERS['session_id']]

    feature_vec = extract_all_features(data_folder, patient_id, site_id,
                                       session_id, verbose=False)
    if feature_vec is None:
        feature_vec = np.zeros(NUM_FEATURES, dtype=np.float32)

    feature_vec = np.nan_to_num(feature_vec.reshape(1, -1).astype(np.float32),
                                nan=0.0, posinf=0.0, neginf=0.0)
    feature_vec = scaler.transform(feature_vec)

    probability_output = float(clf.predict_proba(feature_vec)[0][1])
    binary_output = int(probability_output >= threshold)

    return binary_output, probability_output


################################################################################
#
# Classifier construction
#
################################################################################

def make_classifier(pos_weight):
    """LightGBM if available, otherwise sklearn fallbacks."""
    if HAVE_LGB:
        return lgb.LGBMClassifier(
            n_estimators=600, learning_rate=0.03, num_leaves=15, max_depth=4,
            min_child_samples=20, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=pos_weight, random_state=RANDOM_STATE,
            n_jobs=-1, verbose=-1)
    if HAVE_HGB:
        try:
            return HistGradientBoostingClassifier(
                max_iter=400, learning_rate=0.05, max_depth=4,
                min_samples_leaf=20, l2_regularization=1.0,
                class_weight='balanced', random_state=RANDOM_STATE)
        except TypeError:
            return HistGradientBoostingClassifier(
                max_iter=400, learning_rate=0.05, max_depth=4,
                min_samples_leaf=20, l2_regularization=1.0,
                random_state=RANDOM_STATE)
    return GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        min_samples_split=10, min_samples_leaf=5, subsample=0.8,
        max_features='sqrt', random_state=RANDOM_STATE)


def fit_classifier(clf, X, y, pos_weight):
    """LightGBM/HistGB handle imbalance internally; GB needs sample_weight."""
    if HAVE_LGB or (HAVE_HGB and not isinstance(clf, GradientBoostingClassifier)):
        clf.fit(X, y)
    else:
        clf.fit(X, y, sample_weight=np.where(y == 1, pos_weight, 1.0))
    return clf


def tune_threshold(X, y, groups, pos_weight, n_splits=5, verbose=False):
    """Cross-validated out-of-fold probabilities -> threshold maximising F1.

    Folds are grouped by patient so repeat sessions cannot leak."""
    n_pos = int(y.sum())
    n_splits = int(max(2, min(n_splits, n_pos)))

    if HAVE_SGKF:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                        random_state=RANDOM_STATE)
        split_iter = splitter.split(X, y, groups)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                   random_state=RANDOM_STATE)
        split_iter = splitter.split(X, y)

    oof = np.zeros(len(y), dtype=float)
    for tr, te in split_iter:
        if len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = fit_classifier(make_classifier(pos_weight), sc.transform(X[tr]),
                             y[tr], pos_weight)
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]

    if len(np.unique(y)) < 2:
        return 0.5

    prec, rec, thr = precision_recall_curve(y, oof)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    f1 = f1[:-1]
    if f1.size == 0 or not np.isfinite(f1).any():
        return 0.5

    best = int(np.nanargmax(f1))
    if verbose:
        try:
            from sklearn.metrics import (roc_auc_score, average_precision_score,
                                         f1_score as _f1)
            f1_default = _f1(y, (oof >= 0.5).astype(int), zero_division=0)
            print(f'  OOF AUROC={roc_auc_score(y, oof):.3f} '
                  f'AUPRC={average_precision_score(y, oof):.3f} '
                  f'(prevalence {y.mean():.3f})')
            print(f'  OOF F1={f1[best]:.3f} at thr={thr[best]:.3f} '
                  f'vs F1={f1_default:.3f} at the default 0.500')
        except Exception:
            pass
    return float(np.clip(thr[best], 1e-4, 1 - 1e-4))


################################################################################
#
# Feature extraction - top level
#
################################################################################

def extract_all_features(data_folder, patient_id, site_id, session_id,
                         csv_path=DEFAULT_CSV_PATH, verbose=False):
    """Assemble the full 312-dimensional feature vector for one record."""

    # --- Algorithmic annotations first: we need the hypnogram to condition
    #     the spectral features on sleep stage. ---
    algo_features = np.zeros(NUM_ALGO_FEATURES, dtype=np.float32)
    stage_1hz = None
    algo_file = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
                             site_id, f'{patient_id}_ses-{session_id}_caisr_annotations.edf')
    if os.path.exists(algo_file):
        try:
            algo_data, _ = load_signal_data(algo_file)
            algo_features = extract_algorithmic_annotations_features(algo_data)
            for key, val in algo_data.items():
                if str(key).lower().strip() == 'stage_caisr':
                    stage_1hz = np.asarray(val, dtype=float)
                    break
            del algo_data
        except Exception:
            pass

    # --- Physiological signals ---
    phys_file = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER, site_id,
                             f'{patient_id}_ses-{session_id}.edf')
    if not os.path.exists(phys_file):
        if verbose:
            try:
                tqdm.write(f'  ! Missing physiological data for {patient_id}.')
            except Exception:
                pass
        base = np.zeros(NUM_BASE, dtype=np.float32)
        stage_f = np.zeros(NUM_STAGE, dtype=np.float32)
        micro_f = np.zeros(NUM_MICRO, dtype=np.float32)
    else:
        try:
            physiological_data, physiological_fs = load_signal_data(phys_file)
            base, stage_f, micro_f = extract_physiological_features(
                physiological_data, physiological_fs, stage_1hz,
                csv_path=csv_path)
            del physiological_data
        except Exception as e:
            if verbose:
                try:
                    tqdm.write(f'  ! Signal extraction failed for {patient_id}: {e}')
                except Exception:
                    pass
            base = np.zeros(NUM_BASE, dtype=np.float32)
            stage_f = np.zeros(NUM_STAGE, dtype=np.float32)
            micro_f = np.zeros(NUM_MICRO, dtype=np.float32)

    def fit_len(a, n):
        a = np.asarray(a, dtype=np.float32).ravel()
        return a[:n] if a.size >= n else np.pad(a, (0, n - a.size))

    return np.hstack([fit_len(base, NUM_BASE),
                      fit_len(stage_f, NUM_STAGE),
                      fit_len(micro_f, NUM_MICRO),
                      fit_len(algo_features, NUM_ALGO_FEATURES)]).astype(np.float32)


################################################################################
#
# Feature extraction - physiological signals
#
################################################################################

def extract_physiological_features(physiological_data, physiological_fs,
                                   stage_1hz=None, csv_path=DEFAULT_CSV_PATH):
    """Returns (base 180, stage-conditional 108, micro 12)."""

    channels, fs_map = prepare_channels(physiological_data, physiological_fs,
                                        csv_path)

    base_list, stage_list = [], []
    micro_done = None

    for group, candidates in LEADS_TO_CHECK.items():
        sig, fs = None, None
        for candidate in candidates:
            if candidate in channels and channels[candidate] is not None \
                    and len(channels[candidate]) > 1:
                sig = channels[candidate]
                fs = fs_map.get(candidate)
                break

        if sig is None or not fs or fs <= 0:
            base_list.extend([0.0] * NUM_FEATURES_PER_CHANNEL)
            if group in STAGE_REGIONS:
                stage_list.extend([0.0] * (len(STAGE_CODES) * len(STAGE_FEATS)))
            continue

        resampled = resample_signal(sig, fs, TARGET_FS)
        max_samples = TARGET_FS * MAX_SECONDS
        if len(resampled) > max_samples:
            resampled = resampled[:max_samples]

        base_list.extend(compute_channel_features(resampled, TARGET_FS))

        if group in STAGE_REGIONS:
            abs_bp, rel_bp, psd, freqs = epoch_band_powers(resampled, TARGET_FS)
            if rel_bp is None:
                stage_list.extend([0.0] * (len(STAGE_CODES) * len(STAGE_FEATS)))
            else:
                ep_stage = stages_per_epoch(stage_1hz, len(rel_bp))
                stage_list.extend(
                    stage_conditional_features(rel_bp, psd, freqs, ep_stage))
                if group == 'eeg_c' and micro_done is None:
                    micro_done = micro_features(rel_bp, ep_stage)

        del resampled

    del channels
    micro = micro_done if micro_done is not None else [0.0] * NUM_MICRO
    return (np.array(base_list, dtype=np.float32),
            np.array(stage_list, dtype=np.float32),
            np.array(micro, dtype=np.float32))


def prepare_channels(physiological_data, physiological_fs, csv_path):
    """Standardize channel names (helper_code rules, with a built-in fallback)
    and build bipolar derivations."""

    original_labels = list(physiological_data.keys())
    rename_map, cols_to_drop = {}, set()

    try:
        rename_rules = load_rename_rules(os.path.abspath(csv_path))
        rename_map, cols_to_drop = standardize_channel_names_rename_only(
            original_labels, rename_rules)
    except Exception:
        rename_map, cols_to_drop = {}, set()

    channels, fs_map = {}, {}
    for old_label, data in physiological_data.items():
        if old_label in cols_to_drop:
            continue
        new_label = rename_map.get(old_label)
        if not new_label:
            new_label = fallback_channel_name(old_label)
        if new_label in channels:
            continue
        channels[new_label] = data
        fs_map[new_label] = float(physiological_fs.get(old_label, 0.0))

    for target, pos, neg_list in BIPOLAR_CONFIGS:
        if target in channels or pos not in channels:
            continue
        if not all(n in channels for n in neg_list):
            continue
        fs_values = [fs_map[c] for c in [pos] + neg_list]
        if len(set(fs_values)) > 1:
            continue
        try:
            ref = channels[neg_list[0]] if len(neg_list) == 1 \
                else tuple(channels[n] for n in neg_list)
            derived = derive_bipolar_signal(channels[pos], ref)
        except Exception:
            n = min(len(channels[pos]), len(channels[neg_list[0]]))
            derived = (np.asarray(channels[pos][:n], dtype=np.float32) -
                       np.asarray(channels[neg_list[0]][:n], dtype=np.float32)) \
                if n > 1 else None
        if derived is not None:
            channels[target] = derived
            fs_map[target] = fs_map[pos]

    return channels, fs_map


FALLBACK_ALIASES = {
    'a1': 'm1', 'a2': 'm2', 'loc': 'e1', 'roc': 'e2', 'eogl': 'e1', 'eogr': 'e2',
    'l eog': 'e1', 'r eog': 'e2', 'e1 m2': 'e1-m2', 'e2 m1': 'e2-m1',
    'ekg': 'ecg', 'ecgl': 'ecg', 'ekg1': 'ecg', 'ecg1': 'ecg', 'ecg i': 'ecg',
    'chin1': 'chin 1', 'chin2': 'chin 2', 'chin3': 'chin 3',
    'emg1': 'chin 1', 'emg2': 'chin 2',
    'lleg': 'lleg+', 'rleg': 'rleg+', 'leg l': 'lleg+', 'leg r': 'rleg+',
    'lat1': 'lleg+', 'lat2': 'lleg-', 'rat1': 'rleg+', 'rat2': 'rleg-',
    'sao2': 'spo2', 'spo2 1': 'spo2', 'osat': 'spo2',
    'flow': 'airflow', 'nasal pressure': 'ptaf', 'pt af': 'ptaf',
    'thorax': 'chest', 'thor': 'chest', 'abdomen': 'abd',
    'c3 m2': 'c3-m2', 'c4 m1': 'c4-m1', 'f3 m2': 'f3-m2', 'f4 m1': 'f4-m1',
    'o1 m2': 'o1-m2', 'o2 m1': 'o2-m1', 'chin1 chin2': 'chin1-chin2',
}


def fallback_channel_name(label):
    s = str(label).lower().strip()
    s = re.sub(r'^(eeg|eog|emg|ecg|ekg|resp)[\s\-_.]+', '', s)
    s = re.sub(r'[\s_]*-?\s*(ref|le|linked ears|avg)\s*$', '', s)
    s = s.replace('_', ' ').replace('.', ' ')
    s = re.sub(r'\s+', ' ', s).strip()
    return FALLBACK_ALIASES.get(s, s)


def resample_signal(signal, original_fs, target_fs):
    """Resample a 1D signal using linear interpolation."""
    signal = np.asarray(signal, dtype=np.float32)
    if original_fs <= 0 or abs(original_fs - target_fs) < 0.01:
        return signal

    original_length = len(signal)
    target_length = int(original_length * target_fs / original_fs)
    if target_length <= 0:
        return np.zeros(1, dtype=np.float32)

    x_original = np.linspace(0, 1, original_length, endpoint=False)
    x_target = np.linspace(0, 1, target_length, endpoint=False)
    return np.interp(x_target, x_original, signal).astype(np.float32)


def compute_channel_features(sig, fs):
    """20 features: 8 time-domain, 10 spectral, 2 percentile."""
    features = []
    n = len(sig)

    # ===== TIME DOMAIN (8) =====
    zcr_val = np.mean(np.diff(np.sign(sig)) != 0) if n > 1 else 0.0
    var_sig = np.var(sig)
    diff1 = np.diff(sig)
    var_d1 = np.var(diff1) if diff1.size else 0.0
    diff2 = np.diff(diff1) if diff1.size else np.array([])
    var_d2 = np.var(diff2) if diff2.size else 0.0
    mobility = np.sqrt(var_d1 / var_sig) if var_sig > 1e-12 else 0.0
    complexity = (np.sqrt(var_d2 / var_d1) / mobility) \
        if (var_d1 > 1e-12 and mobility > 1e-12) else 0.0

    features.extend([np.std(sig), np.mean(np.abs(sig)),
                     np.sqrt(np.mean(sig ** 2)), zcr_val,
                     skew(sig) if n > 2 else 0.0,
                     kurtosis(sig) if n > 2 else 0.0,
                     mobility, complexity])

    # ===== SPECTRAL (10) =====
    nperseg = int(min(4 * fs, n))
    if nperseg < 4:
        features.extend([0.0] * 10)
    else:
        freqs, psd = scipy_signal.welch(sig, fs=fs, nperseg=nperseg,
                                        noverlap=nperseg // 2)
        bp = {}
        for name, lo, hi in BANDS:
            mask = (freqs >= lo) & (freqs <= hi)
            bp[name] = _trapz(psd[mask], freqs[mask]) if np.any(mask) else 0.0

        dt_ratio = bp['delta'] / bp['theta'] if bp['theta'] > 1e-12 else 0.0
        ta_ratio = bp['theta'] / bp['alpha'] if bp['alpha'] > 1e-12 else 0.0

        total_power = _trapz(psd, freqs) if psd.size else 1e-12
        cumulative = np.cumsum(psd) * (freqs[1] - freqs[0]) \
            if freqs.size > 1 else np.array([0.0])
        cnorm = cumulative / (total_power + 1e-12)
        se50 = freqs[min(int(np.searchsorted(cnorm, 0.50)), freqs.size - 1)]
        se95 = freqs[min(int(np.searchsorted(cnorm, 0.95)), freqs.size - 1)]
        psd_norm = psd / (psd.sum() + 1e-12)

        features.extend([bp['delta'], bp['theta'], bp['alpha'], bp['sigma'],
                         bp['beta'], dt_ratio, ta_ratio, se50, se95,
                         float(entropy(psd_norm + 1e-12))])

    # ===== PERCENTILES (2) =====
    features.extend([np.percentile(sig, 5), np.percentile(sig, 95)])
    return features


################################################################################
#
# Feature extraction - stage-conditional spectra
#
################################################################################

def epoch_band_powers(sig, fs=TARGET_FS):
    """Split into 30 s epochs; return (abs_bp, rel_bp, psd, freqs) per epoch."""
    samples_per_epoch = int(EPOCH_SEC * fs)
    n_epochs = len(sig) // samples_per_epoch
    if n_epochs < 2:
        return None, None, None, None

    epochs = sig[:n_epochs * samples_per_epoch].reshape(n_epochs, samples_per_epoch)
    nperseg = int(4 * fs)
    freqs, psd = scipy_signal.welch(epochs, fs=fs, nperseg=nperseg,
                                    noverlap=nperseg // 2, axis=-1)

    abs_bp = np.zeros((n_epochs, len(BANDS)), dtype=np.float32)
    for j, (_name, lo, hi) in enumerate(BANDS):
        mask = (freqs >= lo) & (freqs <= hi)
        if np.any(mask):
            abs_bp[:, j] = _trapz(psd[:, mask], freqs[mask], axis=-1)

    rel_bp = abs_bp / (abs_bp.sum(axis=1, keepdims=True) + 1e-12)
    return abs_bp, rel_bp, psd, freqs


def stages_per_epoch(stage_1hz, n_epochs):
    """Majority CAISR stage code within each 30 s epoch."""
    if stage_1hz is None or len(stage_1hz) < EPOCH_SEC:
        return None
    s = np.asarray(stage_1hz, dtype=float)
    n = min(n_epochs, len(s) // EPOCH_SEC)
    if n < 1:
        return None

    blocks = s[:n * EPOCH_SEC].reshape(n, EPOCH_SEC)
    out = np.full(n_epochs, 9.0)
    for i in range(n):
        b = blocks[i]
        b = b[b < 9.0]
        if b.size:
            vals, counts = np.unique(b, return_counts=True)
            out[i] = vals[int(np.argmax(counts))]
    return out


def stage_conditional_features(rel_bp, psd, freqs, ep_stage):
    """9 features x 4 stages for one EEG region.

    Spectral power in N3 and in wake are different quantities; averaging them
    over the whole night washes out the slowing that the label depends on."""
    out = []
    for _stage_name, code in STAGE_CODES:
        if ep_stage is None:
            out.extend([0.0] * len(STAGE_FEATS))
            continue
        mask = ep_stage == code
        if mask.sum() < 3:
            out.extend([0.0] * len(STAGE_FEATS))
            continue

        rb = rel_bp[mask].mean(axis=0)
        dt = rb[0] / rb[1] if rb[1] > 1e-12 else 0.0
        ta = rb[1] / rb[2] if rb[2] > 1e-12 else 0.0

        p = psd[mask].mean(axis=0)
        cumulative = np.cumsum(p) * (freqs[1] - freqs[0]) \
            if freqs.size > 1 else np.array([0.0])
        cnorm = cumulative / (cumulative[-1] + 1e-12)
        se95 = freqs[min(int(np.searchsorted(cnorm, 0.95)), freqs.size - 1)]
        pn = p / (p.sum() + 1e-12)

        out.extend(list(rb) + [dt, ta, se95, float(entropy(pn + 1e-12))])
    return out


def micro_features(rel_bp, ep_stage):
    """Night-wide variability plus spindle / slow-oscillation density proxies."""
    if rel_bp is None or len(rel_bp) < 5:
        return [0.0] * NUM_MICRO

    if ep_stage is not None:
        sleep = (ep_stage >= 1) & (ep_stage <= 4)
        rb = rel_bp[sleep] if sleep.sum() >= 5 else rel_bp
    else:
        rb = rel_bp

    out = []
    for j in range(len(BANDS)):
        col = rb[:, j]
        out.extend([float(np.std(col)),
                    float(np.percentile(col, 75) - np.percentile(col, 25))])

    sigma, delta = rel_bp[:, 3], rel_bp[:, 0]
    if ep_stage is not None and (ep_stage == 2).sum() >= 5:
        out.append(float(np.mean(sigma[ep_stage == 2] > np.percentile(sigma, 75))))
    else:
        out.append(0.0)
    if ep_stage is not None and (ep_stage == 1).sum() >= 5:
        out.append(float(np.mean(delta[ep_stage == 1] > np.percentile(delta, 75))))
    else:
        out.append(0.0)
    return out


################################################################################
#
# Feature extraction - algorithmic annotations
#
################################################################################

def extract_algorithmic_annotations_features(algo_data):
    """Sleep architecture and event density features from CAISR outputs (12)."""
    if not algo_data:
        return np.zeros(NUM_ALGO_FEATURES, dtype=np.float32)

    lut = {str(k).lower().strip(): np.asarray(v, dtype=float)
           for k, v in algo_data.items()}

    def get(key):
        return lut.get(key, np.array([]))

    features = []
    total_hours = len(get('resp_caisr')) / 3600.0

    def count_discrete_events(key):
        sig = get(key)
        if sig.size == 0 or total_hours <= 0:
            return 0.0
        binary_sig = (sig > 0).astype(int)
        diff = np.diff(binary_sig, prepend=0)
        return float(np.count_nonzero(diff == 1) / total_hours)

    features.extend([
        count_discrete_events('resp_caisr'),      # AHI
        count_discrete_events('arousal_caisr'),   # arousal index
        count_discrete_events('limb_caisr'),      # limb movement index
    ])

    stages = get('stage_caisr')
    valid_stages = stages[stages < 9.0] if stages.size else np.array([])
    if valid_stages.size:
        features.extend([
            float(np.mean(valid_stages == 5)),    # W
            float(np.mean(valid_stages == 3)),    # N1
            float(np.mean(valid_stages == 2)),    # N2
            float(np.mean(valid_stages == 1)),    # N3
            float(np.mean(valid_stages == 4)),    # REM
            float(np.mean((valid_stages >= 1) & (valid_stages <= 4))),
        ])
    else:
        features.extend([0.0] * 6)

    def safe_mean(key):
        a = get(key)
        m = float(np.mean(a)) if a.size else 0.0
        return m if m < 1.0 else 0.0

    features.extend([safe_mean('caisr_prob_w'), safe_mean('caisr_prob_n3'),
                     safe_mean('caisr_prob_arous')])

    return np.array(features, dtype=np.float32)


################################################################################
#
# Save / load utilities
#
################################################################################

def save_model(model_folder, model, scaler, threshold=0.5):
    """Save model, scaler and tuned decision threshold."""
    d = {
        'model': model,
        'scaler': scaler,
        'threshold': float(threshold),
        'feature_names': get_feature_names(),
    }
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)
