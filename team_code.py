#!/usr/bin/env python

# PhysioNet Challenge 2026 - Screening for Cognitive Impairment During Sleep Studies
#
# Team: Rishabh Jha, University of Victoria MCV Lab
#
# ---------------------------------------------------------------------------
# Method summary
# ---------------------------------------------------------------------------
# The Challenge metric is the AGE-CONDITIONED AUROC: positives are only ever
# compared against negatives within +/- 2 years of age. Age is therefore
# differenced away inside every scored pair, and a model that predicts age well
# scores no better than chance. The quantity the metric rewards is deviation
# from what is normal FOR THAT AGE, so this entry is built around it:
#
#   1. Signal features (312)
#        180  time/spectral/Hjorth over 9 groups (EEG frontal/central/occipital,
#             EOG, chin, leg, ECG, respiratory, SpO2)
#        108  stage-conditional spectra - band powers computed separately inside
#             N3/N2/REM/Wake using the CAISR hypnogram, per EEG region. Uses
#             RELATIVE power, which is invariant to per-site amplitude scaling.
#         12  epoch-level variability + spindle / slow-oscillation density
#         12  CAISR event densities, sleep architecture, model confidence
#
#   2. Age residualisation (168)
#        Every EEG feature is regressed on age using a quadratic fitted on the
#        TRAINING NEGATIVES ONLY, and the standardised residual is added as a
#        new feature. This is the age-conditioned signal the metric scores.
#
#   3. Demographics (14)
#        Age, sex, BMI, race, ethnicity with explicit missingness flags. Age is
#        included for interactions; sex/BMI/race/ethnicity are NOT conditioned
#        out by the metric and contribute ordinary predictive signal.
#        SiteID is deliberately EXCLUDED: the validation and test sets come from
#        sites absent from training, so a site feature cannot generalise.
#
#   Total feature vector: 494
#
#   4. Time_to_Event weighting
#        A patient diagnosed 14 months after the PSG is a stronger positive than
#        one diagnosed at 5.8 years. Positives are weighted by proximity of
#        diagnosis to the sleep study.
#
#   5. Seed-bagged LightGBM, model selection by age-conditioned AUROC, with a
#      leave-one-site-out diagnostic that estimates cross-site generalisation
#      (the real deployment condition for this Challenge).
#
# No PyTorch dependency. Falls back to sklearn if LightGBM is unavailable.

################################################################################
#
# Libraries
#
################################################################################

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
    from sklearn.model_selection import StratifiedKFold
except Exception:
    StratifiedKFold = None


################################################################################
#
# Configuration
#
################################################################################

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, 'channel_table.csv')

TARGET_FS = 64            # resample all channels to 64 Hz
MAX_SECONDS = 28800       # use up to 8 h of recording
EPOCH_SEC = 30
RANDOM_STATE = 42

DELTA_YEARS = 2.0         # age window used by the Challenge metric
N_SEEDS = 5               # seed-bagged ensemble
CV_FOLDS = 5
TIME_WEIGHT_BETA = 0.5    # strength of Time_to_Event weighting (0 disables)
MAX_EVENT_DAYS = 6 * 365.0
THRESHOLD_OBJECTIVE = 'f1'   # 'f1' or 'reward'; binary output does not affect
                             # the age-conditioned AUROC that decides the winner

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
STAGE_CODES = [('n3', 1), ('n2', 2), ('rem', 4), ('wake', 5)]
STAGE_FEATS = ['delta', 'theta', 'alpha', 'sigma', 'beta',
               'dt_ratio', 'ta_ratio', 'se95', 'spec_ent']
NUM_STAGE = len(STAGE_REGIONS) * len(STAGE_CODES) * len(STAGE_FEATS)  # 108

MICRO_FEATS = ([f'{b}_{s}' for b in ['delta', 'theta', 'alpha', 'sigma', 'beta']
                for s in ['std', 'iqr']] + ['spindle_density', 'so_density'])
NUM_MICRO = len(MICRO_FEATS)                                          # 12

ALGO_FEATS = ['ahi', 'arousal_idx', 'limb_idx', 'w_pct', 'n1_pct', 'n2_pct',
              'n3_pct', 'rem_pct', 'sleep_eff', 'prob_w', 'prob_n3', 'prob_arous']
NUM_ALGO_FEATURES = len(ALGO_FEATS)                                   # 12

NUM_SIGNAL = NUM_BASE + NUM_STAGE + NUM_MICRO + NUM_ALGO_FEATURES     # 312

DEMO_FEATS = ['age', 'age_missing', 'sex_m', 'sex_f', 'bmi', 'bmi_missing',
              'race_asian', 'race_black', 'race_white', 'race_other',
              'race_unavailable', 'eth_hispanic', 'eth_not_hispanic',
              'eth_unavailable']
NUM_DEMO = len(DEMO_FEATS)                                            # 14

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


def get_signal_feature_names():
    names = [f'{g}_{i:02d}' for g in SIGNAL_GROUPS
             for i in range(NUM_FEATURES_PER_CHANNEL)]
    names += [f'{r}_{s}_{f}' for r in STAGE_REGIONS
              for s, _ in STAGE_CODES for f in STAGE_FEATS]
    names += [f'micro_{m}' for m in MICRO_FEATS]
    names += list(ALGO_FEATS)
    return names


def get_residual_indices():
    """Indices of the EEG-derived signal features that get age-residualised.

    All three EEG base blocks plus the whole stage-conditional block. These are
    the features with a strong, well-characterised age trend; residualising
    respiratory or SpO2 features adds noise without adding signal."""
    idx = []
    for gi, g in enumerate(SIGNAL_GROUPS):
        if g in STAGE_REGIONS:
            start = gi * NUM_FEATURES_PER_CHANNEL
            idx.extend(range(start, start + NUM_FEATURES_PER_CHANNEL))
    idx.extend(range(NUM_BASE, NUM_BASE + NUM_STAGE))
    return np.array(sorted(set(idx)), dtype=int)


RESIDUAL_IDX = get_residual_indices()                                  # 168
NUM_RESIDUAL = len(RESIDUAL_IDX)
NUM_FEATURES = NUM_SIGNAL + NUM_RESIDUAL + NUM_DEMO                    # 494


def get_feature_names():
    sig = get_signal_feature_names()
    return sig + [f'resid_{sig[i]}' for i in RESIDUAL_IDX] + \
        [f'demo_{d}' for d in DEMO_FEATS]


################################################################################
#
# Challenge metric: age-conditioned AUROC
#
################################################################################

def age_conditioned_auroc(labels, scores, ages, delta=DELTA_YEARS):
    """Pr(positive outranks negative | ages within delta). The Challenge metric.

    Ties contribute 0.5. Returns nan if no comparable pair exists."""
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    ages = np.asarray(ages, dtype=float)

    ok = np.isfinite(scores) & np.isin(labels, [0.0, 1.0])
    labels, scores, ages = labels[ok], scores[ok], ages[ok]
    ages = np.where(np.isfinite(ages), ages, np.nanmedian(ages)
                    if np.isfinite(ages).any() else 0.0)

    zp, ap = scores[labels == 1], ages[labels == 1]
    zn, an = scores[labels == 0], ages[labels == 0]
    if zp.size == 0 or zn.size == 0:
        return float('nan')

    num, den = 0.0, 0
    chunk = max(1, int(2e7 // max(zn.size, 1)))
    for s in range(0, zp.size, chunk):
        e = min(s + chunk, zp.size)
        comparable = np.abs(ap[s:e, None] - an[None, :]) <= delta
        if not comparable.any():
            continue
        diff = zp[s:e, None] - zn[None, :]
        num += float(((diff > 0) & comparable).sum())
        num += 0.5 * float(((diff == 0) & comparable).sum())
        den += int(comparable.sum())

    return num / den if den else float('nan')


def prevalence_reward(labels, binary_preds, ages, delta=DELTA_YEARS):
    """Secondary Challenge metric; used only if THRESHOLD_OBJECTIVE='reward'."""
    labels = np.asarray(labels, dtype=float)
    preds = np.asarray(binary_preds, dtype=float)
    ages = np.asarray(ages, dtype=float)
    good = np.isfinite(ages)
    fill = np.nanmedian(ages[good]) if good.any() else 0.0
    ages = np.where(good, ages, fill)

    p = np.empty(len(labels))
    for i, a in enumerate(ages):
        m = np.abs(ages - a) <= delta
        p[i] = labels[m].mean() if m.sum() else labels.mean()
    p = np.clip(p, 1e-6, 1 - 1e-6)

    r = np.zeros(len(labels))
    tp = (labels == 1) & (preds == 1)
    fp = (labels == 0) & (preds == 1)
    fn = (labels == 1) & (preds == 0)
    tn = (labels == 0) & (preds == 0)
    r[tp] = 1.0 / p[tp] - 1.0
    r[fp] = -1.0
    r[fn] = -1.0
    r[tn] = 1.0 / (1.0 - p[tn]) - 1.0
    return float(np.mean(r)) if r.size else float('nan')


################################################################################
#
# Age residualisation
#
################################################################################

class AgeResidualizer:
    """Standardised residual of each feature after removing its age trend.

    The trend is a quadratic fitted on TRAINING NEGATIVES ONLY, so 'expected for
    this age' means expected for a patient who does not go on to be diagnosed.
    A positive patient whose N3 delta power sits well below that curve produces
    a large negative residual - exactly what the age-conditioned metric rewards.
    Fitting on all patients instead would partly absorb the signal into the
    baseline."""

    def __init__(self, indices):
        self.indices = np.asarray(indices, dtype=int)
        self.coefs = None
        self.scales = None
        self.age_fill = 65.0
        self.age_lo = 20.0
        self.age_hi = 90.0

    @staticmethod
    def _transform(v):
        """Signed log compresses heavy-tailed power features so the quadratic
        fit is not dominated by a handful of high-amplitude recordings."""
        return np.sign(v) * np.log1p(np.abs(v))

    def fit(self, X, ages, labels):
        X = np.asarray(X, dtype=np.float64)
        ages = np.asarray(ages, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)

        finite_age = np.isfinite(ages)
        if finite_age.any():
            self.age_fill = float(np.median(ages[finite_age]))
            self.age_lo = float(np.percentile(ages[finite_age], 1))
            self.age_hi = float(np.percentile(ages[finite_age], 99))

        ref = finite_age & (labels == 0)
        if ref.sum() < 30:                      # not enough negatives: use all
            ref = finite_age
        if ref.sum() < 10:                      # no usable age at all
            self.coefs = np.zeros((len(self.indices), 3))
            self.scales = np.ones(len(self.indices))
            return self

        a_ref = np.clip(ages[ref], self.age_lo, self.age_hi)
        self.coefs = np.zeros((len(self.indices), 3))
        self.scales = np.ones(len(self.indices))

        for k, j in enumerate(self.indices):
            v = self._transform(X[ref, j])
            if not np.isfinite(v).all() or np.std(v) < 1e-12:
                continue
            try:
                c = np.polyfit(a_ref, v, 2)
            except Exception:
                continue
            resid = v - np.polyval(c, a_ref)
            # robust scale: MAD is unaffected by a few pathological recordings
            mad = np.median(np.abs(resid - np.median(resid)))
            scale = 1.4826 * mad if mad > 1e-12 else np.std(resid)
            self.coefs[k] = c
            self.scales[k] = scale if scale > 1e-12 else 1.0
        return self

    def transform(self, X, ages):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        ages = np.asarray(ages, dtype=np.float64).ravel()
        ages = np.where(np.isfinite(ages), ages, self.age_fill)
        ages = np.clip(ages, self.age_lo, self.age_hi)

        out = np.zeros((X.shape[0], len(self.indices)), dtype=np.float32)
        if self.coefs is None:
            return out
        for k, j in enumerate(self.indices):
            v = self._transform(X[:, j])
            pred = np.polyval(self.coefs[k], ages)
            out[:, k] = np.clip((v - pred) / self.scales[k], -12.0, 12.0)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


################################################################################
#
# Demographics
#
################################################################################

def _record_get(record, *candidates):
    """Fetch a metadata column robustly across HEADERS / raw column names."""
    for c in candidates:
        try:
            if isinstance(HEADERS, dict) and c in HEADERS and HEADERS[c] in record:
                return record[HEADERS[c]]
        except Exception:
            pass
        if c in record:
            return record[c]
    lower = {str(k).lower().replace(' ', '').replace('_', ''): k
             for k in record.keys()}
    for c in candidates:
        k = str(c).lower().replace(' ', '').replace('_', '')
        if k in lower:
            return record[lower[k]]
    return None


def _to_float(v):
    try:
        if v is None:
            return np.nan
        s = str(v).strip()
        if s == '' or s.lower() in ('nan', 'na', 'none', 'unavailable', 'unknown'):
            return np.nan
        return float(s)
    except Exception:
        return np.nan


def get_age(record):
    return _to_float(_record_get(record, 'age', 'Age'))


def get_time_to_event(record):
    return _to_float(_record_get(record, 'time_to_event', 'Time_to_Event'))


def extract_demographic_features(record):
    """14 demographic features. SiteID is intentionally omitted - the hidden
    validation and test sets come from sites that never appear in training."""
    age = get_age(record)
    bmi = _to_float(_record_get(record, 'bmi', 'BMI'))
    sex = str(_record_get(record, 'sex', 'Sex') or '').strip().lower()
    race = str(_record_get(record, 'race', 'Race') or '').strip().lower()
    eth = str(_record_get(record, 'ethnicity', 'Ethnicity') or '').strip().lower()

    f = []
    f.append(age if np.isfinite(age) else 65.0)
    f.append(0.0 if np.isfinite(age) else 1.0)
    f.append(1.0 if sex.startswith('m') else 0.0)
    f.append(1.0 if sex.startswith('f') else 0.0)
    f.append(bmi if np.isfinite(bmi) else 28.0)
    f.append(0.0 if np.isfinite(bmi) else 1.0)

    known_race = False
    for key in ('asian', 'black', 'white'):
        hit = 1.0 if key in race else 0.0
        known_race = known_race or bool(hit)
        f.append(hit)
    f.append(1.0 if ('other' in race) else 0.0)
    f.append(0.0 if (known_race or 'other' in race) else 1.0)

    is_hisp = ('hispanic' in race or 'hispanic' in eth) and 'not' not in eth
    not_hisp = 'not' in eth and 'hispanic' in eth
    f.append(1.0 if is_hisp else 0.0)
    f.append(1.0 if not_hisp else 0.0)
    f.append(0.0 if (is_hisp or not_hisp) else 1.0)

    return np.array(f, dtype=np.float32)


def sample_weights(labels, times_to_event, pos_weight, beta=TIME_WEIGHT_BETA):
    """Class weight, with positives scaled by how soon the diagnosis followed
    the sleep study. A diagnosis 14 months after the PSG reflects pathology far
    more likely to be visible in that PSG than one 5.8 years later."""
    labels = np.asarray(labels, dtype=float)
    t = np.asarray(times_to_event, dtype=float)
    w = np.ones(len(labels), dtype=float)
    w[labels == 1] = pos_weight
    if beta and beta > 0:
        pos = labels == 1
        valid = pos & np.isfinite(t) & (t > 0)
        if valid.any():
            prox = 1.0 - np.clip(t[valid] / MAX_EVENT_DAYS, 0.0, 1.0)
            w[valid] *= np.clip(1.0 + beta * (2.0 * prox - 1.0), 0.4, 2.0)
    return w


################################################################################
#
# Required functions
#
################################################################################

def train_model(data_folder, model_folder, verbose, csv_path=DEFAULT_CSV_PATH):
    """Extract features, fit the age model, train a seed-bagged ensemble, and
    tune the decision threshold under patient-level cross-validation."""

    if verbose:
        print('Finding the Challenge data...')

    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_metadata_list = find_patients(patient_data_file)
    num_records = len(patient_metadata_list)
    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    if verbose:
        print(f'Found {num_records} records.')
        print(f'Feature vector: {NUM_FEATURES} = {NUM_SIGNAL} signal + '
              f'{NUM_RESIDUAL} age-residual + {NUM_DEMO} demographic')
        print('Extracting features...')

    signal_feats, demo_feats = [], []
    labels, ages, sites, times = [], [], [], []

    pbar = tqdm(range(num_records), desc='Extracting features', unit='rec',
                disable=not verbose)
    for i in pbar:
        try:
            record = patient_metadata_list[i]
            patient_id = record[HEADERS['bids_folder']]
            site_id = record[HEADERS['site_id']]
            session_id = record[HEADERS['session_id']]

            label = load_diagnoses(os.path.join(data_folder, DEMOGRAPHICS_FILE),
                                   patient_id)
            if label != 0 and label != 1:
                continue

            sig = extract_signal_features(data_folder, patient_id, site_id,
                                          session_id, csv_path=csv_path,
                                          verbose=verbose)
            if sig is None:
                continue

            signal_feats.append(sig)
            demo_feats.append(extract_demographic_features(record))
            labels.append(label)
            ages.append(get_age(record))
            sites.append(str(site_id))
            times.append(get_time_to_event(record))

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

    S = np.nan_to_num(np.asarray(signal_feats, dtype=np.float32),
                      nan=0.0, posinf=0.0, neginf=0.0)
    D = np.nan_to_num(np.asarray(demo_feats, dtype=np.float32),
                      nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray(labels, dtype=np.float32)
    ages = np.asarray(ages, dtype=np.float64)
    sites = np.asarray(sites)
    times = np.asarray(times, dtype=np.float64)

    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if verbose:
        n_age = int(np.isfinite(ages).sum())
        print(f'\nTraining set: {len(y)} records ({n_pos} positive, {n_neg} negative)')
        print(f'  age available for {n_age}/{len(y)}; sites: '
              f'{dict(zip(*np.unique(sites, return_counts=True)))}')

    if n_pos == 0 or n_neg == 0:
        # The organizers stress-test entries by removing a class or changing
        # prevalence. Degrade to a constant predictor rather than crashing.
        if verbose:
            print('WARNING: only one class present. Saving a constant model.')
        residualizer = AgeResidualizer(RESIDUAL_IDX).fit(S, ages, y)
        X = np.hstack([S, residualizer.transform(S, ages), D]).astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        scaler = StandardScaler().fit(X)
        os.makedirs(model_folder, exist_ok=True)
        save_model(model_folder, [ConstantClassifier(float(y.mean()))],
                   scaler, residualizer, 0.5)
        if verbose:
            print('Done. Model saved.')
        return

    # ---- Age model, fitted on negatives only ------------------------------
    if verbose:
        print('Fitting age-residualisation model on training negatives...')
    residualizer = AgeResidualizer(RESIDUAL_IDX).fit(S, ages, y)
    X = np.hstack([S, residualizer.transform(S, ages), D]).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    pos_weight = float(n_neg / max(n_pos, 1))
    w = sample_weights(y, times, pos_weight)
    if verbose:
        n_t = int((np.isfinite(times) & (y == 1)).sum())
        print(f'  class weight {pos_weight:.1f}; Time_to_Event available for '
              f'{n_t}/{n_pos} positives')

    # ---- Cross-validation: age-conditioned AUROC and threshold ------------
    threshold = 0.5
    if n_pos >= 10 and n_neg >= 10 and StratifiedKFold is not None:
        try:
            threshold = cross_validate(X, y, ages, sites, w, pos_weight,
                                       verbose=verbose)
        except Exception as e:
            if verbose:
                print(f'  Cross-validation failed ({e}); threshold 0.5')
    elif verbose:
        print('Too few records for cross-validation; threshold 0.5')

    # ---- Final seed-bagged ensemble on all data ---------------------------
    if verbose:
        backend = 'LightGBM' if HAVE_LGB else ('HistGB' if HAVE_HGB else 'GB')
        print(f'Training final {N_SEEDS}-seed {backend} ensemble...')

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    models = [fit_classifier(make_classifier(pos_weight, seed), Xs, y, w)
              for seed in range(N_SEEDS)]

    os.makedirs(model_folder, exist_ok=True)
    save_model(model_folder, models, scaler, residualizer, threshold)

    if verbose:
        report_importance(models, verbose)
        print('Done. Model saved.')
        print()


def load_model(model_folder, verbose):
    """Load the trained ensemble, scaler, age model and threshold."""
    return joblib.load(os.path.join(model_folder, 'model.sav'))


def run_model(model, record, data_folder, verbose):
    """Run inference on a single record."""

    models = model['models']
    scaler = model['scaler']
    residualizer = model['residualizer']
    threshold = float(model.get('threshold', 0.5))

    patient_id = record[HEADERS['bids_folder']]
    site_id = record[HEADERS['site_id']]
    session_id = record[HEADERS['session_id']]

    sig = extract_signal_features(data_folder, patient_id, site_id, session_id,
                                  verbose=False)
    if sig is None:
        sig = np.zeros(NUM_SIGNAL, dtype=np.float32)
    sig = np.nan_to_num(sig.reshape(1, -1), nan=0.0, posinf=0.0, neginf=0.0)

    age = np.array([get_age(record)], dtype=np.float64)
    demo = extract_demographic_features(record).reshape(1, -1)

    X = np.hstack([sig, residualizer.transform(sig, age), demo]).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = scaler.transform(X)

    probability_output = float(np.mean([m.predict_proba(X)[0][1] for m in models]))
    binary_output = int(probability_output >= threshold)

    return binary_output, probability_output


################################################################################
#
# Model fitting and validation
#
################################################################################

class ConstantClassifier:
    """Predicts a fixed probability. Used only when the training set collapses
    to a single class, so that training degrades instead of crashing."""

    def __init__(self, p=0.5):
        self.p = float(np.clip(p, 0.0, 1.0))

    def predict_proba(self, X):
        n = np.atleast_2d(X).shape[0]
        return np.column_stack([np.full(n, 1.0 - self.p), np.full(n, self.p)])


def make_classifier(pos_weight, seed=RANDOM_STATE):
    if HAVE_LGB:
        return lgb.LGBMClassifier(
            n_estimators=700, learning_rate=0.025, num_leaves=15, max_depth=4,
            min_child_samples=20, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.5, reg_alpha=0.1, reg_lambda=1.0,
            random_state=seed, n_jobs=-1, verbose=-1)
    if HAVE_HGB:
        return HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_depth=4, min_samples_leaf=20,
            l2_regularization=1.0, random_state=seed)
    return GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        min_samples_split=10, min_samples_leaf=5, subsample=0.8,
        max_features='sqrt', random_state=seed)


def fit_classifier(clf, X, y, w):
    """Every backend here accepts sample_weight, which carries both the class
    balance and the Time_to_Event weighting."""
    try:
        clf.fit(X, y, sample_weight=w)
    except TypeError:
        clf.fit(X, y)
    return clf


def _predict_bag(models, X):
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def cross_validate(X, y, ages, sites, w, pos_weight, n_splits=CV_FOLDS,
                   verbose=False):
    """Stratified CV for the threshold, plus a leave-one-site-out diagnostic.

    The validation and test sets come from sites absent from training, so the
    leave-one-site-out number is the more honest estimate of what to expect."""

    n_splits = int(max(2, min(n_splits, int(y.sum()))))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=RANDOM_STATE)

    oof = np.zeros(len(y))
    for k, (tr, te) in enumerate(skf.split(X, y), 1):
        if len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        bag = [fit_classifier(make_classifier(pos_weight, s),
                              sc.transform(X[tr]), y[tr], w[tr])
               for s in range(min(N_SEEDS, 3))]
        oof[te] = _predict_bag(bag, sc.transform(X[te]))
        if verbose:
            ac = age_conditioned_auroc(y[te], oof[te], ages[te])
            print(f'  fold {k}: age-conditioned AUROC={ac:.3f}')

    if verbose:
        from sklearn.metrics import roc_auc_score, average_precision_score
        ac = age_conditioned_auroc(y, oof, ages)
        print(f'\n  OOF age-conditioned AUROC : {ac:.3f}   <-- Challenge metric')
        print(f'  OOF plain AUROC           : {roc_auc_score(y, oof):.3f}')
        print(f'  OOF AUPRC                 : {average_precision_score(y, oof):.3f}'
              f'   (prevalence {y.mean():.3f})')

        # Leave-one-site-out: the realistic cross-site estimate.
        uniq = [s for s in np.unique(sites)
                if (y[sites == s] == 1).sum() >= 3 and (y[sites == s] == 0).sum() >= 3]
        if len(uniq) >= 2:
            print('  Leave-one-site-out (validation/test come from unseen sites):')
            for s in uniq:
                te = sites == s
                tr = ~te
                if len(np.unique(y[tr])) < 2:
                    continue
                sc = StandardScaler().fit(X[tr])
                bag = [fit_classifier(make_classifier(pos_weight, sd),
                                      sc.transform(X[tr]), y[tr], w[tr])
                       for sd in range(min(N_SEEDS, 3))]
                p = _predict_bag(bag, sc.transform(X[te]))
                print(f'    held-out {s}: age-conditioned AUROC='
                      f'{age_conditioned_auroc(y[te], p, ages[te]):.3f} '
                      f'(n={int(te.sum())}, pos={int(y[te].sum())})')

    return pick_threshold(y, oof, ages, verbose=verbose)


def pick_threshold(y, prob, ages, verbose=False):
    """The binary output does not affect the age-conditioned AUROC, so this only
    moves the secondary metrics. At ~8% prevalence the default 0.5 leaves F1 on
    the floor, which is the failure this corrects."""
    if len(np.unique(y)) < 2:
        return 0.5

    if THRESHOLD_OBJECTIVE == 'reward':
        best_t, best_r = 0.5, -np.inf
        for t in np.unique(np.quantile(prob, np.linspace(0.01, 0.99, 100))):
            r = prevalence_reward(y, (prob >= t).astype(int), ages)
            if np.isfinite(r) and r > best_r:
                best_r, best_t = r, float(t)
        if verbose:
            print(f'  threshold {best_t:.3f} (prevalence reward {best_r:.3f})')
        return float(np.clip(best_t, 1e-4, 1 - 1e-4))

    prec, rec, thr = precision_recall_curve(y, prob)
    f1 = (2 * prec * rec / (prec + rec + 1e-12))[:-1]
    if f1.size == 0 or not np.isfinite(f1).any():
        return 0.5
    best = int(np.nanargmax(f1))
    if verbose:
        from sklearn.metrics import f1_score
        f1_default = f1_score(y, (prob >= 0.5).astype(int), zero_division=0)
        print(f'  threshold {thr[best]:.3f}: OOF F1={f1[best]:.3f} '
              f'(vs {f1_default:.3f} at the default 0.500)')
    return float(np.clip(thr[best], 1e-4, 1 - 1e-4))


def report_importance(models, verbose):
    try:
        imps = [m.feature_importances_ for m in models
                if hasattr(m, 'feature_importances_')]
        if not imps:
            return
        imp = np.mean(imps, axis=0)
        names = get_feature_names()
        order = np.argsort(imp)[::-1]
        print('\nTop 15 features:')
        for i in order[:15]:
            print(f'  {names[i]:32s} {imp[i]:.4f}')
        fam = {}
        for i, n in enumerate(names):
            key = 'residual' if n.startswith('resid_') else \
                  'demographic' if n.startswith('demo_') else \
                  'stage-conditional' if any(f'_{s}_' in n for s, _ in STAGE_CODES) else \
                  'micro' if n.startswith('micro_') else \
                  'algorithmic' if n in ALGO_FEATS else 'base signal'
            fam[key] = fam.get(key, 0.0) + imp[i]
        total = sum(fam.values()) or 1.0
        print('\nImportance by family:')
        for k, v in sorted(fam.items(), key=lambda kv: -kv[1]):
            print(f'  {k:20s} {100 * v / total:5.1f}%')
    except Exception:
        pass


################################################################################
#
# Feature extraction - top level
#
################################################################################

def extract_signal_features(data_folder, patient_id, site_id, session_id,
                            csv_path=DEFAULT_CSV_PATH, verbose=False):
    """The 312-dimensional signal feature vector for one record."""

    algo_features = np.zeros(NUM_ALGO_FEATURES, dtype=np.float32)
    stage_1hz = None
    algo_file = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
                             site_id,
                             f'{patient_id}_ses-{session_id}_caisr_annotations.edf')
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

    phys_file = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER, site_id,
                             f'{patient_id}_ses-{session_id}.edf')
    base = np.zeros(NUM_BASE, dtype=np.float32)
    stage_f = np.zeros(NUM_STAGE, dtype=np.float32)
    micro_f = np.zeros(NUM_MICRO, dtype=np.float32)

    if os.path.exists(phys_file):
        try:
            physiological_data, physiological_fs = load_signal_data(phys_file)
            base, stage_f, micro_f = extract_physiological_features(
                physiological_data, physiological_fs, stage_1hz, csv_path=csv_path)
            del physiological_data
        except Exception as e:
            if verbose:
                try:
                    tqdm.write(f'  ! Signal extraction failed for {patient_id}: {e}')
                except Exception:
                    pass
    elif verbose:
        try:
            tqdm.write(f'  ! Missing physiological data for {patient_id}.')
        except Exception:
            pass

    def fit_len(a, n):
        a = np.asarray(a, dtype=np.float32).ravel()
        return a[:n] if a.size >= n else np.pad(a, (0, n - a.size))

    return np.hstack([fit_len(base, NUM_BASE), fit_len(stage_f, NUM_STAGE),
                      fit_len(micro_f, NUM_MICRO),
                      fit_len(algo_features, NUM_ALGO_FEATURES)]).astype(np.float32)


################################################################################
#
# Feature extraction - physiological signals
#
################################################################################

def extract_physiological_features(physiological_data, physiological_fs,
                                   stage_1hz=None, csv_path=DEFAULT_CSV_PATH):
    channels, fs_map = prepare_channels(physiological_data, physiological_fs,
                                        csv_path)
    base_list, stage_list = [], []
    micro_done = None

    for group, candidates in LEADS_TO_CHECK.items():
        sig, fs = None, None
        for candidate in candidates:
            if candidate in channels and channels[candidate] is not None \
                    and len(channels[candidate]) > 1:
                sig, fs = channels[candidate], fs_map.get(candidate)
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
            _abs_bp, rel_bp, psd, freqs = epoch_band_powers(resampled, TARGET_FS)
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
    original_labels = list(physiological_data.keys())
    rename_map, cols_to_drop = {}, set()
    try:
        rules = load_rename_rules(os.path.abspath(csv_path))
        rename_map, cols_to_drop = standardize_channel_names_rename_only(
            original_labels, rules)
    except Exception:
        rename_map, cols_to_drop = {}, set()

    channels, fs_map = {}, {}
    for old_label, data in physiological_data.items():
        if old_label in cols_to_drop:
            continue
        new_label = rename_map.get(old_label) or fallback_channel_name(old_label)
        if new_label in channels:
            continue
        channels[new_label] = data
        fs_map[new_label] = float(physiological_fs.get(old_label, 0.0))

    for target, pos, neg_list in BIPOLAR_CONFIGS:
        if target in channels or pos not in channels:
            continue
        if not all(n in channels for n in neg_list):
            continue
        if len(set(fs_map[c] for c in [pos] + neg_list)) > 1:
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
    signal = np.asarray(signal, dtype=np.float32)
    if original_fs <= 0 or abs(original_fs - target_fs) < 0.01:
        return signal
    n0 = len(signal)
    n1 = int(n0 * target_fs / original_fs)
    if n1 <= 0:
        return np.zeros(1, dtype=np.float32)
    return np.interp(np.linspace(0, 1, n1, endpoint=False),
                     np.linspace(0, 1, n0, endpoint=False),
                     signal).astype(np.float32)


def compute_channel_features(sig, fs):
    """20 features: 8 time-domain, 10 spectral, 2 percentile."""
    features = []
    n = len(sig)

    zcr = np.mean(np.diff(np.sign(sig)) != 0) if n > 1 else 0.0
    var_sig = np.var(sig)
    d1 = np.diff(sig)
    v1 = np.var(d1) if d1.size else 0.0
    d2 = np.diff(d1) if d1.size else np.array([])
    v2 = np.var(d2) if d2.size else 0.0
    mobility = np.sqrt(v1 / var_sig) if var_sig > 1e-12 else 0.0
    complexity = (np.sqrt(v2 / v1) / mobility) \
        if (v1 > 1e-12 and mobility > 1e-12) else 0.0

    features.extend([np.std(sig), np.mean(np.abs(sig)),
                     np.sqrt(np.mean(sig ** 2)), zcr,
                     skew(sig) if n > 2 else 0.0,
                     kurtosis(sig) if n > 2 else 0.0, mobility, complexity])

    nperseg = int(min(4 * fs, n))
    if nperseg < 4:
        features.extend([0.0] * 10)
    else:
        freqs, psd = scipy_signal.welch(sig, fs=fs, nperseg=nperseg,
                                        noverlap=nperseg // 2)
        bp = {}
        for name, lo, hi in BANDS:
            m = (freqs >= lo) & (freqs <= hi)
            bp[name] = _trapz(psd[m], freqs[m]) if np.any(m) else 0.0
        dt = bp['delta'] / bp['theta'] if bp['theta'] > 1e-12 else 0.0
        ta = bp['theta'] / bp['alpha'] if bp['alpha'] > 1e-12 else 0.0
        total = _trapz(psd, freqs) if psd.size else 1e-12
        cum = np.cumsum(psd) * (freqs[1] - freqs[0]) if freqs.size > 1 \
            else np.array([0.0])
        cn = cum / (total + 1e-12)
        se50 = freqs[min(int(np.searchsorted(cn, 0.50)), freqs.size - 1)]
        se95 = freqs[min(int(np.searchsorted(cn, 0.95)), freqs.size - 1)]
        pn = psd / (psd.sum() + 1e-12)
        features.extend([bp['delta'], bp['theta'], bp['alpha'], bp['sigma'],
                         bp['beta'], dt, ta, se50, se95,
                         float(entropy(pn + 1e-12))])

    features.extend([np.percentile(sig, 5), np.percentile(sig, 95)])
    return features


def epoch_band_powers(sig, fs=TARGET_FS):
    """Per-30 s-epoch PSD and band powers, vectorised over epochs."""
    spe = int(EPOCH_SEC * fs)
    n_ep = len(sig) // spe
    if n_ep < 2:
        return None, None, None, None

    epochs = sig[:n_ep * spe].reshape(n_ep, spe)
    nperseg = int(4 * fs)
    freqs, psd = scipy_signal.welch(epochs, fs=fs, nperseg=nperseg,
                                    noverlap=nperseg // 2, axis=-1)
    abs_bp = np.zeros((n_ep, len(BANDS)), dtype=np.float32)
    for j, (_n, lo, hi) in enumerate(BANDS):
        m = (freqs >= lo) & (freqs <= hi)
        if np.any(m):
            abs_bp[:, j] = _trapz(psd[:, m], freqs[m], axis=-1)
    rel_bp = abs_bp / (abs_bp.sum(axis=1, keepdims=True) + 1e-12)
    return abs_bp, rel_bp, psd, freqs


def stages_per_epoch(stage_1hz, n_epochs):
    """Majority CAISR stage within each 30 s epoch (1=N3, 2=N2, 3=N1, 4=REM,
    5=Wake, 9=unavailable)."""
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
    """9 features x 4 stages. Delta power in N3 and in wake are different
    quantities; averaging across the night washes out the slowing."""
    out = []
    for _name, code in STAGE_CODES:
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
        cum = np.cumsum(p) * (freqs[1] - freqs[0]) if freqs.size > 1 \
            else np.array([0.0])
        cn = cum / (cum[-1] + 1e-12)
        se95 = freqs[min(int(np.searchsorted(cn, 0.95)), freqs.size - 1)]
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
    get = lambda k: lut.get(k, np.array([]))

    features = []
    total_hours = len(get('resp_caisr')) / 3600.0

    def count_discrete_events(key):
        sig = get(key)
        if sig.size == 0 or total_hours <= 0:
            return 0.0
        binary = (sig > 0).astype(int)
        return float(np.count_nonzero(np.diff(binary, prepend=0) == 1) / total_hours)

    features.extend([count_discrete_events('resp_caisr'),
                     count_discrete_events('arousal_caisr'),
                     count_discrete_events('limb_caisr')])

    stages = get('stage_caisr')
    valid = stages[stages < 9.0] if stages.size else np.array([])
    if valid.size:
        features.extend([float(np.mean(valid == 5)), float(np.mean(valid == 3)),
                         float(np.mean(valid == 2)), float(np.mean(valid == 1)),
                         float(np.mean(valid == 4)),
                         float(np.mean((valid >= 1) & (valid <= 4)))])
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

def save_model(model_folder, models, scaler, residualizer, threshold=0.5):
    d = {
        'models': models,
        'scaler': scaler,
        'residualizer': residualizer,
        'threshold': float(threshold),
        'feature_names': get_feature_names(),
    }
    joblib.dump(d, os.path.join(model_folder, 'model.sav'), protocol=0)
