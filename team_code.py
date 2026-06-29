
import joblib
import numpy as np
import os
import sys
from scipy import signal as scipy_signal
from scipy.stats import skew, kurtosis, entropy
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
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
# Demographics are available on the hidden validation/test sets, so they are
# safe to use at inference. The unofficial entry omitted them; the official
# template includes them and they are clinically informative for cognitive
# impairment. Set to False to reproduce the signal-only behaviour.
USE_DEMOGRAPHICS = True

# Human (expert) annotations are NOT available on the hidden sets. They must
# never enter the model feature vector, or train/inference dims will mismatch.
# Kept False by design; flip only if you implement a train-time-only scheme.
USE_HUMAN_ANNOTATIONS = False

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

################################################################################
#
# Required functions. Do NOT change the arguments of these functions.
#
################################################################################

def train_model(data_folder, model_folder, verbose, csv_path=DEFAULT_CSV_PATH):
    """Train a GradientBoosting model on extracted PSG (+ demographic) features."""

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

    pbar = tqdm(range(num_records), desc="Extracting Features", unit="rec", disable=not verbose)
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
                if verbose:
                    tqdm.write(f"  ! Missing physiological data for {patient_id}. Skipping...")
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

        except Exception as e:
            tqdm.write(f"  !!! Error processing record {i + 1} ({patient_id}): {e}")
            continue

    pbar.close()

    if len(labels) == 0:
        raise ValueError("No valid labeled records found for training.")

    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    if verbose:
        n_pos = int(labels.sum())
        n_neg = len(labels) - n_pos
        print(f'\nTraining set: {len(labels)} records ({n_pos} positive, {n_neg} negative)')
        print(f'Feature vector dimension: {features.shape[1]} (expected {_total_feature_dim()})')

    # ---- Normalize features ----
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # ---- Train GradientBoosting classifier ----
    if verbose:
        print('Training the model on the data...')

    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        max_features='sqrt',
        random_state=42,
        verbose=0,
    )
    model.fit(features, labels)

    # ---- Save model bundle ----
    os.makedirs(model_folder, exist_ok=True)
    save_model(model_folder, model, scaler)

    if verbose:
        print('Done.')
        print()


def load_model(model_folder, verbose):
    """Load the trained model bundle (classifier + scaler + metadata)."""
    model_filename = os.path.join(model_folder, 'model.sav')
    bundle = joblib.load(model_filename)
    return bundle


def run_model(model, record, data_folder, verbose):
    """Run inference on a single record. Returns (binary_label, probability)."""

    clf    = model['model']
    scaler = model['scaler']

    feature_vec = assemble_features(record, data_folder).reshape(1, -1)
    feature_vec = scaler.transform(feature_vec)

    binary_output = clf.predict(feature_vec)[0]
    probability_output = clf.predict_proba(feature_vec)[0][1]

    return binary_output, probability_output


################################################################################
#
# Shared feature assembly (used by BOTH train_model and run_model)
#
################################################################################

def assemble_features(record, data_folder, csv_path=DEFAULT_CSV_PATH):
    """
    Build the complete, fixed-length feature vector for one record.

    Order (must be identical at train and inference time):
        [ demographics? ] + physiological(140) + algorithmic(12)

    Every block falls back to a zero vector of the correct length when its
    source file is missing or unreadable, guaranteeing a constant dimension.
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
        except Exception:
            demographic_features = np.zeros(NUM_DEMOGRAPHIC_FEATURES, dtype=np.float32)
        blocks.append(np.asarray(demographic_features, dtype=np.float32))

    # ---- Physiological signal features ----
    phys_file = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER,
                             site_id, f"{patient_id}_ses-{session_id}.edf")
    if os.path.exists(phys_file):
        try:
            phys_data, phys_fs = load_signal_data(phys_file)
            phys_features = extract_physiological_features(phys_data, phys_fs, csv_path=csv_path)
            del phys_data
        except Exception:
            phys_features = np.zeros(NUM_PHYS_FEATURES, dtype=np.float32)
    else:
        phys_features = np.zeros(NUM_PHYS_FEATURES, dtype=np.float32)
    blocks.append(np.asarray(phys_features, dtype=np.float32))

    # ---- Algorithmic (CAISR) annotation features ----
    algo_file = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
                             site_id, f"{patient_id}_ses-{session_id}_caisr_annotations.edf")
    if os.path.exists(algo_file):
        try:
            algo_data, _ = load_signal_data(algo_file)
            algo_features = extract_algorithmic_annotations_features(algo_data)
        except Exception:
            algo_features = np.zeros(NUM_ALGO_FEATURES, dtype=np.float32)
    else:
        algo_features = np.zeros(NUM_ALGO_FEATURES, dtype=np.float32)
    blocks.append(np.asarray(algo_features, dtype=np.float32))

    feature_vec = np.hstack(blocks).astype(np.float32)
    return np.nan_to_num(feature_vec, nan=0.0, posinf=0.0, neginf=0.0)


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
    signal group, then compute 20 features per group.

    Per channel (20):
        Time-domain (8):  std, MAV, RMS, ZCR, skewness, kurtosis,
                          Hjorth mobility, Hjorth complexity
        Spectral (10):    delta, theta, alpha, sigma, beta power;
                          delta/theta ratio, theta/alpha ratio,
                          spectral edge 50%, spectral edge 95%, spectral entropy
        Percentile (2):   5th percentile, 95th percentile
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
            final_features.extend([0.0] * NUM_FEATURES_PER_CHANNEL)

    del processed_channels
    return np.array(final_features, dtype=np.float32)


def compute_channel_features(sig, fs):
    """Compute 20 features for a single channel signal (see docstring above)."""
    features = []
    n = len(sig)

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

        total_power = np.trapz(psd, freqs) if len(psd) > 0 else 1e-12
        cumulative = np.cumsum(psd) * (freqs[1] - freqs[0]) if len(freqs) > 1 else np.array([0])
        cumulative_norm = cumulative / (total_power + 1e-12)

        se50_idx = np.searchsorted(cumulative_norm, 0.50)
        se95_idx = np.searchsorted(cumulative_norm, 0.95)
        se50 = freqs[min(se50_idx, len(freqs) - 1)]
        se95 = freqs[min(se95_idx, len(freqs) - 1)]

        psd_norm = psd / (psd.sum() + 1e-12)
        spec_entropy = entropy(psd_norm + 1e-12)

        features.extend([delta_p, theta_p, alpha_p, sigma_p, beta_p,
                         dt_ratio, ta_ratio, se50, se95, spec_entropy])

    # ===== PERCENTILES (2) =====
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
    """Sleep architecture and event-density features from CAISR outputs."""
    if not algo_data:
        return np.zeros(NUM_ALGO_FEATURES, dtype=np.float32)

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
# These data are NOT available on the hidden validation/test sets, so they are
# deliberately excluded from the model feature vector (USE_HUMAN_ANNOTATIONS).
# Provided for experimentation (e.g. auxiliary targets / distillation).
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

def save_model(model_folder, model, scaler):
    """Persist classifier, scaler, and the config needed to reproduce features."""
    bundle = {
        'model': model,
        'scaler': scaler,
        'use_demographics': USE_DEMOGRAPHICS,
        'n_features': _total_feature_dim(),
    }
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(bundle, filename, protocol=0)
