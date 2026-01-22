import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, detrend  
from scipy.stats import skew    
import os
import glob
import gzip
import shutil

## add a Read me with start up details 
# flow diagram to capture how we inout data, steps to filtering window, add to read me, explaining outputs and signal processing

# ============================================================
# CONFIG (SQI)
# ============================================================
FS = 50                   # sampling frequency (Hz)
SQI_WINDOW_SEC = 8        # configurable 5–10 sec window
SQI_STEP_SEC = 1          # required 1 sec stride
BP_LOW_HZ = 0.5           # bandpass low cutoff
BP_HIGH_HZ = 8.0          # bandpass high cutoff
BP_ORDER = 4              # 4th order bandpass


# ---------------------------------
# CHANNEL MAP 
# ---------------------------------

CHANNEL_MAP = {
    "Unpolarized_A": {"Green": "c5",  "Red": "c2",  "IR": "c4"},
    "Unpolarized_B": {"Green": "c11", "Red": "c8",  "IR": "c10"},
    "Co-Polarized":  {"Green": "c13", "Red": "c12", "IR": "c15"},
    "Cross-Polarized": {"Green": "c19", "Red": "c18", "IR": "c21"},
}

# ---------------------------------
# SIGNAL PROCESSING HELPERS
# ---------------------------------

def unzip_file(filepath):
    """Unzip .json.gz -> .json once, return the .json path."""
    if filepath.endswith(".gz"):
        new_path = filepath[:-3]
        if not os.path.exists(new_path):  # unzip only if needed
            with gzip.open(filepath, "rb") as f_in, open(new_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        return new_path
    return filepath

# def lowpass_filter(signal, cutoff_hz, fs, order=4):
#     """Zero-phase low-pass filter to remove high-frequency noise."""
#     b, a = butter(order, cutoff_hz / (fs / 2), btype="low")
#     return filtfilt(b, a, signal)

# ---------------------------------
# FILTERING bandpass instead of lowpass)
# ---------------------------------
def bandpass_filter(signal, low_hz, high_hz, fs, order=4):
    """
    4th-order Butterworth bandpass filter (0.5–8 Hz), zero-phase.
    """
    signal = np.asarray(signal, dtype=float)
    nyq = fs / 2.0
    low = low_hz / nyq
    high = high_hz / nyq
    b, a = butter(order, [low, high], btype="bandpass")
    return filtfilt(b, a, signal)


# ---------------------------------
# WINDOWING + PREPROCESSING (NEW for SQI)
# ---------------------------------
def iter_windows(x, fs, window_sec=8, step_sec=1):
    """
    Sliding windows (5–10 sec) with 1-sec stride.
    Yields: (start_idx, end_idx, window_data)
    """
    x = np.asarray(x, dtype=float)
    win = int(window_sec * fs)
    step = int(step_sec * fs)

    if win <= 0 or step <= 0:
        raise ValueError("window_sec and step_sec must be > 0")

    if len(x) < win:
        return

    for start in range(0, len(x) - win + 1, step):
        end = start + win
        yield start, end, x[start:end]


def preprocess_window_ppg(win, fs, low_hz=0.5, high_hz=8.0, order=4):
    """
      1) bandpass filter
      2) zero-center (subtract DC per window)
      3) detrend each window
    """
    filtered = bandpass_filter(win, low_hz, high_hz, fs, order=order)
    centered = filtered - np.nanmean(filtered)         # zero-center (DC removal)
    detr = detrend(centered, type="linear")            # detrend each window
    return detr

def compute_sqi_windows(signal, condition_info, label, fs=50,
                        window_sec=8, step_sec=1,
                        low_hz=0.5, high_hz=8.0, order=4):
    """
    Compute SQI stats per window (includes Skewness).
    Returns a DataFrame (one row per window).
    """
    rows = []
    for start, end, win in iter_windows(signal, fs, window_sec, step_sec):
        proc = preprocess_window_ppg(win, fs, low_hz, high_hz, order)

        rows.append({
            "Channel": label,
            "Hardware Channel": channel_name_map(label),
            "WindowStartIdx": start,
            "WindowEndIdx": end,
            "WindowStartSec": start / fs,
            "WindowEndSec": end / fs,
            "Mean": float(np.nanmean(proc)),
            "Std": float(np.nanstd(proc)),
            "Skewness": float(skew(proc, nan_policy="omit")),
            **(condition_info if condition_info else {})
        })
    return pd.DataFrame(rows)

# ---------------------------------
# METRICS
# ---------------------------------

def compute_PI(peaks, troughs):
    """Compute perfusion index (AC/DC)*100, sign-corrected."""
    AC = peaks + troughs
    PI = -(AC / troughs) * 100
    return PI

def compute_baseline(signal):
    """Blackout offset baseline mean/std (no PI)."""
    signal = np.asarray(signal, dtype=float)
    return np.nanmean(signal), np.nanstd(signal)

def channel_name_map(clean_col_name):
    """Map cleaned column name (ex: 'Unpolarized_A_Green') -> 'C5'."""
    for pol, mapping in CHANNEL_MAP.items():
        for color, ch in mapping.items():
            if f"{pol}_{color}" == clean_col_name:
                return ch.upper()
    return "N/A"

def process_signal(signal, fs=50, prominence=25, distance=10, label="", condition_info=None):
    """Filter, detect peaks/troughs, compute PI mean/std."""
    #isolated = lowpass_filter(signal, cutoff, fs)

    isolated = bandpass_filter(signal, BP_LOW_HZ, BP_HIGH_HZ, fs, order=BP_ORDER)

    # Extract just the channel (e.g. C5, C10)
    graph_channel = "N/A"
    for pol, mapping in CHANNEL_MAP.items():
        for color, ch in mapping.items():
            if f"{pol}_{color}" == label:
                graph_channel = ch.upper()


    # # --- VISUALIZE THE FILTERED PPG SIGNAL ---

    parts = [
        condition_info.get("SkinTone") if condition_info else None,
        condition_info.get("Day") if condition_info else None,
        condition_info.get("Experiment") if condition_info else None,
        condition_info.get("Speed") if condition_info else None,
        condition_info.get("Depth") if condition_info else None,
        condition_info.get("Wavelength") if condition_info else None,
        condition_info.get("PolCondition") if condition_info else None,
        condition_info.get("Session") if condition_info else None,
        condition_info.get("Condition") if condition_info else None,
        graph_channel
    ]

    full_label = " | ".join([p for p in parts if p])

    plot_dir = os.path.join(
        "FIU_Plots",
        condition_info["Day"],
        condition_info.get("Experiment", condition_info.get("Condition"))
    )

    os.makedirs(plot_dir, exist_ok=True)

    # Safe filename
    plot_name = full_label.replace(" | ", "_").replace(" ", "")
    plot_path = os.path.join(plot_dir, f"{plot_name}.png")

    # Save plot (do NOT show)
    plt.figure(figsize=(8, 4))
    plt.plot(isolated, color="#7289A7", linewidth=1.2)
    plt.title(full_label)
    plt.xlabel("Sample index")
    plt.ylabel("Filtered amplitude")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    """
    The PPG signal was inverted along the y-axis because photodiodes produce 
    higher voltages when more light is reflected, whereas an increase in blood 
    volume actually reduces reflected light intensity. As described by Cennini 
    et al. (2010), this inversion aligns the waveform so that systolic peaks 
    correspond to physiological blood pulses, ensuring correct interpretation 
    of the perfusion index (PI).

    You invert the signal because:
    Photodiodes measure more light → higher voltage, but

    Blood pulses cause less light → lower voltage,
    so the raw PPG signal is upside-down relative to blood volume.
    Flipping it restores the correct orientation where heartbeats appear as upward peaks.

    """
    locs_troughs, _ = find_peaks(-isolated, prominence=prominence, distance=distance)
    locs_peaks, _ = find_peaks(isolated, prominence=prominence, distance=distance)

    troughs = -isolated[locs_troughs]
    peaks = isolated[locs_peaks]
    min_len = min(len(peaks), len(troughs))

    if min_len == 0:
        return np.nan, np.nan  # no valid peaks/troughs


    PI = compute_PI(peaks[:min_len], troughs[:min_len])
    return np.mean(PI), np.std(PI)
    

# ---------------------------------
# LOAD & CLEAN JSON DATA
# ---------------------------------
def load_and_clean_json(json_path, condition_info, mode="ppg"):
    """Load FIU experiment JSON file, flatten nested fields, and clean it."""

    json_path = unzip_file(json_path)

    # --- Load JSON file ---
    with open(json_path, "r") as f:
        data = json.load(f)

    # --- Detect nested data structure ---
    # FIU JSONs may store ADC data under various keys
    if "nirs4v1_adc24_32" in data:
        data_section = data["nirs4v1_adc24_32"]
    elif "nirs4v1_adc2" in data:
        data_section = data["nirs4v1_adc2"]
    elif "semi" in data:
        data_section = data["semi"]
    else:
        raise KeyError("Expected keys 'nirs4v1_adc24_32', 'nirs4v1_adc2', or 'semi' not found in JSON.")

    # --- Convert nested structure into DataFrame ---
    df = pd.DataFrame(data_section)
    
    print(f"\n Loaded JSON: {os.path.basename(json_path)}")
    print(f"   → {len(df.columns)} columns detected")
    print("   → Columns preview:", df.columns.tolist()[:10])

    time_col = df.get("ts", df.index)

    # --- Extract desired channels ---
    cleaned_data = {"time": time_col}
    for pol, mapping in CHANNEL_MAP.items():
        for color, channel in mapping.items():
            if channel in df.columns:
                cleaned_data[f"{pol}_{color}"] = df[channel]

    cleaned_df = pd.DataFrame(cleaned_data)
    
    """
    Adds extra identifying information 
    like skin tone, speed, and depth 
    as new columns in cleaned_df
    """
    for key, val in condition_info.items():
        cleaned_df[key] = val

    # --- Create folder structure ---
    day = condition_info.get("Day", "UnknownDay")

    if day == "Day_3":
        condition = condition_info.get("Condition", "Calibration")
        participant_folder = os.path.join("FIU_Cleaned_Data", "Day_3", "Calibration", condition)
        base_name = condition

    elif day == "Day_4":
        # Day 4 (Experiment 2 & 3) metadata comes from folder names, not Speed/Depth
        exp = condition_info.get("Experiment", "UnknownExperiment")
        wl = condition_info.get("Wavelength", "UnknownWavelength")
        skin = condition_info.get("SkinTone", "UnknownSkin")
        orient = str(condition_info.get("Orientation", "UnknownOrientation"))
        pol = condition_info.get("Pol", "UnknownPol")

        participant_folder = os.path.join(
            "FIU_Cleaned_Data", "Day_4", exp,
            wl, skin, f"{orient}deg", pol.replace(" ", "").replace(".", "")
        )
        base_name = f"{wl}_{skin}_{orient}_{pol}".replace(" ", "").replace(".", "")

    else:
        participant_folder = os.path.join(
            "FIU_Cleaned_Data",
            condition_info["Day"],
            condition_info["SkinTone"],
            condition_info["Experiment"]
        )
        base_name = f"{condition_info['SkinTone']}_{condition_info['Speed']}_{condition_info['Depth']}"

    out_path = os.path.join(participant_folder, f"{base_name}.csv")

    os.makedirs(participant_folder, exist_ok=True)

    # --------------------------------------------------
    # SKIP if this file was already processed
    # --------------------------------------------------
    if os.path.exists(out_path):
        print(f"Skipping (already cleaned): {out_path}")
        return

    #out_path = os.path.join(participant_folder, f"{base_name}.csv")
    cleaned_df.to_csv(out_path, index=False)
    print(f"Cleaned data saved: {out_path}")


    # --- Compute PI metrics for each polarization + wavelength ---
    summary_rows = []
    fs = 50  # sampling frequency (Hz)

    for col in cleaned_df.columns:
        if any(pol in col for pol in ["Unpolarized_A", "Unpolarized_B", "Co-Polarized", "Cross-Polarized"]):

            hw = channel_name_map(col)

            if mode == "baseline":
                mean_val, std_val = compute_baseline(cleaned_df[col].values)
                summary_rows.append({
                    "Participant": base_name,
                    "Channel": col,
                    "Hardware Channel": hw,
                    "Metric": "Baseline",
                    "Mean": mean_val,
                    "Std": std_val,
                    **condition_info
                })
            else:
                mean_PI, std_PI = process_signal(cleaned_df[col].values, fs, label=col, condition_info=condition_info)
                summary_rows.append({
                    "Participant": base_name,
                    "Channel": col,
                    "Hardware Channel": hw,
                    "Metric": "PI",
                    "Mean": mean_PI,
                    "Std": std_PI,
                    **condition_info
                })


    summary_df = pd.DataFrame(summary_rows)

    # --- Save or append summary CSV ---

    summary_path = os.path.join(participant_folder, "summary.csv")

    if os.path.exists(summary_path) and os.path.getsize(summary_path) > 0:
        existing = pd.read_csv(summary_path)
        # Remove duplicate participant entries before appending
        existing = existing[existing["Participant"] != base_name]
        updated = pd.concat([existing, summary_df], ignore_index=True)
        print(f"Updated (no duplicates): {summary_path}")
    else:
        # Always define 'updated' even when creating a new summary
        updated = summary_df.copy()
        print(f"Created new summary: {summary_path}")


    # --- Add hardware channel info to both table and CSV ---
    def get_channel_number(channel_name):
        for pol, mapping in CHANNEL_MAP.items():
            for color, ch_num in mapping.items():
                if f"{pol}_{color}" == channel_name:
                    return ch_num.upper()
        return "N/A"

    updated["Hardware Channel"] = updated["Channel"].apply(get_channel_number)

    # Reorder columns for cleaner CSV organization
    ordered_cols = [
        "Participant", "Channel", "Hardware Channel",
        "Metric", "Mean", "Std"
    ] + list(condition_info.keys())

    ordered_cols = [c for c in ordered_cols if c in updated.columns]

    # keep the ordered ones first, then any leftover columns after
    leftovers = [c for c in updated.columns if c not in ordered_cols]
    updated = updated[ordered_cols + leftovers]

    # Overwrite the summary CSV to include the hardware channel column
    updated.to_csv(summary_path, index=False)
    print(f"Summary file updated with hardware channels: {summary_path}")

    # --- Display table summary in terminal ---
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print("\n Summary Table:")
    print(updated.to_string(index=False))

    print(f"\n Summary updated for {base_name}\n")

# ---------------------------------
# Run JSON files Day 1 experiment
# ---------------------------------

data_folder = "Experiment 1 (Day 1)  copy"

# Include only unzipped .json files (ignore .gz and .zip)
json_files = [
    f for f in glob.glob(os.path.join(data_folder, "*.json"))
    if not f.endswith((".json.gz", ".json.zip"))
]

for json_path in json_files:
    filename = os.path.basename(json_path).replace(".json", "")

    # Detect if this is a repeated experiment (contains "(2)")
    if "(2)" in filename:
        experiment_num = "2"
        clean_name = filename.replace("(2)", "").strip()
    else:
        experiment_num = "1"
        clean_name = filename.strip()

    # Clean and split name
    clean_name = clean_name.replace("_", " ")
    parts = clean_name.split()

    if len(parts) >= 3:
        speed = parts[0].capitalize()
        skin = parts[1].capitalize()
        depth = parts[2] + "mm"

        condition_info = {
            "Day": "Day_1",
            "SkinTone": skin,
            "Speed": speed,
            "Depth": depth,
            "Experiment": f"Experiment_{experiment_num}"
        }


        print(f"\n Processing file: {filename} → {condition_info}")
        load_and_clean_json(json_path, condition_info)

# ---------------------------------
# Day 2 experiment
# ---------------------------------

day2_root = "Experiment 1 Complete  copy"
day2_folders = [d for d in glob.glob(os.path.join(day2_root, "*")) if os.path.isdir(d)]

for folder in day2_folders:

    folder_name = os.path.basename(folder)
    parts = folder_name.split()

    depth = parts[0] + "mm"
    skin = parts[1]
    speed = parts[2]

    condition_info = {
        "Day": "Day_2",
        "SkinTone": skin.capitalize(),
        "Speed": speed.capitalize(),
        "Depth": depth,
        "Experiment": "Experiment_2"
    }

    # unzip_file() will make sure it's only unzipped once.
    json_files = glob.glob(os.path.join(folder, "*.json*"))
    for json_path in json_files:
        load_and_clean_json(json_path, condition_info)



# ---------------------------------
# DAY 3 (Calibration -> Blackout Offset)
# ---------------------------------

"""
Day 3 data were collected for calibration and validation purposes. 
These trials were not intended to capture physiological PPG signals, 
but instead to assess baseline offsets, noise floors, and channel-dependent 
behavior under controlled blackout and polarization conditions. As a result,
peak-based metrics such as perfusion index were not the primary outcome for these trials.
"""

day3_root = "Experiment 2 Test (Day 3) copy"

# ---- Blackout Offset ----
blackout_folder = os.path.join(day3_root, "Calibration", "Blackout Offset")

# unzip once
gz_files = glob.glob(os.path.join(blackout_folder, "*.json.gz"))
for gz_path in gz_files:
    unzip_file(gz_path)

# process ONLY unzipped .json (ignores .json.gz)
json_files = [
    f for f in glob.glob(os.path.join(blackout_folder, "*.json*"))
    if f.endswith(".json")
]

condition_info = {
    "Day": "Day_3",
    "Condition": "Blackout_Offset"
}

for json_path in json_files:
    load_and_clean_json(json_path, condition_info, mode="baseline")

# ---- Red Light Pol. Switched (Sessions) ----
switched_root = os.path.join(day3_root, "Calibration", "Red Light Pol. Switched")
session_folders = [d for d in glob.glob(os.path.join(switched_root, "*")) if os.path.isdir(d)]

for sess in session_folders:
    sess_name = os.path.basename(sess)  # ex: "Session 1"

    # unzip once
    gz_files = glob.glob(os.path.join(sess, "*.json.gz"))
    for gz_path in gz_files:
        unzip_file(gz_path)

    # process only unzipped .json
    json_files = glob.glob(os.path.join(sess, "*.json"))

    condition_info = {
        "Day": "Day_3",
        "Condition": "RedLightPol_Switched",
        "Session": sess_name.replace(" ", "_")
    }

    for json_path in json_files:
        load_and_clean_json(json_path, condition_info, mode="baseline")


# ---------------------------------
# DAY 4 (Experiment 2 & 3)
# ---------------------------------

def parse_day4_folder(folder_name):
    """
    Examples:
      'Green Dark 0 Og. Pol'  -> Wavelength=Green, SkinTone=Dark, Orientation=0, Pol='Og. Pol'
      'IR Fair 180 Un. Pol'   -> Wavelength=IR,    SkinTone=Fair, Orientation=180, Pol='Un. Pol'
    """
    parts = folder_name.split()
    if len(parts) < 4:
        raise ValueError(f"Day 4 folder name not parseable: {folder_name}")

    wavelength = parts[0]               # Green / Red / IR
    skin = parts[1]                     # Dark / Light / Fair (you have both Light and Fair in different days)
    orientation = parts[2]              # 0 / 90 / 180 (string is fine)
    pol = " ".join(parts[3:])           # 'Og. Pol' / 'Flip. Pol' / 'Un. Pol'

    return wavelength, skin, orientation, pol


day4_root = "Experiment 2 & 3 (Day 4) copy"

def process_day4_experiment(exp_label):
    exp_path = os.path.join(day4_root, exp_label)
    if not os.path.exists(exp_path):
        print(f"Day 4: folder not found: {exp_path}")
        return

    condition_folders = [d for d in glob.glob(os.path.join(exp_path, "*")) if os.path.isdir(d)]

    for cond_folder in condition_folders:
        folder_name = os.path.basename(cond_folder)

        # Parse: Wavelength SkinTone Orientation Pol
        wl, skin, orient, pol = parse_day4_folder(folder_name)

        condition_info = {
            "Day": "Day_4",
            "Experiment": exp_label.replace(" ", "_"),  # 'Experiment_2' or 'Experiment_3'
            "Wavelength": wl,
            "SkinTone": skin,
            "Orientation": orient,
            "Pol": pol
        }

        # 1) unzip once
        gz_files = glob.glob(os.path.join(cond_folder, "*.json.gz"))
        for gz_path in gz_files:
            unzip_file(gz_path)

        # 2) process ONLY unzipped .json (ignores .json.gz)
        json_files = glob.glob(os.path.join(cond_folder, "*.json"))

        for json_path in json_files:
            load_and_clean_json(json_path, condition_info, mode="ppg")

# Run both Day 4 experiments
process_day4_experiment("Experiment 2")
process_day4_experiment("Experiment 3")



"""
plot a good 10 second window window for each file experiment -- COMPLETE DAY 1

make sure the graohs are inverted -- exoect notch to be on the right

find the skweness of a signal

want to have windowing in our signal, windowing will allow us to define where ppg signal is good enoguh to calculate prefusion index

tell us wherether we should include that part in the signal to include in the mean or not.
want to define morphalogy of the signal with skweness

based on skweness well know if we have a good signial if we have a pos skewness

_______________________________________________________________________________________

apply notch filter -- will help with viewing graphs

label trial instead of experiment 1 or 2

get peak and troughs for each subsequent experiment per day

Rutendo -- fft code implementation

"""