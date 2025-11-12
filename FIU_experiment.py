import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import os
import glob

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

def lowpass_filter(signal, cutoff_hz, fs, order=4):
    """Zero-phase low-pass filter to remove high-frequency noise."""
    b, a = butter(order, cutoff_hz / (fs / 2), btype="low")
    return filtfilt(b, a, signal)

def compute_PI(peaks, troughs):
    """Compute perfusion index (AC/DC)*100, sign-corrected."""
    AC = peaks + troughs
    PI = -(AC / troughs) * 100
    return PI

def process_signal(signal, fs=50, cutoff=5, prominence=25, distance=10, label="", condition_info=None):
    """Filter, detect peaks/troughs, compute PI mean/std."""
    isolated = lowpass_filter(signal, cutoff, fs)

    # Extract just the channel (e.g. C5, C10)
    graph_channel = "N/A"
    for pol, mapping in CHANNEL_MAP.items():
        for color, ch in mapping.items():
            if f"{pol}_{color}" == label:
                graph_channel = ch.upper()


    # --- VISUALIZE THE FILTERED PPG SIGNAL ---


    if condition_info:
        full_label = f"{condition_info['SkinTone']} | {condition_info['Experiment']} | {condition_info['Speed']} | {condition_info['Depth']} | {graph_channel}"

    plt.figure(figsize=(8, 4))
    plt.plot(isolated, color='#7289A7', linewidth=1.2)
    plt.title(full_label)
    plt.xlabel("Sample index")
    plt.ylabel("Filtered amplitude")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

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
def load_and_clean_json(json_path, condition_info):
    """Load FIU experiment JSON file, flatten nested fields, and clean it."""

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
    participant_folder = os.path.join(
        "FIU_Cleaned_Data",
        condition_info["SkinTone"],
        condition_info["Experiment"])

    os.makedirs(participant_folder, exist_ok=True)

    # --- Save cleaned CSV ---
    base_name = f"{condition_info['SkinTone']}_{condition_info['Speed']}_{condition_info['Depth']}"
    out_path = os.path.join(participant_folder, f"{base_name}.csv")
    cleaned_df.to_csv(out_path, index=False)
    print(f"Cleaned data saved: {out_path}")


    # --- Compute PI metrics for each polarization + wavelength ---
    summary_rows = []
    fs = 50  # sampling frequency (Hz)

    for col in cleaned_df.columns:
        if any(pol in col for pol in ["Unpolarized_A", "Unpolarized_B", "Co-Polarized", "Cross-Polarized"]):
            mean_PI, std_PI = process_signal(cleaned_df[col].values, fs, label=col, condition_info=condition_info)
            summary_rows.append({
                "Participant": base_name,
                "Channel": col,
                "Mean PI": mean_PI,
                "Std PI": std_PI,
                "SkinTone": condition_info["SkinTone"],
                "Speed": condition_info["Speed"],
                "Depth": condition_info["Depth"],
                "Experiment": condition_info["Experiment"]
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
    "Mean PI", "Std PI", "SkinTone", "Speed", "Depth", "Experiment"
    ]

    updated = updated[ordered_cols]

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
            "SkinTone": skin,
            "Speed": speed,
            "Depth": depth,
            "Experiment": f"Experiment_{experiment_num}"
        }

        print(f"\n Processing file: {filename} → {condition_info}")
        load_and_clean_json(json_path, condition_info)



# add bulk data collection to extract skintone, speed, depth, manually via file naming convention

# we want to confirm if we have ppg 

# show visually to prove ppg

# Process it directly
#
