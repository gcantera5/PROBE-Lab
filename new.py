import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt

# ---------------------------------
# Helper Functions
# ---------------------------------

def lowpass_filter(signal, cutoff_hz, fs, order=4):
    """Apply a low-pass Butterworth filter (replicates MATLAB lowpass())."""
    b, a = butter(order, cutoff_hz / (fs / 2), btype='low')
    return filtfilt(b, a, signal)

def compute_PI(peaks, troughs):
    """Compute Perfusion Index as (AC/DC)*100, sign-corrected."""
    AC = peaks + troughs
    PI = -(AC / troughs) * 100
    return PI

# ---------------------------------
# Load Data
# ---------------------------------

# Replace with your actual file path (CSV or Excel)
data = pd.read_csv('/Users/guadalupecantera/Desktop/PROBE Lab/data/subject01.csv', header=None)

time = data.iloc[:, 1].values
co655 = data.iloc[:, 2].values
co940 = data.iloc[:, 3].values
cross655 = data.iloc[:, 4].values
cross940 = data.iloc[:, 5].values

fs = 50   # Sampling frequency (Hz)
cutoff = 5  # Low-pass cutoff frequency (Hz)

# ---------------------------------
# Process Each Channel
# ---------------------------------
"""
Apply a 5 Hz low-pass filter with 50 Hz sampling to remove high-frequency noise and keep cardiac content.
The Python helper uses scipy.signal.butter + filtfilt to replicate MATLAB's zero-phase lowpass()
"""

def process_channel(signal, fs, cutoff, prominence, distance, label, intervals):
    """Filter signal, find peaks/troughs, compute PI per interval, and plot."""
    isolated = lowpass_filter(signal, cutoff, fs)

    # Detect troughs (invert signal)
    troughs, locs_troughs = find_peaks(-isolated, prominence=prominence, distance=distance)
    # Detect peaks
    peaks, locs_peaks = find_peaks(isolated, prominence=prominence, distance=distance)

    results = []
    for idx_list, interval_label in intervals:
        idx_list = np.array(idx_list)
        sel_troughs = troughs[idx_list]
        sel_locs_troughs = locs_troughs[idx_list]
        sel_peaks = peaks[idx_list]
        PI = compute_PI(sel_peaks, sel_troughs)
        results.append(PI)

        # Plot QC waveform
        plt.figure(figsize=(10, 4))
        plt.plot(time, isolated, label=f"{label} Filtered")
        plt.plot(time[sel_locs_troughs], -sel_troughs, "x", label="Troughs")
        plt.plot(time[locs_peaks[idx_list]], sel_peaks, "o", label="Peaks")
        plt.title(f"{label}: {interval_label}")
        plt.xlabel("Time (s)")
        plt.ylabel("Filtered Signal")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Combine all intervals
    all_PI = np.concatenate(results)
    mean_PI = np.mean(all_PI)
    std_PI = np.std(all_PI)
    print(f"{label} — Mean PI: {mean_PI:.3f}, Std PI: {std_PI:.3f}")

    return results, mean_PI, std_PI


# Define intervals (index-based like MATLAB)
"""
same sample indices to pick stable 10-s signal windows.
np.arange replicates MATLAB's start:stop, and np.concatenate replicates [a; b]
"""
intervals = [
    (np.arange(14, 30), "Interval 1 (105–115 s)"),
    (np.concatenate([np.arange(94, 96), np.arange(98, 107)]), "Interval 2 (143–153 s)"),
    (np.arange(110, 126), "Interval 3 (180–190 s)")
]

# ---------------------------------
# Run for All Channels
# ---------------------------------

cross940_results, mean_940cr, std_940cr = process_channel(cross940, fs, cutoff, 25, 10, "940 Cross", intervals)
cross655_results, mean_655cr, std_655cr = process_channel(cross655, fs, cutoff, 18, 10, "655 Cross", intervals)
co940_results, mean_940co, std_940co = process_channel(co940, fs, cutoff, 40, 8, "940 Co", intervals)
co655_results, mean_655co, std_655co = process_channel(co655, fs, cutoff, 70, 11, "655 Co", intervals)

# ---------------------------------
# Plot Comparison (PI Across Channels)
# ---------------------------------

plt.figure(figsize=(8, 6))
plt.step(time[intervals[2][0]], np.concatenate(cross940_results), where='mid', label="940 Cross PI", color='r')
plt.step(time[intervals[2][0]], np.concatenate(cross655_results), where='mid', label="655 Cross PI", color='b')
plt.step(time[intervals[2][0]], np.concatenate(co940_results), where='mid', label="940 Co PI", color='m')
plt.step(time[intervals[2][0]], np.concatenate(co655_results), where='mid', label="655 Co PI", color='k')

plt.xlim([180, 190])
plt.ylim([0, 4])
plt.xlabel("Time (s)")
plt.ylabel("Perfusion Index")
plt.title("Perfusion Index Across Channels (Interval 3 Example)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ---------------------------------
# Signal to Noise Metic
# ---------------------------------

def pick_good_segments(signal, fs, sqi_func, win_sec=10, hop_sec=2, n_segments=3):
    """
    Scan the signal and return indices of n_segments best 10s windows by SQI.
    signal : 1D numpy array
    fs : sampling frequency (Hz)
    sqi_func : function that takes (segment, fs) -> score
    win_sec: window size
    hop_sec: how far to move the window before computing the next segments quality.
    n_segments : number of good windows to select
    """
    win = int(win_sec * fs)
    hop = int(hop_sec * fs)
    N = len(signal)
    scores = []
    
    # Compute SQI for each window
    for start in range(0, N - win + 1, hop):
        seg = signal[start:start + win]
        score = sqi_func(seg, fs)
        scores.append((start, start + win, score))
    
    # Sort windows by descending quality
    # Sorts all the windows so the best ones (highest SQI) come first.
    scores.sort(key=lambda x: x[2], reverse=True)
    
    # Goes through the ranked list from best → worst.

    selected = []
    for start, end, score in scores:
        # ensure chosen windows don't overlap with existing ones
        if all(abs(start - s[0]) > win for s in selected):
            selected.append((start, end, score))
        # Stops once 3 good segments are picked.
        if len(selected) >= n_segments:
            break
    """
    returns three separate 10-second windows spread 
    through the 2 min recording, not three overlapping ones
    """
    return selected