from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# File + output setup
# ---------------------------------------------------------
# This is the raw JSON file we want to clean up
INPUT_FILE = Path("trial1.json")

# This is where all of the cleaned CSVs + plots will go
OUTPUT_DIR = Path("Cleaned Trial1 Data Folder")

# These are the channels we care most about right now
# You can change these later depending on what LEDs/polarizations you want
ADC_CHANNELS_TO_PLOT = ["c2", "c4", "c5", "c12", "c13", "c15", "c18", "c19", "c21"]

# Window size for smoothing
# Larger window = smoother signal but less detail
SMOOTH_WINDOW = 10


# ---------------------------------------------------------
# Load the raw JSON file
# ---------------------------------------------------------
def load_json(file_path: Path) -> dict:

    # Make sure the file actually exists before trying to open it
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find {file_path}")

    # Open the JSON file and load everything into Python
    with open(file_path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------
# Convert raw records into a cleaner dataframe
# ---------------------------------------------------------
def records_to_dataframe(records: list) -> pd.DataFrame:

    # Turn nested JSON into a flat table
    # This makes it way easier to inspect and plot later
    df = pd.json_normalize(records)

    # Check if timestamps exist
    if "ts" in df.columns:

        # Convert timestamps into actual datetime objects
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")

        # Sort everything in time order just in case
        df = df.sort_values("ts").reset_index(drop=True)

        # Create a cleaner time axis that starts at 0 seconds
        # This makes plots WAY easier to read
        df["time_s"] = (
            df["ts"] - df["ts"].min()
        ).dt.total_seconds()

    # These columns are not super useful for visual inspection right now
    # so we remove them to clean up the tables
    columns_to_drop = ["delta_z", "tick"]

    df = df.drop(
        columns=[col for col in columns_to_drop if col in df.columns],
        errors="ignore"
    )

    return df


# ---------------------------------------------------------
# Save all cleaned streams into separate CSV files
# ---------------------------------------------------------
def save_clean_csvs(raw_data: dict, output_dir: Path) -> dict:

    # Create the output folder if it doesn't already exist
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_dfs = {}

    # Loop through each stream in the JSON
    # Example:
    # IMU
    # ADC
    # temperature
    # battery
    for stream_name, records in raw_data.items():

        # Make sure the stream actually contains data
        if isinstance(records, list) and len(records) > 0:

            # Clean the records
            df = records_to_dataframe(records)

            # Store dataframe in dictionary for later plotting
            clean_dfs[stream_name] = df

            # Save cleaned dataframe as CSV
            df.to_csv(
                output_dir / f"{stream_name}_clean.csv",
                index=False
            )

    return clean_dfs


# ---------------------------------------------------------
# Create a quick summary report
# ---------------------------------------------------------
def add_summary_report(clean_dfs: dict, output_dir: Path) -> None:

    summary_rows = []

    # Loop through each cleaned dataframe
    for stream_name, df in clean_dfs.items():

        # Build a quick summary of the dataset
        row = {
            "stream": stream_name,
            "rows": len(df),
            "columns": len(df.columns),

            # Start/end timestamps
            "start_time": df["ts"].min() if "ts" in df.columns else None,
            "end_time": df["ts"].max() if "ts" in df.columns else None,

            # Total recording length in seconds
            "duration_s": df["time_s"].max() if "time_s" in df.columns else None,
        }

        summary_rows.append(row)

    # Save summary as CSV
    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        output_dir / "summary_report.csv",
        index=False
    )


# ---------------------------------------------------------
# Plot IMU acceleration signals
# ---------------------------------------------------------
def plot_imu(df: pd.DataFrame, output_dir: Path) -> None:

    needed = {"time_s", "x", "y", "z"}

    # Skip plotting if required columns don't exist
    if not needed.issubset(df.columns):
        return

    plot_df = df.copy()

    # Compute total motion magnitude
    # This helps show overall movement regardless of direction
    plot_df["accel_magnitude"] = np.sqrt(
        plot_df["x"]**2 +
        plot_df["y"]**2 +
        plot_df["z"]**2
    )

    # Plot x/y/z separately
    plt.figure(figsize=(12, 6))

    plt.plot(plot_df["time_s"], plot_df["x"], label="x")
    plt.plot(plot_df["time_s"], plot_df["y"], label="y")
    plt.plot(plot_df["time_s"], plot_df["z"], label="z")

    plt.xlabel("Time (s)")
    plt.ylabel("IMU raw value")
    plt.title("IMU Acceleration Signals")

    plt.legend()
    plt.tight_layout()

    plt.savefig(output_dir / "imu_xyz.png", dpi=300)
    plt.close()

    # Plot overall movement magnitude
    plt.figure(figsize=(12, 5))

    plt.plot(
        plot_df["time_s"],
        plot_df["accel_magnitude"]
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration magnitude")
    plt.title("IMU Motion Magnitude")

    plt.tight_layout()

    plt.savefig(output_dir / "imu_magnitude.png", dpi=300)
    plt.close()


# ---------------------------------------------------------
# Plot temperature data
# ---------------------------------------------------------
def plot_temperature(df: pd.DataFrame, output_dir: Path) -> None:

    needed = {"time_s", "inner", "outer"}

    if not needed.issubset(df.columns):
        return

    plt.figure(figsize=(12, 5))

    # Plot internal + external temperature
    plt.plot(df["time_s"], df["inner"], label="inner")
    plt.plot(df["time_s"], df["outer"], label="outer")

    plt.xlabel("Time (s)")
    plt.ylabel("Temperature")
    plt.title("Temperature Over Time")

    plt.legend()
    plt.tight_layout()

    plt.savefig(output_dir / "temperature.png", dpi=300)
    plt.close()


# ---------------------------------------------------------
# Plot ADC channels
# ---------------------------------------------------------
def plot_adc(df: pd.DataFrame, output_dir: Path) -> None:

    # Need time axis to plot signals
    if "time_s" not in df.columns:
        return

    # Find all columns that look like ADC channels
    # Example:
    # c1
    # c2
    # c3
    adc_cols = [
        col for col in df.columns
        if col.startswith("c") and col[1:].isdigit()
    ]

    # Only plot selected channels for now
    selected_cols = [
        col for col in ADC_CHANNELS_TO_PLOT
        if col in adc_cols
    ]

    # Fallback if channels are missing
    if not selected_cols:
        selected_cols = adc_cols[:8]

    # -----------------------------------------------------
    # Raw ADC plot
    # -----------------------------------------------------
    plt.figure(figsize=(12, 6))

    for col in selected_cols:
        plt.plot(df["time_s"], df[col], label=col)

    plt.xlabel("Time (s)")
    plt.ylabel("Raw ADC value")
    plt.title("Selected ADC Channels: Raw Signal")

    plt.legend(ncol=3)
    plt.tight_layout()

    plt.savefig(output_dir / "adc_selected_raw.png", dpi=300)
    plt.close()

    # -----------------------------------------------------
    # Smoothed ADC plot
    # -----------------------------------------------------
    # Rolling average helps reduce noise so trends are easier to see
    smoothed = df[selected_cols].rolling(
        window=SMOOTH_WINDOW,
        center=True,
        min_periods=1
    ).mean()

    plt.figure(figsize=(12, 6))

    for col in selected_cols:
        plt.plot(df["time_s"], smoothed[col], label=col)

    plt.xlabel("Time (s)")
    plt.ylabel("Smoothed ADC value")

    plt.title(
        f"Selected ADC Channels: Smoothed Signal, Window = {SMOOTH_WINDOW}"
    )

    plt.legend(ncol=3)
    plt.tight_layout()

    plt.savefig(output_dir / "adc_selected_smoothed.png", dpi=300)
    plt.close()

    # -----------------------------------------------------
    # Normalized ADC plot
    # -----------------------------------------------------
    # Z-scoring helps compare channels even if they have
    # very different amplitudes
    normalized = df[selected_cols].copy()

    normalized = (
        normalized - normalized.mean()
    ) / normalized.std(ddof=0)

    plt.figure(figsize=(12, 6))

    for col in selected_cols:
        plt.plot(df["time_s"], normalized[col], label=col)

    plt.xlabel("Time (s)")
    plt.ylabel("Z-scored ADC value")

    plt.title(
        "Selected ADC Channels: Normalized for Visual Comparison"
    )

    plt.legend(ncol=3)
    plt.tight_layout()

    plt.savefig(output_dir / "adc_selected_normalized.png", dpi=300)
    plt.close()


# ---------------------------------------------------------
# Main script
# ---------------------------------------------------------
def main():

    # Load raw JSON
    raw_data = load_json(INPUT_FILE)

    # Clean + save all streams
    clean_dfs = save_clean_csvs(raw_data, OUTPUT_DIR)

    # Create quick overview summary
    add_summary_report(clean_dfs, OUTPUT_DIR)

    # Plot IMU data if it exists
    if "nirs4v1_imu_xl" in clean_dfs:
        plot_imu(clean_dfs["nirs4v1_imu_xl"], OUTPUT_DIR)

    # Plot temperature data if it exists
    if "nirs4v1_temp" in clean_dfs:
        plot_temperature(clean_dfs["nirs4v1_temp"], OUTPUT_DIR)

    # Plot ADC data if it exists
    if "nirs4v1_adc24_32" in clean_dfs:
        plot_adc(clean_dfs["nirs4v1_adc24_32"], OUTPUT_DIR)

    print(
        f"Done! Cleaned files and plots were saved in: "
        f"{OUTPUT_DIR.resolve()}"
    )

    print(
        "Start by opening summary_report.csv "
        "and then look at the PNG plots."
    )


# Run the script
if __name__ == "__main__":
    main()