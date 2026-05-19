import os
import glob
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# HELPER: Parse polarization + wavelength from channel name
# ============================================================
def parse_pol_and_wavelength(channel_name):
    """
    Example channel names:
      Unpolarized_A_Green
      Unpolarized_B_IR
      Co-Polarized_Red
      Cross-Polarized_Green

    Returns:
      (pol_group, wavelength) or (None, None)
    """
    if not isinstance(channel_name, str):
        return None, None

    parts = channel_name.split("_")
    if len(parts) < 2:
        return None, None

    wavelength = parts[-1]

    if channel_name.startswith("Unpolarized_A") or channel_name.startswith("Unpolarized_B"):
        pol_group = "Unpolarized"
    elif channel_name.startswith("Co-Polarized"):
        pol_group = "Co-Polarized"
    elif channel_name.startswith("Cross-Polarized"):
        pol_group = "Cross-Polarized"
    else:
        return None, None

    if wavelength not in ["Green", "Red", "IR"]:
        return None, None

    return pol_group, wavelength


# ============================================================
# Build one dataset from all sqi.csv files
# ============================================================
def build_sqi_ml_dataset(
    cleaned_root="FIU_Cleaned_Data",
    out_csv="ML_SQI_Dataset.csv",
    min_std=0.02
):
    """
    Searches recursively for all sqi.csv files under cleaned_root,
    combines them into one DataFrame, and creates a binary label:

      Label = 1 (Good) if Std >= min_std and Skewness > 0
      Label = 0 (Bad) otherwise

    Saves the final dataset to out_csv.
    """
    sqi_files = glob.glob(os.path.join(cleaned_root, "**", "sqi.csv"), recursive=True)

    if not sqi_files:
        print(f"No sqi.csv files found under {cleaned_root}")
        return None

    dfs = []
    for f in sqi_files:
        try:
            df = pd.read_csv(f)
            df["_source_file"] = f
            dfs.append(df)
        except Exception as e:
            print(f"Skipping {f}: {e}")

    if not dfs:
        print("No readable sqi.csv files found.")
        return None

    all_sqi = pd.concat(dfs, ignore_index=True)

    # Make sure required SQI features exist
    required_cols = ["Mean", "Std", "Skewness"]
    missing_cols = [c for c in required_cols if c not in all_sqi.columns]
    if missing_cols:
        raise ValueError(f"Missing required SQI columns: {missing_cols}")

    # Convert to numeric just in case
    for col in required_cols:
        all_sqi[col] = pd.to_numeric(all_sqi[col], errors="coerce")

    # Drop rows missing core SQI features
    all_sqi = all_sqi.dropna(subset=required_cols).copy()

    # Create label using your current SQI rule
    all_sqi["Label"] = np.where(
        (all_sqi["Std"] >= min_std) & (all_sqi["Skewness"] > 0),
        1,
        0
    )

    all_sqi["LabelName"] = all_sqi["Label"].map({1: "Good", 0: "Bad"})

    # Optional extra columns from Channel
    if "Channel" in all_sqi.columns:
        parsed = all_sqi["Channel"].apply(parse_pol_and_wavelength)
        all_sqi["PolGroup"] = parsed.apply(lambda x: x[0])
        all_sqi["Wavelength"] = parsed.apply(lambda x: x[1])

    # Nice column order
    preferred_order = [
        "Channel", "Hardware Channel",
        "WindowStartIdx", "WindowEndIdx",
        "WindowStartSec", "WindowEndSec",
        "Mean", "Std", "Skewness",
        "Label", "LabelName",
        "Day", "Experiment", "SkinTone", "Depth", "Speed",
        "Orientation", "Pol", "Session", "Condition",
        "PolGroup", "Wavelength",
        "_source_file"
    ]

    ordered_cols = [c for c in preferred_order if c in all_sqi.columns]
    leftover_cols = [c for c in all_sqi.columns if c not in ordered_cols]
    all_sqi = all_sqi[ordered_cols + leftover_cols]

    all_sqi.to_csv(out_csv, index=False)

    print(f"Done. Combined {len(sqi_files)} sqi.csv files.")
    print(f"Final dataset shape: {all_sqi.shape}")
    print(f"Saved dataset to: {out_csv}")
    print("\nLabel counts:")
    print(all_sqi["LabelName"].value_counts(dropna=False))

    return all_sqi


# ============================================================
# STEP 2: Train a simple ML classifier
# ============================================================
def train_sqi_classifier(
    dataset_csv="ML_SQI_Dataset.csv",
    model_out="sqi_random_forest.pkl"
):
    """
    Trains a simple Random Forest classifier using:
      Mean, Std, Skewness

    Saves the trained model with joblib.
    """
    df = pd.read_csv(dataset_csv)

    feature_cols = ["Mean", "Std", "Skewness"]
    target_col = "Label"

    # Make sure columns exist
    needed = feature_cols + [target_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    # Drop rows with missing values
    df = df.dropna(subset=needed).copy()

    X = df[feature_cols]
    y = df[target_col]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("\n==============================")
    print("MODEL RESULTS")
    print("==============================")
    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    # Feature importance
    importances = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    print("\nFeature Importances:")
    print(importances.to_string(index=False))

    # Save model
    joblib.dump(model, model_out)
    print(f"\nSaved trained model to: {model_out}")

    return model, importances


# ============================================================
# STEP 3: Example helper to predict on new SQI rows
# ============================================================
def predict_signal_quality(model_path, input_df):
    """
    Predict Good/Bad on a DataFrame that already contains:
      Mean, Std, Skewness
    """
    model = joblib.load(model_path)

    feature_cols = ["Mean", "Std", "Skewness"]
    missing = [c for c in feature_cols if c not in input_df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    X = input_df[feature_cols].copy()
    preds = model.predict(X)

    out = input_df.copy()
    out["PredictedLabel"] = preds
    out["PredictedLabelName"] = out["PredictedLabel"].map({1: "Good", 0: "Bad"})
    return out


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # Build combined ML dataset from all sqi.csv files
    ml_df = build_sqi_ml_dataset(
        cleaned_root="FIU_Cleaned_Data",
        out_csv="ML_SQI_Dataset.csv",
        min_std=0.02
    )

    #Train model 
    if ml_df is not None and len(ml_df) > 0:
        model, importances = train_sqi_classifier(
            dataset_csv="ML_SQI_Dataset.csv",
            model_out="sqi_random_forest.pkl"
        )