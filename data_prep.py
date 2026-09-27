import numpy as np
import pandas as pd

OVERSTRAIN_THRESHOLD = {"L": 11000, "M": 12000, "H": 13000}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["temp_diff"] = df["process_temp"] - df["air_temp"]
    df["power_w"] = df["torque"] * (df["rot_speed"] * 2 * np.pi / 60)

    df["wear_torque_product"] = df["tool_wear"] * df["torque"]
    df["overstrain_threshold"] = df["type"].map(OVERSTRAIN_THRESHOLD)
    df["overstrain_ratio"] = df["wear_torque_product"] / df["overstrain_threshold"]

    df["tool_wear_bucket"] = pd.cut(
        df["tool_wear"],
        bins=[-1, 50, 100, 150, 200, 250, 300],
        labels=["0-50", "51-100", "101-150", "151-200", "201-250", "251-300"],
    )

    return df
