"""
Northstar Forecasting Lead Script
--------------------------------
Purpose:
- Read the validated Northstar dataset produced by the Python Pipeline Lead,\n  with fallback to the Data Quality cleaned dataset.
- Forecast 28 days: 2022-06-03 to 2022-06-30.
- Use data only through 2022-06-02 for model training.
- Compare a 28-day Seasonal Naive baseline against a Random Forest model.
- Report MAE and WAPE.
- Save prediction CSVs and Forecast-vs-Actual charts.

Leakage control:
The Random Forest uses calendar features plus lag/rolling features built only from
values available before each prediction date. During the 28-day holdout, later lag
features use earlier model predictions rather than the actual holdout target values.
"""

from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# Prefer the validated output from the Python Pipeline Lead.
# If it is not present, fall back to the Data Quality cleaned dataset.
PIPELINE_FILE = BASE_DIR / "northstar_clean.csv"
DATA_QUALITY_FILE = BASE_DIR / "northstar_daily_sku_market_cleaned.csv"

if PIPELINE_FILE.exists():
    DATA_FILE = PIPELINE_FILE
    DATA_SOURCE = "Python Pipeline Lead output (northstar_clean.csv)"
elif DATA_QUALITY_FILE.exists():
    DATA_FILE = DATA_QUALITY_FILE
    DATA_SOURCE = "Data Quality cleaned dataset (northstar_daily_sku_market_cleaned.csv)"
else:
    DATA_FILE = None
    DATA_SOURCE = None

OUTPUT_DIR = BASE_DIR / "forecast_outputs"

TRAIN_END = pd.Timestamp("2022-06-02")
HOLDOUT_START = pd.Timestamp("2022-06-03")
HOLDOUT_END = pd.Timestamp("2022-06-30")

# Three comparable series: same market, channel and category, different routes.
SELECTED_SERIES = [
    {"sku": "NS-003", "market": "France", "channel": "E-commerce"},
    {"sku": "NS-033", "market": "France", "channel": "E-commerce"},
    {"sku": "NS-048", "market": "France", "channel": "E-commerce"},
]

RANDOM_STATE = 42


# -----------------------------------------------------------------------------
# METRICS
# -----------------------------------------------------------------------------
def wape(actual, forecast):
    """Weighted Absolute Percentage Error, expressed as a percentage."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    denominator = np.abs(actual).sum()
    if denominator == 0:
        return np.nan
    return np.abs(actual - forecast).sum() / denominator * 100.0


# -----------------------------------------------------------------------------
# FEATURE ENGINEERING
# -----------------------------------------------------------------------------
def build_features(date, history, first_date):
    """
    Create leakage-safe features for one date.

    history is a dictionary: {Timestamp: known_or_predicted_units_sold}.
    No value from the current/future date is read.
    """

    def lag(days):
        return history.get(date - pd.Timedelta(days=days), np.nan)

    lag_values = [lag(1), lag(7), lag(14), lag(28)]
    previous_7 = [lag(i) for i in range(1, 8)]
    previous_28 = [lag(i) for i in range(1, 29)]

    required = lag_values + previous_7 + previous_28
    if any(pd.isna(value) for value in required):
        return None

    day_of_week = date.dayofweek
    day_of_year = date.dayofyear

    return [
        (date - first_date).days,                      # trend
        day_of_week,
        date.month,
        math.sin(2 * math.pi * day_of_week / 7),      # weekly seasonality
        math.cos(2 * math.pi * day_of_week / 7),
        math.sin(2 * math.pi * day_of_year / 365.25), # annual seasonality
        math.cos(2 * math.pi * day_of_year / 365.25),
        lag_values[0],   # lag 1
        lag_values[1],   # lag 7
        lag_values[2],   # lag 14
        lag_values[3],   # lag 28
        float(np.mean(previous_7)),
        float(np.mean(previous_28)),
        float(np.std(previous_7)),
        float(np.std(previous_28)),
    ]


FEATURE_NAMES = [
    "trend_day",
    "day_of_week",
    "month",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_28",
]


# -----------------------------------------------------------------------------
# FORECAST ONE SERIES
# -----------------------------------------------------------------------------
def forecast_series(series_df):
    series_df = series_df.sort_values("date").copy()

    if series_df.empty:
        raise ValueError("Selected series has no rows.")

    # Confirm there is exactly one observation per date in this series.
    duplicate_dates = int(series_df["date"].duplicated().sum())
    if duplicate_dates:
        raise ValueError(f"Series contains {duplicate_dates} duplicate date(s).")

    first_date = series_df["date"].min()
    actual_history = dict(
        zip(series_df["date"], series_df["units_sold"].astype(float))
    )

    # Build training rows. Each target uses only earlier values as features.
    X_train = []
    y_train = []

    training_rows = series_df[series_df["date"] <= TRAIN_END]
    for row in training_rows.itertuples(index=False):
        features = build_features(row.date, actual_history, first_date)
        if features is not None:
            X_train.append(features)
            y_train.append(float(row.units_sold))

    if not X_train:
        raise ValueError("Not enough historical data to create lag features.")

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Keep only information available at the forecast origin.
    recursive_history = {
        date: value
        for date, value in actual_history.items()
        if date <= TRAIN_END
    }

    rows = []
    forecast_dates = pd.date_range(HOLDOUT_START, HOLDOUT_END, freq="D")

    for date in forecast_dates:
        if date not in actual_history:
            raise ValueError(f"Holdout actual is missing for {date.date()}.")

        features = build_features(date, recursive_history, first_date)
        if features is None:
            raise ValueError(f"Cannot construct forecast features for {date.date()}.")

        rf_forecast = max(0.0, float(model.predict([features])[0]))

        # Recursive multi-step forecasting: save prediction as history for later days.
        recursive_history[date] = rf_forecast

        seasonal_source_date = date - pd.Timedelta(days=28)
        seasonal_naive = actual_history.get(seasonal_source_date)
        if seasonal_naive is None:
            raise ValueError(
                f"Missing 28-day baseline source for {date.date()}: "
                f"{seasonal_source_date.date()}"
            )

        rows.append(
            {
                "date": date,
                "actual_units_sold": float(actual_history[date]),
                "seasonal_naive_forecast": float(seasonal_naive),
                "random_forest_forecast": rf_forecast,
            }
        )

    forecast_df = pd.DataFrame(rows)
    return forecast_df, model, len(X_train)


# -----------------------------------------------------------------------------
# MAIN PROGRAM
# -----------------------------------------------------------------------------
def main():
    if DATA_FILE is None:
        raise FileNotFoundError(
            "Could not find either northstar_clean.csv or "
            "northstar_daily_sku_market_cleaned.csv. "
            "Put one of these files in the same folder as forecasting.py."
        )

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load the team-cleaned/pipeline dataset.
    df = pd.read_csv(DATA_FILE)

    required_columns = {
        "date", "market", "channel", "sku", "category",
        "route_exposure", "units_sold"
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")

    # Forecast-specific preparation:
    # The Pipeline Lead intentionally keeps unparseable dates as missing.
    # Forecasting needs valid chronological dates and a valid target value.
    rows_received = len(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce")

    invalid_dates = int(df["date"].isna().sum())
    missing_targets = int(df["units_sold"].isna().sum())

    df = df.dropna(subset=["date", "units_sold"]).copy()
    df = df.sort_values(["sku", "market", "channel", "date"])

    print(f"Data source: {DATA_SOURCE}")
    print(f"Rows received: {rows_received:,}")
    print(f"Rows excluded for invalid/missing date: {invalid_dates:,}")
    print(f"Rows excluded for missing units_sold: {missing_targets:,}")
    print(f"Rows usable for forecasting: {len(df):,}")
    print()

    all_predictions = []
    metric_rows = []
    importance_rows = []

    for spec in SELECTED_SERIES:
        mask = (
            (df["sku"] == spec["sku"])
            & (df["market"] == spec["market"])
            & (df["channel"] == spec["channel"])
        )
        series_df = df.loc[mask].copy()

        if series_df.empty:
            raise ValueError(f"Series not found: {spec}")

        category = str(series_df.iloc[0]["category"])
        route = str(series_df.iloc[0]["route_exposure"])

        predictions, model, training_samples = forecast_series(series_df)
        predictions.insert(0, "sku", spec["sku"])
        predictions.insert(1, "market", spec["market"])
        predictions.insert(2, "channel", spec["channel"])
        predictions.insert(3, "category", category)
        predictions.insert(4, "route_exposure", route)
        all_predictions.append(predictions)

        actual = predictions["actual_units_sold"]
        baseline = predictions["seasonal_naive_forecast"]
        rf_pred = predictions["random_forest_forecast"]

        metric_rows.extend(
            [
                {
                    "sku": spec["sku"],
                    "market": spec["market"],
                    "channel": spec["channel"],
                    "category": category,
                    "route_exposure": route,
                    "model": "Seasonal Naive (28-day lag)",
                    "training_end": TRAIN_END.date(),
                    "holdout_start": HOLDOUT_START.date(),
                    "holdout_end": HOLDOUT_END.date(),
                    "training_samples": training_samples,
                    "mae": mean_absolute_error(actual, baseline),
                    "wape_percent": wape(actual, baseline),
                    "actual_total_units": actual.sum(),
                    "forecast_total_units": baseline.sum(),
                },
                {
                    "sku": spec["sku"],
                    "market": spec["market"],
                    "channel": spec["channel"],
                    "category": category,
                    "route_exposure": route,
                    "model": "Random Forest (recursive lags)",
                    "training_end": TRAIN_END.date(),
                    "holdout_start": HOLDOUT_START.date(),
                    "holdout_end": HOLDOUT_END.date(),
                    "training_samples": training_samples,
                    "mae": mean_absolute_error(actual, rf_pred),
                    "wape_percent": wape(actual, rf_pred),
                    "actual_total_units": actual.sum(),
                    "forecast_total_units": rf_pred.sum(),
                },
            ]
        )

        for feature_name, importance in zip(FEATURE_NAMES, model.feature_importances_):
            importance_rows.append(
                {
                    "sku": spec["sku"],
                    "route_exposure": route,
                    "feature": feature_name,
                    "importance": float(importance),
                }
            )

        # Individual chart: Actual vs both forecast methods.
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.plot(
            predictions["date"],
            predictions["actual_units_sold"],
            marker="o",
            markersize=3,
            label="Actual",
        )
        ax.plot(
            predictions["date"],
            predictions["seasonal_naive_forecast"],
            label="Seasonal Naive",
        )
        ax.plot(
            predictions["date"],
            predictions["random_forest_forecast"],
            label="Random Forest",
        )
        ax.set_title(
            f"28-Day Forecast vs Actual — {spec['sku']} | "
            f"{spec['market']} {spec['channel']} | {route}"
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Units Sold")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.autofmt_xdate()
        fig.tight_layout()
        chart_path = OUTPUT_DIR / f"forecast_{spec['sku']}.png"
        fig.savefig(chart_path, dpi=160)
        plt.close(fig)

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    metrics_df = pd.DataFrame(metric_rows)
    importance_df = pd.DataFrame(importance_rows)

    predictions_df.to_csv(OUTPUT_DIR / "forecast_predictions.csv", index=False)
    metrics_df.to_csv(OUTPUT_DIR / "forecast_metrics.csv", index=False)
    importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    # Human-readable summary for presentation/report preparation.
    summary_lines = [
        "NORTHSTAR FORECASTING SUMMARY",
        "=" * 70,
        f"Training data end: {TRAIN_END.date()}",
        f"Holdout period: {HOLDOUT_START.date()} to {HOLDOUT_END.date()} (28 days)",
        "Baseline: Seasonal Naive using the value from exactly 28 days earlier",
        "Second model: Random Forest with leakage-safe lag/rolling/calendar features",
        "Metrics: MAE and WAPE",
        "",
    ]

    for _, row in metrics_df.iterrows():
        summary_lines.append(
            f"{row['sku']} | {row['route_exposure']} | {row['model']} | "
            f"MAE={row['mae']:.3f} | WAPE={row['wape_percent']:.2f}% | "
            f"Actual total={row['actual_total_units']:.0f} | "
            f"Forecast total={row['forecast_total_units']:.1f}"
        )

    summary_lines.extend(
        [
            "",
            "Interpretation reminder:",
            "Lower MAE and lower WAPE indicate lower forecast error on the held-out period.",
            "These forecast results measure predictive accuracy; they do not prove that route",
            "exposure or the Suez event caused changes in sales.",
        ]
    )

    (OUTPUT_DIR / "forecast_summary.txt").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )

    print("Forecasting completed successfully.")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("\nMetrics:")
    display_cols = ["sku", "route_exposure", "model", "mae", "wape_percent"]
    print(metrics_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()