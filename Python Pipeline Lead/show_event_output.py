import pandas as pd

df = pd.read_csv("northstar_event_impact_ready.csv")

# Select the columns to display
output = df[
    [
        "event_period",
        "sku",
        "route_exposure",
        "units_sold",
        "stockout_flag",
        "cancelled_units",
        "transit_delay_days",
        "opening_inventory_units"
    ]
]

# Summarize by event period, SKU, and route exposure
summary = (
    output
    .groupby(["event_period", "sku", "route_exposure"], as_index=False)
    .agg(
        records=("sku", "size"),
        units_sold=("units_sold", "sum"),
        stockout_rate=("stockout_flag", "mean"),
        cancelled_units=("cancelled_units", "sum"),
        transit_delay_days=("transit_delay_days", "mean"),
        opening_inventory_units=("opening_inventory_units", "mean")
    )
)

# Convert stockout rate to percentage
summary["stockout_rate"] = summary["stockout_rate"] * 100

print("\n===== EVENT IMPACT SUMMARY =====\n")
print(summary.to_string(index=False))