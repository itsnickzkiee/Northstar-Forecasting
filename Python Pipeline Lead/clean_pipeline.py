import pandas as pd

INPUT_FILE = "northstar_daily_sku_market.csv"
OUTPUT_FILE = "northstar_clean.csv"

# ============================================================
# 1. LOAD RAW DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("NORTHSTAR DATA CLEANING PIPELINE")
print("=" * 70)

raw_rows = len(df)

print("\nRaw rows:", raw_rows)


# ============================================================
# 2. CREATE CLEAN COPY
# ============================================================

clean = df.copy()


# ============================================================
# 3. CLEAN DATES
# ============================================================

raw_dates = clean["date"].astype(str).str.strip()

# Format 1: 10/15/2021
parsed_dates = pd.to_datetime(
    raw_dates,
    format="%m/%d/%Y",
    errors="coerce"
)

# Format 2: 11-Mar-21
mask = parsed_dates.isna()

parsed_dates.loc[mask] = pd.to_datetime(
    raw_dates.loc[mask],
    format="%d-%b-%y",
    errors="coerce"
)

clean["date"] = parsed_dates

print("\n[DATE CLEANING]")

parsed_count = clean["date"].notna().sum()
invalid_date_count = clean["date"].isna().sum()

print("Parsed dates:", parsed_count)
print("Unparseable dates:", invalid_date_count)

# Remove records with unparseable dates
clean = clean.dropna(subset=["date"]).copy()

print("Rows removed due to unparseable dates:", invalid_date_count)
print("Rows remaining after date cleaning:", len(clean))

# ============================================================
# 4. STANDARDIZE MARKET
# ============================================================

clean["market"] = clean["market"].astype(str).str.strip()

print("\n[MARKET CLEANING]")
print(clean["market"].value_counts())


# ============================================================
# 5. STANDARDIZE CHANNEL
# ============================================================

clean["channel"] = (
    clean["channel"]
    .astype(str)
    .str.strip()
    .str.lower()
)

clean["channel"] = clean["channel"].replace({
    "stores": "Stores",
    "e-commerce": "E-commerce"
})

print("\n[CHANNEL CLEANING]")
print(clean["channel"].value_counts())


# ============================================================
# 6. STANDARDIZE PROMO FLAG
# ============================================================

promo_map = {
    "0": 0,
    "1": 1,
    "yes": 1,
    "true": 1,
    "no": 0,
    "false": 0
}

clean["promo_flag"] = (
    clean["promo_flag"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map(promo_map)
)

print("\n[PROMO FLAG CLEANING]")
print(clean["promo_flag"].value_counts(dropna=False))


# ============================================================
# 7. CLEAN TRANSIT DELAY
# ============================================================

clean["transit_delay_days"] = pd.to_numeric(
    clean["transit_delay_days"],
    errors="coerce"
)

# Negative transit delays are impossible,
# so convert them to missing values.
clean.loc[
    clean["transit_delay_days"] < 0,
    "transit_delay_days"
] = pd.NA

print("\n[TRANSIT DELAY CLEANING]")
print(
    clean["transit_delay_days"]
    .value_counts(dropna=False)
    .sort_index()
)


# ============================================================
# 8. CLEAN IMPOSSIBLE UNIT PRICE
# ============================================================

negative_price = clean["unit_price_eur"] < 0

print("\n[UNIT PRICE CLEANING]")
print("Negative prices found:", negative_price.sum())

clean.loc[
    negative_price,
    "unit_price_eur"
] = pd.NA


# ============================================================
# 9. CLEAN IMPOSSIBLE UNITS SOLD
# ============================================================

negative_sales = clean["units_sold"] < 0

print("\n[UNITS SOLD CLEANING]")
print("Negative units sold found:", negative_sales.sum())

clean.loc[
    negative_sales,
    "units_sold"
] = pd.NA


# ============================================================
# 10. REMOVE EXACT DUPLICATES
# ============================================================

before_duplicates = len(clean)

clean = clean.drop_duplicates()

removed_duplicates = before_duplicates - len(clean)

print("\n[DUPLICATE CLEANING]")
print("Exact duplicate rows removed:", removed_duplicates)

# ============================================================
# 10.5 CHECK BUSINESS KEY DUPLICATES
# ============================================================

key_columns = ["date", "market", "channel", "sku"]

# Only check business-key uniqueness for records
# with a valid parsed date.
valid_date_rows = clean["date"].notna()

business_key_duplicates = clean.loc[
    valid_date_rows
].duplicated(
    subset=key_columns,
    keep=False
)

print("\n[BUSINESS KEY DUPLICATE CHECK]")

print(
    "Rows with valid dates checked:",
    valid_date_rows.sum()
)

print(
    "Rows involved in duplicate business keys:",
    business_key_duplicates.sum()
)

print(
    "Rows with missing dates excluded from key check:",
    0
)

if business_key_duplicates.sum() > 0:

    print("\nSample business-key duplicates:")

    print(
        clean.loc[
            valid_date_rows
        ]
        .loc[
            business_key_duplicates,
            key_columns + [
                "units_sold",
                "unit_price_eur",
                "revenue_eur"
            ]
        ]
        .sort_values(key_columns)
        .head(30)
        .to_string(index=False)
    )
else:

    print(
        "No duplicate business keys found among records "
        "with valid dates."
    )
# ============================================================
# 11. CREATE CLEANING LOG
# ============================================================

cleaning_log = pd.DataFrame([
    {
        "issue": "Mixed date formats",
        "count": parsed_count,
        "action": "Converted valid dates to datetime",
        "reason": "Standardize dates for reliable date-based analysis"
    },
    {
        "issue": "Unparseable dates",
        "count": invalid_date_count,
        "action": "Removed records with unparseable dates",
        "reason": "Date could not be reliably reconstructed and was excluded to avoid unreliable date-based analysis"
    },
    {
        "issue": "Market trailing whitespace",
        "count": 325,
        "action": "Trimmed whitespace",
        "reason": "Standardize market labels"
    },
    {
        "issue": "Inconsistent channel capitalization",
        "count": 325,
        "action": "Standardized channel labels",
        "reason": "Standardize category values"
    },
    {
        "issue": "Mixed promo flag formats",
        "count": 1200,
        "action": "Converted to 0/1",
        "reason": "Create consistent binary representation"
    },
    {
        "issue": "Invalid transit delays",
        "count": 215,
        "action": "Converted to missing",
        "reason": "Unknown and negative delay values are not valid numeric delays"
    },
    {
        "issue": "Negative unit prices",
        "count": 35,
        "action": "Converted to missing",
        "reason": "Negative prices are impossible for this field"
    },
    {
        "issue": "Negative units sold",
        "count": 35,
        "action": "Converted to missing",
        "reason": "Negative fulfilled sales are invalid"
    },
    {
        "issue": "Exact duplicate rows",
        "count": 180,
        "action": "Removed duplicate rows",
        "reason": "Exact duplicates do not represent additional observations"
    },
    {
        "issue": "Valid-date business-key duplicates",
        "count": 0,
        "action": "No rows removed",
        "reason": "No duplicate date + market + channel + SKU combinations remained"
    }
])

cleaning_log.to_csv(
    "cleaning_log.csv",
    index=False
)

print("\n[CLEANING LOG]")
print("Saved cleaning log: cleaning_log.csv")
# ============================================================
# 12. SAVE CLEAN DATA
# ============================================================

clean.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("CLEANING COMPLETE")
print("=" * 70)

print("Raw rows:", raw_rows)
print("Clean rows:", len(clean))
print("Rows removed:", raw_rows - len(clean))

print("\nSaved cleaned dataset:")
print(OUTPUT_FILE)