"""
Extract Regional Network data from Excel and create clean CSVs for analysis.

Purpose: 
- Extract key economic indicators from Regional Network reports
- Prepare data for merging with interest rate meeting dates
- Output clean CSVs with quarterly observations

Variables extracted:
- Output: Aggregated current quarter & next quarter (GDP proxy)
- Capacity utilisation: Capacity constraints & labour shortage (slack indicators)
- Employment: Current quarter & next quarter (labour market)
"""

import pandas as pd
from pathlib import Path

# File paths
output_dir = Path('02_Data/clean_consolidated')
output_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("EXTRACTING REGIONAL NETWORK DATA")
print("="*80)

# ============================================================================
# 1. OUTPUT (GDP PROXY)
# ============================================================================
print("\n1. Processing Output data...")

df_output_clean = pd.read_csv('Environment/rn_output.csv', parse_dates=['date'])
df_output_clean.rename(columns={
    'output_current': 'output_current_quarter',
    'output_next': 'output_next_quarter'
}, inplace=True)

print(f"   Clean shape: {df_output_clean.shape}")
print(f"   Date range: {df_output_clean['date'].min()} to {df_output_clean['date'].max()}")
print(f"   Sample data:\n{df_output_clean.head(3)}")

# ============================================================================
# 2. CAPACITY UTILISATION (SLACK INDICATORS)
# ============================================================================
print("\n2. Processing Capacity Utilisation data...")

df_capacity_clean = pd.read_csv('Environment/rn_capacity.csv', parse_dates=['date'])
df_capacity_clean.rename(columns={
    'capacity_constraints': 'capacity_constraints_pct',
    'labour_shortage': 'labour_shortage_pct'
}, inplace=True)

print(f"   Clean shape: {df_capacity_clean.shape}")
print(f"   Date range: {df_capacity_clean['date'].min()} to {df_capacity_clean['date'].max()}")
print(f"   Sample data:\n{df_capacity_clean.head(3)}")

# ============================================================================
# 3. EMPLOYMENT (LABOUR MARKET)
# ============================================================================
print("\n3. Processing Employment data...")

df_employment_clean = pd.read_csv('Environment/rn_employment.csv', parse_dates=['date'])

print(f"   Clean shape: {df_employment_clean.shape}")
print(f"   Date range: {df_employment_clean['date'].min()} to {df_employment_clean['date'].max()}")
print(f"   Sample data:\n{df_employment_clean.head(3)}")

# ============================================================================
# 4. MERGE ALL INDICATORS
# ============================================================================
print("\n4. Merging all indicators...")

# Start with output (likely longest series)
df_merged = df_output_clean.copy()

# Merge capacity utilisation
df_merged = df_merged.merge(df_capacity_clean, on='date', how='outer')

# Merge employment
df_merged = df_merged.merge(df_employment_clean, on='date', how='outer')

# Sort by date
df_merged = df_merged.sort_values('date').reset_index(drop=True)

# Filter to 2005 onwards (as per user's requirement)
df_merged = df_merged[df_merged['date'] >= '2005-01-01']

print(f"   Final shape: {df_merged.shape}")
print(f"   Date range: {df_merged['date'].min()} to {df_merged['date'].max()}")
print(f"   Total observations: {len(df_merged)}")

# Check for missing values
print(f"\n   Missing values by column:")
print(df_merged.isnull().sum())

# ============================================================================
# 5. SAVE TO CSV
# ============================================================================
output_file = output_dir / 'regional_network_quarterly.csv'
df_merged.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n{'='*80}")
print(f"SUCCESS: Regional Network data saved to:")
print(f"  {output_file}")
print(f"{'='*80}")

# Display summary statistics
print("\nSummary Statistics:")
print(df_merged.describe())

print("\nFirst 10 observations:")
print(df_merged.head(10).to_string(index=False))

print("\nLast 5 observations:")
print(df_merged.tail(5).to_string(index=False))
