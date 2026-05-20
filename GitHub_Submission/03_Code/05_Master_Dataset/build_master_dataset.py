"""Build the macro master dataset for ordered probit analysis."""

# so we import pandas to read and manipulate dataframes, which really does the important work in this script in terms of merging all the data.
# numpy is important especially for the conditional logic, like "if rate change > 0, decision = 1" etc...
# datetime actually also plays a key role here, especially in defining the date where target policy rate changed from 2.5 to 2.0%

import pandas as pd
import numpy as np
from datetime import datetime

def load_meeting_dates():
    """
    Load Norges Bank meeting dates and policy rates from Excel.
    
    Returns:
        DataFrame with columns: meeting_date, policy_rate
    """
    # We use an excel file downloaded from Norges Bank in order to get the dates of decision announcement and the rates.
    # Importantly, the meeting_date variable I defined here is somehwat misleading, because it is actually the announcement date.
    # I didnt realize this at first, so this logic is handled later in the script. Obviously this is absolutely crucial to get right,
    # as it could introduce some nasty look-ahead bias if coincidentally the scripts (by accident) assumes that the committee
    # has access to economic indicators that are published on the same day as the announcement. 
    df = pd.read_csv('Environment/meeting_date_and_rate.csv',
                     parse_dates=['meeting_date'])
    df = df.dropna(subset=['meeting_date']).copy()
    df = df.sort_values('meeting_date').reset_index(drop=True)
    
    print(f"meetings={len(df)}, {df['meeting_date'].min().date()} to {df['meeting_date'].max().date()}")
    
    return df

# This is really the key RN variable used in the analysis. It is a forward-looking indicator of economic activity based on surveys of business leaders etc.
# It is a taylor-rule-esque indicator, but I thought it would be cool to include it as a sort of "real-time" indicator,
# as some of the literature supports and suggests actually serves as an improvement to the taylor framework.
def extract_rn_output():
    """
    Extract Regional Network Output indicators.
    
    Returns:
        DataFrame with columns: date, output_next
    """
    df_clean = pd.read_csv('Environment/rn_output.csv',
                          parse_dates=['date'])
    df_clean = df_clean.dropna(subset=['date']).reset_index(drop=True)
    
    print(f"rn_output_quarters={len(df_clean)}, {df_clean['date'].min().date()} to {df_clean['date'].max().date()}")
    
    return df_clean

# I also wanted to include some indicators of capacity constraints and labor shortages, but I scrapped this idea at a later point.
# for one they are quite highly correlated, and second there was not really need for this many RN indicators.
# So, this really just reflects an artifact of the process. I still kept the code here, just for illustrating the process.
def extract_rn_capacity():
    """
    Extract Regional Network Capacity Utilisation indicators.
    
    Returns:
        DataFrame with columns: date, capacity_constraints, labour_shortage
    """
    df_clean = pd.read_csv('Environment/rn_capacity.csv',
                          parse_dates=['date'])
    df_clean = df_clean.dropna(subset=['date']).reset_index(drop=True)
    
    print(f"rn_capacity_quarters={len(df_clean)}, {df_clean['date'].min().date()} to {df_clean['date'].max().date()}")
    
    return df_clean


# Once again, this is a simple merge function to merge the RN indicators on date, and thus partfly an artifact of the previous idea to expand upon Taylor-rule.
# This is because capacity contraints and labor shortage indicators were never actually used in regression, and I keep them in the code justfor illustrating the process
# as I built it and learned along the way.
def merge_rn_indicators(output_df, capacity_df):
    """
    Merge all RN indicators on date.
    
    Returns:
        DataFrame with all RN indicators
    """
    rn_data = output_df.merge(capacity_df, on='date', how='outer')
    rn_data = rn_data.sort_values('date').reset_index(drop=True)
    
    print(f"rn_merged_quarters={len(rn_data)}")
    
    return rn_data


def asof_merge_meetings_rn(meetings_df, rn_df):
    """
    Perform as-of merge: each meeting gets the most recent RN data published
    strictly before the decision date.
    
    Norges Bank's Executive Board makes its rate decision one day before
    the public announcement (meeting_date). To avoid look-ahead bias,
    we use decision_date = meeting_date - 1 day as the reference point
    for backward merging.  With pd.merge_asof(direction='backward'),
    only data published on or before decision_date is matched.
    
    Args:
        meetings_df: DataFrame with meeting_date column
        rn_df: DataFrame with date column (RN publication dates)
    
    Returns:
        DataFrame with meetings and matched RN indicators
    """
    meetings_df = meetings_df.sort_values('meeting_date').copy()
    rn_df = rn_df.sort_values('date').copy()
    
    # Decision is made one day before publication - use that as merge key. This logic is important to ensure we do not
    # accidentally introduce any look-ahead bias by merging any values that were not actually avalable at the decision date.
    meetings_df['decision_date'] = meetings_df['meeting_date'] - pd.Timedelta(days=1)
    
    meetings_df_indexed = meetings_df.set_index('decision_date')
    rn_df_indexed = rn_df.set_index('date')
    
    merged = pd.merge_asof(
        meetings_df_indexed,
        rn_df_indexed,
        left_index=True,
        right_index=True,
        direction='backward'
    )
    
    merged = merged.reset_index(drop=True)
    
    print(f"rn_meeting_rows={len(merged)}")
    
    rn_columns = ['output_next', 'capacity_constraints', 'labour_shortage']
    meetings_with_data = merged[rn_columns].notna().all(axis=1).sum()
    print(f"rn_complete_meetings={meetings_with_data}/{len(merged)}")
    
    return merged

# This function is another crucial part of the analysis. It adds the lagged policy rate, which is a key predictor in the regression.
# Because of central banks' preference for rate smoothing, adding a lagged rate component is a must. Here, shift(1) is used to get the previous
# meeting's rate, which is the current rate at the time of the decision.
def add_lagged_rate(df):
    """
    Add lagged policy rate (rate at meeting T for predicting decision at T+1).
    
    Args:
        df: Master dataset with policy_rate column
    
    Returns:
        DataFrame with lagged_rate column
    """
    # Use shift(1) to get the previous meeting's policy rate as the lagged rate for any current meeting.
    df['lagged_rate'] = df['policy_rate'].shift(1)
    
    return df

# CPI-ATE, which we gathered from SSB and curated in an excel file is another key variable that is used for the inflation gap measure.
# Initially I believed that the figure was released the 10th of every month without exception, but I realized that there is some deviation
# because of weekends and holidays. Thus, I curated a file using SSB's archives to get every single, exact, publication date.
def load_cpi_ate_data():
    """
    Load CPI-ATE YoY figures from SSB Excel file.
    
    The Excel file uses actual SSB publication dates as column headers in
    DD.MM.YYYY format (such that "10.01.2006" = CPI-ATE for Dec 2005, published
    10 January 2006).  These dates are used directly as the publication_date
    for the as-of merge with meeting dates, no synthetic lag is applied.
    It is not necessary to apply such a lag because the publication dates are
    already the actual release dates.
    
    Returns:
        DataFrame with columns: publication_date, cpi_ate_yoy
    """
    cpi_data = pd.read_csv('Environment/cpi_ate.csv',
                          parse_dates=['publication_date'])
    cpi_data = cpi_data.sort_values('publication_date').reset_index(drop=True)
    
    print(f"cpi_ate_observations={len(cpi_data)}")
    print(f"cpi_ate_publication_range={cpi_data['publication_date'].min().date()} to {cpi_data['publication_date'].max().date()}")
    print(f"cpi_ate_range={cpi_data['cpi_ate_yoy'].min():.1f}% to {cpi_data['cpi_ate_yoy'].max():.1f}%")
    
    return cpi_data


# For this function I believe the docstring already explains the logic quite well, but the point is that the inflation target
# changed from 2.5% to 2.0% in March 2018, so we need to account for this in order to get the correct measure of inflation gap.
def calculate_inflation_target(meeting_date):
    """
    Calculate inflation target for a given meeting date.
    
    Target regime:
    - Before March 2, 2018: 2.5%
      (Inflation targeting formally adopted March 29, 2001, target set at 2.5%)
    - March 2, 2018 and after: 2.0%
      (Target lowered to 2.0% by the Norwegian government on March 2, 2018)
    
    Note: This function compares against meeting_date (announcement date),
    not decision_date. The nearest meetings are 2018-01-25 and 2018-03-15
    (decision dates: Jan 24 and Mar 14). Both fall clearly on opposite sides
    of the March 2 cutoff, so the result is correct regardless of whether
    we compare announcement or decision date.
    
    Args:
        meeting_date: datetime object
    
    Returns:
        float: Inflation target (2.5 or 2.0)
    """
    regime_change_date = pd.to_datetime('2018-03-02')
    
    if meeting_date < regime_change_date:
        return 2.5
    else:
        return 2.0

# So for this function the logic is the same as for the RN variables. Decision_date - 1 day is used in order to ensure
# that we simulate the information set available to the committee at the time of the decision.
# In addition, every meeting is now paired with the most recent CPI-ATE figures published to every meeting date.
def merge_cpi_with_meetings(master_df, cpi_data):
    """
    Merge CPI-ATE data with meeting dates using as-of logic.
    
    Args:
        master_df: Master dataset with meeting dates
        cpi_data: CPI-ATE data with publication dates
    
    Returns:
        DataFrame with CPI-ATE data merged
    """
    master_df = master_df.sort_values('meeting_date').copy()
    cpi_data = cpi_data.sort_values('publication_date').copy()
    
    # Decision is made one day before publication - use that as merge key. This is a crucial step that I talked about earlier,
    # as it ensures that we do not accidentally introduce any look-ahead bias by merging any values that were not actually avalable at the decision date.
    # As I communicated, I wasnt initially aware that the decision was made one day before the announcement, even though it is clearly communicated on Norges Banks website.
    # Luckily I caught the error, but I also ran the regression prior to this, and the results were very similar, as there was only very few cases where this was a problem.
    master_df['decision_date'] = master_df['meeting_date'] - pd.Timedelta(days=1)
    
    master_indexed = master_df.set_index('decision_date')
    cpi_indexed = cpi_data[['publication_date', 'cpi_ate_yoy']].set_index('publication_date')
    
    # Notice in this merge logic we use direction=backward, which means that for each decision_date, we get the most recent CPI-ATE available.
    merged = pd.merge_asof(
        master_indexed,
        cpi_indexed,
        left_index=True,
        right_index=True,
        direction='backward' 
    )
    
    merged = merged.reset_index(drop=True)
    
    print(f"cpi_meeting_rows={len(merged)}")
    
    missing = merged['cpi_ate_yoy'].isna().sum()
    if missing > 0:
        print(f"  WARNING: {missing} meetings have no CPI-ATE data (too early)")
    
    return merged


def add_inflation_gap(master_df):
    """
    Calculate inflation gap: actual CPI-ATE - inflation target.
    
    Args:
        master_df: Master dataset with cpi_ate_yoy column
    
    Returns:
        DataFrame with inflation_gap column added
    """
    # Calculate the inflation gap as actual CPI_ATE minus the target (2.5 or 2.0%). Positive means inflation above target, and negative opposite.
    master_df['inflation_target'] = master_df['meeting_date'].apply(calculate_inflation_target)
    
    # Round to 2 decimals to avoid floating point precision errors
    master_df['inflation_gap'] = (master_df['cpi_ate_yoy'] - master_df['inflation_target']).round(2)
    
    print(f"inflation_target_2_5={(master_df['inflation_target'] == 2.5).sum()}")
    print(f"inflation_target_2_0={(master_df['inflation_target'] == 2.0).sum()}")
    print(f"inflation_gap_range={master_df['inflation_gap'].min():.2f} to {master_df['inflation_gap'].max():.2f}")
    
    return master_df

# This function create the decision variable, the dependent variable in the regression.
def create_decision_variable(master_df):
    """
    Create decision_y: the rate change at this meeting (T+1) in percentage points.
    
    Args:
        master_df: Master dataset with policy_rate and lagged_rate
    
    Returns:
        DataFrame with decision_y column
    """
    # Calculate the rate change between the current 'policy_rate' and the 'lagged_rate' of the previous meeting to find bp change.
    master_df['decision_y'] = master_df['policy_rate'] - master_df['lagged_rate']
    
    print("\ndecision_distribution:")
    decision_dist = master_df['decision_y'].value_counts().sort_index()
    for change, count in decision_dist.items():
        if pd.notna(change):
            print(f"  {change:+.2f}: {count} ({100*count/len(master_df):.1f}%)")
    
    return master_df

# 'finalize_dataset' is our final function, which brings everything together and selects the columns we want to include in the dataset for regression.
# As Ive mentioned, I decided to remove capacity_constraints and labour_shortage at a later stage, so theyre here more for illustrative purposes.
def finalize_dataset(master_df):
    """
    Select final columns for lean model specification.
    
    Returns:
        DataFrame with selected columns only
    """
    final_columns = [
        'meeting_date',
        'decision_y',
        'inflation_gap',
        'output_next',
        'capacity_constraints',
        'labour_shortage',
        'lagged_rate'
    ]
    
    final_df = master_df[final_columns].copy()
    
    print(f"final_shape={final_df.shape}")
    
    return final_df

def main():
    """
    Main execution: Build complete master dataset.
    """
    # Part 1: Meeting dates and RN indicators
    meetings = load_meeting_dates()
    rn_output = extract_rn_output()
    rn_capacity = extract_rn_capacity()
    rn_data = merge_rn_indicators(rn_output, rn_capacity)
    master = asof_merge_meetings_rn(meetings, rn_data)
    
    # Compute lag BEFORE filtering so the Dec-2005 meeting provides lagged_rate. Because we forward-fill the most recent
    # figures available, for the Jan 2006 (first meeting in the sample) it is crucial we forward-fill from the Dec 2005 meeting.
    # The first couple of runs I really couldnt figure out why I had missing values for the lagged_rate variable, but it soon became obvious...
    # for the first in-sample meeting (2006-01-25).
    master = add_lagged_rate(master)
    
    # Filter to analysis period (2006-2025)
    master = master[master['meeting_date'].dt.year >= 2006].copy()
    master = master[master['meeting_date'] <= '2025-12-31'].copy()
    print(f"analysis_meetings={len(master)}, {master['meeting_date'].min().date()} to {master['meeting_date'].max().date()}")
    
    # Part 2: CPI-ATE and inflation gap
    # Publication dates are now actual release dates (DD.MM.YYYY column headers),
    # so no lag function is needed - merge_cpi_with_meetings uses them directly.
    cpi_data = load_cpi_ate_data()
    master = merge_cpi_with_meetings(master, cpi_data)
    master = add_inflation_gap(master)
    master = create_decision_variable(master)
    
    # Finalize and save
    final_master = finalize_dataset(master)
    
    output_path = '02_Data/master_dataset_2006_2025.csv'
    final_master.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    # Summary
    print(f"\nsaved: {output_path}")
    print(f"meetings_final={len(final_master)}")
    print(f"date_range={final_master['meeting_date'].min().date()} to {final_master['meeting_date'].max().date()}")
    
    print("missing_values:")
    print(final_master.isnull().sum())


if __name__ == "__main__":
    main()
