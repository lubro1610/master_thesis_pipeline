"""Ordered probit regressions for the main model specifications."""

import pandas as pd
import numpy as np
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('02_Data/master_dataset_with_sentiment.csv')
df['meeting_date'] = pd.to_datetime(df['meeting_date'])

print(f"sample: n={len(df)}, {df['meeting_date'].min().date()} to {df['meeting_date'].max().date()}")

# Check for missing values
print("\nmissing values:")
key_vars = ['decision_ordinal', 'inflation_gap', 'output_next', 'lagged_rate', 
            'lm_sentiment_std', 'finbert_sentiment_std']
for var in key_vars:
    missing = df[var].isna().sum()
    print(f"  {var:25s}: {missing:3d} ({missing/len(df)*100:.1f}%)")

# Drop rows with missing values in key variables
# Use dropna(subset=...) to preserve all columns (meeting_date etc.) for later use
df_clean = df.dropna(subset=key_vars).copy()
print(f"complete_cases: {len(df_clean)}/{len(df)} ({len(df_clean)/len(df)*100:.1f}%)")

# Distribution of dependent variable
print("\ndecision_ordinal distribution:")
print(df_clean['decision_ordinal'].value_counts().sort_index())
cuts = (df_clean['decision_ordinal'] == -1).sum()
holds = (df_clean['decision_ordinal'] == 0).sum()
hikes = (df_clean['decision_ordinal'] == 1).sum()
print(f"  Cuts:  {cuts:3d} ({cuts/len(df_clean)*100:.1f}%)")
print(f"  Holds: {holds:3d} ({holds/len(df_clean)*100:.1f}%)")
print(f"  Hikes: {hikes:3d} ({hikes/len(df_clean)*100:.1f}%)")

# Descriptive statistics
print("\ndescriptive statistics:")
print("\n" + df_clean[['inflation_gap', 'output_next', 'lagged_rate', 
                        'lm_sentiment_std', 'finbert_sentiment_std']].describe().to_string())

# Correlation matrix
print("\ncorrelation matrix:")
corr = df_clean[['inflation_gap', 'output_next', 'lagged_rate', 
                  'lm_sentiment_std', 'finbert_sentiment_std']].corr()
print("\n" + corr.round(3).to_string())

lm_fb_corr = corr.loc['lm_sentiment_std', 'finbert_sentiment_std']
print(f"lm_finbert_corr: {lm_fb_corr:.3f}")

# Variance Inflation Factors (VIF) for Model 4 (Combined)
print("\nvariance inflation factors:")
vif_vars = ['inflation_gap', 'output_next', 'lagged_rate',
            'lm_sentiment_std', 'finbert_sentiment_std']
X_vif = df_clean[vif_vars].copy()
X_vif.insert(0, 'const', 1)  # VIF requires intercept column
print(f"\n{'Variable':<25} {'VIF':>8}")
for i, var in enumerate(X_vif.columns):
    if var == 'const':
        continue
    vif_val = variance_inflation_factor(X_vif.values, i)
    flag = " high" if vif_val > 5 else ""
    print(f"  {var:<23} {vif_val:>8.2f}{flag}")

y = df_clean['decision_ordinal']  # -1 = cut, 0 = hold, 1 = hike


def get_actual_cutpoints(result):
    cutpoints = result.model.transform_threshold_params(result.params.values)[1:-1]
    return cutpoints[0], cutpoints[1]


def print_actual_cutpoints(result):
    tau_1, tau_2 = get_actual_cutpoints(result)
    print(f"cutpoints: tau_1={tau_1:.6f}, tau_2={tau_2:.6f}")

# Model 1: Baseline (macro only)
print("\nModel 1: Baseline")
X1 = df_clean[['inflation_gap', 'output_next', 'lagged_rate']]
model1 = OrderedModel(y, X1, distr='probit')
result1 = model1.fit(method='bfgs', disp=False)
print(result1.summary())
print_actual_cutpoints(result1)

# Model 2: LM Sentiment
print("\nModel 2: LM")
X2 = df_clean[['inflation_gap', 'output_next', 'lagged_rate', 'lm_sentiment_std']]
model2 = OrderedModel(y, X2, distr='probit')
result2 = model2.fit(method='bfgs', disp=False)
print(result2.summary())
print_actual_cutpoints(result2)

# Model 3: FinBERT Sentiment
print("\nModel 3: FinBERT")
X3 = df_clean[['inflation_gap', 'output_next', 'lagged_rate', 'finbert_sentiment_std']]
model3 = OrderedModel(y, X3, distr='probit')
result3 = model3.fit(method='bfgs', disp=False)
print(result3.summary())
print_actual_cutpoints(result3)

# Model 4: Combined (Horse Race)
print("\nModel 4: Combined")
X4 = df_clean[['inflation_gap', 'output_next', 'lagged_rate', 
               'lm_sentiment_std', 'finbert_sentiment_std']]
model4 = OrderedModel(y, X4, distr='probit')
result4 = model4.fit(method='bfgs', disp=False)
print(result4.summary())
print_actual_cutpoints(result4)

print("\nmodel comparison:")

models = {
    'Baseline': result1,
    'LM Only': result2,
    'FinBERT Only': result3,
    'Combined': result4
}

print(f"\n{'Model':<20} {'Log-Lik':>12} {'Pseudo R²':>12} {'AIC':>12} {'BIC':>12} {'N':>6}")

for name, result in models.items():
    ll = result.llf
    pseudo_r2 = result.prsquared
    aic = result.aic
    bic = result.bic
    n = result.nobs
    print(f"{name:<20} {ll:>12.2f} {pseudo_r2:>12.4f} {aic:>12.2f} {bic:>12.2f} {n:>6.0f}")

best_aic = min(models.items(), key=lambda x: x[1].aic)
best_bic = min(models.items(), key=lambda x: x[1].bic)
best_r2 = max(models.items(), key=lambda x: x[1].prsquared)

print(f"\nBest model by AIC: {best_aic[0]}")
print(f"Best model by BIC: {best_bic[0]}")
print(f"Best model by Pseudo R²: {best_r2[0]}")

print("\nsentiment coefficients:")

# LM coefficient from Model 2
lm_coef = result2.params['lm_sentiment_std']
lm_pval = result2.pvalues['lm_sentiment_std']
lm_se = result2.bse['lm_sentiment_std']
lm_sig = "***" if lm_pval < 0.01 else "**" if lm_pval < 0.05 else "*" if lm_pval < 0.10 else ""

print(f"LM sentiment (Model 2):")
print(f"  Coefficient: {lm_coef:+.4f} {lm_sig}")
print(f"  Std Error:   {lm_se:.4f}")
print(f"  p-value:     {lm_pval:.4f}")

# FinBERT coefficient from Model 3
fb_coef = result3.params['finbert_sentiment_std']
fb_pval = result3.pvalues['finbert_sentiment_std']
fb_se = result3.bse['finbert_sentiment_std']
fb_sig = "***" if fb_pval < 0.01 else "**" if fb_pval < 0.05 else "*" if fb_pval < 0.10 else ""

print(f"\nFinBERT sentiment (Model 3):")
print(f"  Coefficient: {fb_coef:+.4f} {fb_sig}")
print(f"  Std Error:   {fb_se:.4f}")
print(f"  p-value:     {fb_pval:.4f}")

# Compare magnitudes
if abs(fb_coef) > abs(lm_coef):
    ratio = abs(fb_coef) / abs(lm_coef) if lm_coef != 0 else float('inf')
    print(f"coefficient_ratio: FinBERT/LM={ratio:.1f}")
else:
    ratio = abs(lm_coef) / abs(fb_coef) if fb_coef != 0 else float('inf')
    print(f"coefficient_ratio: LM/FinBERT={ratio:.1f}")

# Coefficients in combined model (Model 4)
print(f"\ncombined model:")
lm_comb_coef = result4.params['lm_sentiment_std']
lm_comb_pval = result4.pvalues['lm_sentiment_std']
lm_comb_sig = "***" if lm_comb_pval < 0.01 else "**" if lm_comb_pval < 0.05 else "*" if lm_comb_pval < 0.10 else ""

fb_comb_coef = result4.params['finbert_sentiment_std']
fb_comb_pval = result4.pvalues['finbert_sentiment_std']
fb_comb_sig = "***" if fb_comb_pval < 0.01 else "**" if fb_comb_pval < 0.05 else "*" if fb_comb_pval < 0.10 else ""

print(f"  LM:      {lm_comb_coef:+.4f} {lm_comb_sig} (p={lm_comb_pval:.4f})")
print(f"  FinBERT: {fb_comb_coef:+.4f} {fb_comb_sig} (p={fb_comb_pval:.4f})")

print("\nlikelihood ratio tests:")

# LR test: Does LM improve over baseline?
lr_stat_lm = 2 * (result2.llf - result1.llf)
lr_pval_lm = stats.chi2.sf(lr_stat_lm, 1)
print(f"\nLM vs Baseline:")
print(f"  LR statistic: {lr_stat_lm:.4f}")
print(f"  p-value:      {lr_pval_lm:.4f} {'***' if lr_pval_lm < 0.01 else '**' if lr_pval_lm < 0.05 else '*' if lr_pval_lm < 0.10 else 'ns'}")

# LR test: Does FinBERT improve over baseline?
lr_stat_fb = 2 * (result3.llf - result1.llf)
lr_pval_fb = stats.chi2.sf(lr_stat_fb, 1)
print(f"\nFinBERT vs Baseline:")
print(f"  LR statistic: {lr_stat_fb:.4f}")
print(f"  p-value:      {lr_pval_fb:.4f} {'***' if lr_pval_fb < 0.01 else '**' if lr_pval_fb < 0.05 else '*' if lr_pval_fb < 0.10 else 'ns'}")

# LR test: Does combined improve over FinBERT alone?
lr_stat_comb = 2 * (result4.llf - result3.llf)
lr_pval_comb = stats.chi2.sf(lr_stat_comb, 1)
print(f"\nCombined vs FinBERT:")
print(f"  LR statistic: {lr_stat_comb:.4f}")
print(f"  p-value:      {lr_pval_comb:.4f} {'***' if lr_pval_comb < 0.01 else '**' if lr_pval_comb < 0.05 else '*' if lr_pval_comb < 0.10 else 'ns'}")

# Create results summary table
results_summary = pd.DataFrame({
    'Model': ['Baseline', 'LM Only', 'FinBERT Only', 'Combined'],
    'Log_Likelihood': [result1.llf, result2.llf, result3.llf, result4.llf],
    'Pseudo_R2': [result1.prsquared, result2.prsquared, result3.prsquared, result4.prsquared],
    'AIC': [result1.aic, result2.aic, result3.aic, result4.aic],
    'BIC': [result1.bic, result2.bic, result3.bic, result4.bic],
    'N_obs': [result1.nobs, result2.nobs, result3.nobs, result4.nobs]
})

results_summary.to_csv('04_Output/regression_model_comparison.csv', index=False)
print("saved: 04_Output/regression_model_comparison.csv")

# Coefficient table
coef_data = []
for name, result in [('Baseline', result1), ('LM Only', result2), 
                     ('FinBERT Only', result3), ('Combined', result4)]:
    for var in result.params.index:
        if not var.startswith('-1') and not var.startswith('0'):  # Skip threshold params
            coef_data.append({
                'Model': name,
                'Variable': var,
                'Coefficient': result.params[var],
                'Std_Error': result.bse[var],
                'z_stat': result.tvalues[var],
                'p_value': result.pvalues[var]
            })

coef_table = pd.DataFrame(coef_data)
coef_table.to_csv('04_Output/regression_coefficients.csv', index=False)
print("saved: 04_Output/regression_coefficients.csv")

cutpoint_data = []
for name, result in [('Baseline', result1), ('LM Only', result2),
                     ('FinBERT Only', result3), ('Combined', result4)]:
    tau_1, tau_2 = get_actual_cutpoints(result)
    cutpoint_data.append({
        'Model': name,
        'tau_1_cut_hold': tau_1,
        'tau_2_hold_hike': tau_2,
        'raw_threshold_1': result.params['-1/0'],
        'raw_threshold_2': result.params['0/1']
    })

cutpoint_table = pd.DataFrame(cutpoint_data)
cutpoint_table.to_csv('04_Output/regression_cutpoints.csv', index=False)
print("saved: 04_Output/regression_cutpoints.csv")
