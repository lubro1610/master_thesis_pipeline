"""Robustness check excluding the COVID period."""

import pandas as pd
import numpy as np
from statsmodels.miscmodels.ordinal_model import OrderedModel
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('02_Data/master_dataset_with_sentiment.csv')
df['meeting_date'] = pd.to_datetime(df['meeting_date'])
df['year'] = df['meeting_date'].dt.year

df_full = df.copy()
df_no_covid = df[(df['year'] < 2020) | (df['year'] > 2021)].copy()

print(f"full_sample_n={len(df_full)}")
print(f"no_covid_n={len(df_no_covid)}")
print(f"removed_n={len(df_full) - len(df_no_covid)}")

# Check decision distribution
print("\ndecision distribution, no COVID:")
print(df_no_covid['decision_ordinal'].value_counts().sort_index())
cuts = (df_no_covid['decision_ordinal'] == -1).sum()
holds = (df_no_covid['decision_ordinal'] == 0).sum()
hikes = (df_no_covid['decision_ordinal'] == 1).sum()
print(f"  Cuts:  {cuts:3d} ({cuts/len(df_no_covid)*100:.1f}%)")
print(f"  Holds: {holds:3d} ({holds/len(df_no_covid)*100:.1f}%)")
print(f"  Hikes: {hikes:3d} ({hikes/len(df_no_covid)*100:.1f}%)")

key_vars = ['decision_ordinal', 'inflation_gap', 'output_next', 'lagged_rate', 
            'lm_sentiment_std', 'finbert_sentiment_std']
df_clean = df_no_covid[key_vars].dropna()

print(f"complete_cases={len(df_clean)}/{len(df_no_covid)}")

y = df_clean['decision_ordinal']


def get_actual_cutpoints(result):
    cutpoints = result.model.transform_threshold_params(result.params.values)[1:-1]
    return cutpoints[0], cutpoints[1]


def print_actual_cutpoints(result):
    tau_1, tau_2 = get_actual_cutpoints(result)
    print(f"cutpoints: tau_1={tau_1:.6f}, tau_2={tau_2:.6f}")

# Model 1: Baseline
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

# Model 4: Combined
print("\nModel 4: Combined")
X4 = df_clean[['inflation_gap', 'output_next', 'lagged_rate', 
               'lm_sentiment_std', 'finbert_sentiment_std']]
model4 = OrderedModel(y, X4, distr='probit')
result4 = model4.fit(method='bfgs', disp=False)
print(result4.summary())
print_actual_cutpoints(result4)

print("\nmodel comparison, no COVID:")

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
print(f"\nBest model by AIC: {best_aic[0]}")

print("\nsentiment coefficients, no COVID:")

lm_coef = result2.params['lm_sentiment_std']
lm_pval = result2.pvalues['lm_sentiment_std']
lm_sig = "***" if lm_pval < 0.01 else "**" if lm_pval < 0.05 else "*" if lm_pval < 0.10 else ""

print(f"\nLM sentiment (Model 2):")
print(f"  Coefficient: {lm_coef:+.4f} {lm_sig}")
print(f"  p-value:     {lm_pval:.4f}")

fb_coef = result3.params['finbert_sentiment_std']
fb_pval = result3.pvalues['finbert_sentiment_std']
fb_sig = "***" if fb_pval < 0.01 else "**" if fb_pval < 0.05 else "*" if fb_pval < 0.10 else ""

print(f"\nFinBERT sentiment (Model 3):")
print(f"  Coefficient: {fb_coef:+.4f} {fb_sig}")
print(f"  p-value:     {fb_pval:.4f}")

lr_stat_lm = 2 * (result2.llf - result1.llf)
lr_pval_lm = stats.chi2.sf(lr_stat_lm, 1)

lr_stat_fb = 2 * (result3.llf - result1.llf)
lr_pval_fb = stats.chi2.sf(lr_stat_fb, 1)

print("\nlikelihood ratio tests, no COVID:")

print(f"\nLM vs Baseline:")
print(f"  LR statistic: {lr_stat_lm:.4f}")
print(f"  p-value:      {lr_pval_lm:.4f} {'***' if lr_pval_lm < 0.01 else '**' if lr_pval_lm < 0.05 else '*' if lr_pval_lm < 0.10 else 'ns'}")

print(f"\nFinBERT vs Baseline:")
print(f"  LR statistic: {lr_stat_fb:.4f}")
print(f"  p-value:      {lr_pval_fb:.4f} {'***' if lr_pval_fb < 0.01 else '**' if lr_pval_fb < 0.05 else '*' if lr_pval_fb < 0.10 else 'ns'}")

print("\nfull sample vs no COVID:")

df_full_clean = df_full[key_vars].dropna()
y_full = df_full_clean['decision_ordinal']
X3_full = df_full_clean[['inflation_gap', 'output_next', 'lagged_rate', 'finbert_sentiment_std']]
model3_full = OrderedModel(y_full, X3_full, distr='probit')
result3_full = model3_full.fit(method='bfgs', disp=False)

fb_coef_full = result3_full.params['finbert_sentiment_std']
fb_pval_full = result3_full.pvalues['finbert_sentiment_std']

print("\nFinBERT Coefficient:")
print(f"  Full sample (N={len(df_full_clean)}):   {fb_coef_full:+.4f} (p={fb_pval_full:.4f})")
print(f"  No COVID (N={len(df_clean)}):       {fb_coef:+.4f} (p={fb_pval:.4f})")

change_coef = ((fb_coef - fb_coef_full) / abs(fb_coef_full)) * 100
print(f"coefficient_change={change_coef:+.1f}%")
