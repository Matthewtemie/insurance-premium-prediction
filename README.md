# Insurance Premium Prediction

A supervised learning project predicting individual health insurance premiums
from demographic and lifestyle features. Compares linear regression against
a tuned XGBoost model, and includes detailed error analysis identifying
where and why the model fails.

## Problem

Health insurers need to price policies accurately: over-price and customers
leave, under-price and claims exceed revenue. This project builds a
regression model to predict annual premium amount from a set of customer
features, then investigates where the model's predictions can and cannot
be trusted.

## Dataset

- **Source:** `premiums_with_life_style.xlsx` [ADJUST: add source if from Kaggle or a course]
- **Size:** ~15,000 customer records [ADJUST if different]
- **Target variable:** `annual_premium_amount` (continuous, in local currency)
- **Feature types:**
  - Numerical: age, number of dependants, income (level and lakhs),
    insurance plan tier, normalized risk score, lifestyle risk score
  - Categorical (one-hot encoded): gender, region, marital status,
    BMI category, smoking status, employment status

**Total features after encoding:** 18

## Approach

### 1. Data Cleaning

- Identified 8,882 missing values in `normalized_risk_score` (roughly 60% of
  that column) — imputed with the column median rather than dropping rows,
  to preserve dataset size for training.
- Verified no infinite values, correct numeric dtypes across all columns.
- One-hot encoded all categorical variables; dropped one level per category
  to avoid the dummy variable trap.

### 2. Multicollinearity Check

Ran Variance Inflation Factor (VIF) analysis on all numeric features.
All VIF values under [ADJUST — fill in from your VIF output, if you have it],
indicating no serious multicollinearity that would destabilize a linear
regression's coefficients.

### 3. Train / Test Split

Standard 80 / 20 split with a fixed random seed for reproducibility.
Final test set: 2,958 records.

### 4. Model Comparison

Trained and evaluated two models on the same split:

- **Linear regression** — baseline, chosen for interpretability and to
  establish a floor for what a simple model can do.
- **XGBoost regressor** — gradient-boosted trees, hyperparameters tuned via
  `RandomizedSearchCV`. Best parameters: `n_estimators=50`, `max_depth=5`,
  `learning_rate=0.1`.

### 5. Evaluation Discipline

Beyond a single MSE number, I evaluated the models on:

- Distribution of residuals (visual check for bias and shape)
- Percentage of "extreme errors" — predictions off by more than 10%
- Feature-level analysis of where errors concentrate

## Results

### XGBoost Performance

| Metric | Value |
|--------|-------|
| Test MSE | [ADJUST — fill in] |
| Extreme error rate (>10% off) | 7.2% |
| Extreme errors: over-predictions | 168 |
| Extreme errors: under-predictions | 45 |

The residual distribution is roughly Gaussian, centered near zero with a
mild right skew — indicating no systematic bias but the presence of a
subset of customers whose true premium is higher than the model expects.

### Feature Importance

XGBoost's feature importances revealed strong dominance by two features:

| Feature | Importance |
|---------|-----------|
| age | ~0.48 |
| insurance_plan | ~0.43 |
| normalized_risk_score | ~0.03 |
| bmi_category_Obesity | ~0.02 |
| smoking_status_Regular | ~0.02 |
| _(all others)_ | < 0.01 |

Two features carry roughly 91% of the model's decision-making. This is a
double-edged finding: the model has correctly identified the biggest
signals, but has little to fall back on when those signals don't
differentiate customers well.

## Error Analysis Findings

Three concrete findings from investigating the 213 extreme-error cases:

**1. Failures are 4x more likely to be over-predictions than under-predictions**
(168 vs 45). When the model is wrong, it's usually quoting too high.

**2. Errors concentrate in unmarried, basic-plan customers with lower-than-average
risk profiles.** Feature comparison between extreme errors and the full test set:

- `insurance_plan` averages 0.08 in errors vs 0.44 in full set (−0.36)
- `marital_status_Unmarried` averages 0.76 in errors vs 0.42 in full set (+0.34)
- `age`, `normalized_risk_score`, `bmi_overweight`, `dependants`,
  `smoking_regular`, `self_employed` — all under-represented in errors

**3. The failure profile is the "healthy simple case" group**, not high-risk
outliers. The model handles high-premium and premium-plan customers well; it
struggles to differentiate among cheaper basic-plan customers where the two
dominant features (age, plan) give weaker signal.

## Interpretation

The model behaves as follows: for a customer with strong age and plan
signals (older customer on premium plan, or younger customer on basic plan),
the prediction is confident and usually accurate. For customers where those
signals are weak or contradictory (young customer on basic plan with
mid-range risk), the model has little to distinguish them from each other
and falls back on a middle-of-the-road prediction — which is often too high.

This is a data-completeness problem more than a modeling problem. The
features available (demographics, BMI, smoking, employment) capture the
broad strokes of insurance risk but miss the individual-level factors that
differentiate similar customers.

## Recommendations for Future Work

Based on this analysis, three directions could improve the model:

1. **Test whether age and insurance_plan are truly essential.** Train the
   model with those two features removed. If MSE stays similar, other
   features contain signal the current model isn't using. If MSE collapses,
   those two are foundational and the rest of the effort should focus
   elsewhere.

2. **Consider a segmented approach:** one model for premium-plan customers
   (where the current model excels), one for basic-plan customers (where
   it struggles). This "different models for different regimes" pattern is
   standard in insurance pricing.

3. **Add features that capture individual-level risk:** claim history,
   occupation risk category, hobbies, family medical history. Even a
   binary "has filed prior claim" feature would likely help the basic-plan
   segment where current features don't discriminate well.

## What I Learned

- **Averages hide the story.** A single MSE number missed that 79% of
  extreme errors are over-predictions, which is operationally very
  different from balanced errors.
- **Feature importance can be too concentrated.** When two features carry
  90% of the model, the model is fragile — it works when those signals
  are strong and fails when they aren't.
- **Data cleaning matters more than model choice.** The initial SVD error
  from missing values, and the subsequent decision to impute rather than
  drop, shaped everything downstream. Model comparison is straightforward;
  data decisions require judgment.

## Tech Stack

- Python 3.13
- pandas, numpy — data manipulation
- scikit-learn — train/test split, linear regression, hyperparameter search
- xgboost — final model
- statsmodels — multicollinearity (VIF) diagnostics
- matplotlib, seaborn — visualization
- Jupyter Notebook — exploratory analysis environment

## Reproducing This Analysis

```bash
# Clone the repository
git clone https://github.com/Matthewtemie/insurance-premium-prediction.git
cd insurance-premium-prediction

# Set up a virtual environment
python3 -m venv .venv
source .venv/bin/activate           # on Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook project_one.ipynb
```

## Project Structure
