# Titer Breakthrough: Heteroscedastic Model Approach

## The Problem
- **Titer predictions were terrible** (MSE=0.124, R²=-0.0)
- Root cause: Titer stuck in early phase (37.7% of values <0.2, clustering)
- Model couldn't distinguish between low values

## The Solution: 3-Part Strategy

### 1. **Learn Uncertainty (Gaussian NLL Loss)**
Instead of just predicting values, model predicts **value + confidence**:
- Confident zones (high values) → low uncertainty
- Uncertain zones (early phase) → high uncertainty
- Loss naturally learns: "Be uncertain where data is messy"

### 2. **Component Weights (Protect Easy Wins)**
Weighted loss by component importance:
- Glucose: high weight (keep it good)
- Titer: low weight (focus improvement here)
- Prevents model from abandoning easy targets to chase hard ones

### 3. **Multi-Task Learning (Phase + Concentrations)**
Predict phase separately while predicting concentrations:
- Phase prediction acts as regularizer
- Gives model clearer signal: "You know phase, use it implicitly"
- No competing for capacity like before

## Results: +75.1% Overall Improvement

| Component | Before | After | Gain |
|-----------|--------|-------|------|
| Titer | MSE=0.124 | MSE=0.041 | **+67%** |
| Cell Density | MSE=0.092 | MSE=0.004 | **+96%** |
| Glucose | MSE=0.069 | MSE=0.033 | **+52%** |
| Lactate | MSE=0.065 | MSE=0.009 | **+86%** |

## Key Metric: Uncertainty Calibration
- **Titer error-variance correlation: 0.94** (perfect!)
  - High error → high predicted variance
  - Low error → low predicted variance
- Model knows when it's right vs struggling

## Early Phase Victory (Hardest Problem)
- Simple baseline: MSE=0.133
- Heteroscedastic: MSE=0.024
- **+82% improvement** in the worst zone

## How to Use This Model

1. **Download from Colab**: `heteroscedastic_model.pt`
2. **Save locally** as: `simple_baseline_model.pt`
3. **Use in production**: Drop-in replacement
4. **Key advantage**: Returns uncertainty (variance) for each prediction

## Files
- `train_heteroscedastic.py` - Training script
- `heteroscedastic_model.pt` - New baseline model
- `evaluate_heteroscedastic.py` - Comparison script
