# GSTN Presents Analytics Hackathon on Developing a Predictive Model in GST

## Problem Statement
The Hackathon spans 45 days from the start of registration to the final submission date for developed prototypes. Participants will receive a dataset containing 900,000 records with around 21 attributes each. This data is anonymized and labeled, including trained, validated, and non-validated datasets.

Given a dataset \( D \) with:
- \( D_{\text{train}} \): Training data matrix \( R(m \times n) \)
- \( D_{\text{test}} \): Test data matrix \( R(m1 \times n) \)
- \( Y_{\text{train}} \): Target variable matrix \( R(m \times 1) \)
- \( Y_{\text{test}} \): Target variable matrix \( R(m1 \times 1) \)

The goal is to build a predictive model \( F_{\theta}(X) \rightarrow Y_{\text{pred}} \) that accurately predicts \( Y_i \) for new inputs \( X_i \).

## Operations Performed
1. **Data Cleaning**
2. **Feature Scaling**
   - Distribution analysis
   - Scaling techniques: MinMax and Robust Scaling
3. **Feature Engineering**
   - Forward Feature Selection
   - Pearson Correlation Coefficient
   - Recursive Feature Elimination
4. **Model Training**
   - Baseline models: Decision Tree, Random Forest
   - Ensemble techniques: ADA-Boosting ,  Gradient Boosting
5. **Hyperparameter Tuning**
   - RandomSearch CV
   - Bayesian Optimization
6. **Results and Conclusion**
   - The AdaBoost algorithm with a runtime of 23 seconds and RandomSearchCV parameters achieved an **accuracy of 97.5%.**

## References
- For detailed analysis and all steps, refer to `notebooks/model_building.ipynb`.
- For model code, refer to `scripts/model.py`.
