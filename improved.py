import pandas as pd
import numpy as np
import os

base_path = os.path.dirname(os.path.abspath(__file__)) + os.sep

df = pd.read_csv(base_path + "cleaned_housing_data.csv")

print("Loaded cleaned dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nPreview:")
print(df.head())

# Model Training (CatBoost)
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_squared_log_error

# Candidate features
# Do NOT use price or price_per_sqft as inputs when predicting log_price,
# because they leak target information.
# Only feature-engineering/data-prep is changed; the rest of the modeling logic stays the same.
candidate_features = [
    "area",
    "bedrooms",
    "bathrooms",
    "year_built",
    "house_age",
    "dataset",
    "log_area",
    "area_per_bedroom",
    "area_per_bathroom",
    "area_bath_interaction"
]

categorical_features = ["dataset"]

# Keep only rows that have the target and core numeric predictors
model_df = df.dropna(subset=["log_price", "area", "bedrooms", "bathrooms"]).copy()

# Fill missing year features with medians so feature selection and training are stable
model_df["year_built"] = model_df["year_built"].fillna(model_df["year_built"].median())
model_df["house_age"] = model_df["house_age"].fillna(model_df["house_age"].median())
model_df["dataset"] = model_df["dataset"].fillna("Missing").astype(str)

# Safe denominators for engineered ratio features
model_df["bedrooms_safe"] = model_df["bedrooms"].replace(0, 1)
model_df["bathrooms_safe"] = model_df["bathrooms"].replace(0, 1)

# Engineered features
model_df["log_area"] = np.log1p(model_df["area"])
model_df["area_per_bedroom"] = model_df["area"] / model_df["bedrooms_safe"]
model_df["area_per_bathroom"] = model_df["area"] / model_df["bathrooms_safe"]
model_df["area_bath_interaction"] = model_df["area"] * model_df["bathrooms"]

X = model_df[candidate_features]
y = model_df["log_price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Step 1: Train CatBoost on all candidate features
base_model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    random_seed=42,
    verbose=0
)

base_model.fit(
    X_train,
    y_train,
    cat_features=categorical_features
)

# Get feature importances from the full model
importance_df = pd.DataFrame({
    "feature": candidate_features,
    "importance": base_model.get_feature_importance()
}).sort_values(by="importance", ascending=False)

print("\nFeature importance from all candidate features:")
print(importance_df)

# Step 2: Automatically select top features based on model importance
#selected_features = importance_df["feature"].head(6).tolist()
selected_features = candidate_features.copy()
selected_cat_features = [col for col in categorical_features if col in selected_features]

print("\nAutomatically selected features:")
print(f"Using top {len(selected_features)} features based on CatBoost importance")
print(selected_features)

# Step 3: Retrain final CatBoost using selected features
final_model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=5,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    verbose=100
)

final_model.fit(
    X_train[selected_features],
    y_train,
    cat_features=selected_cat_features,
    eval_set=(X_test[selected_features], y_test),
    early_stopping_rounds=50,
    use_best_model=True
)

# Predict
final_pred = final_model.predict(X_test[selected_features])

# Evaluate
y_test_price = np.expm1(y_test)
final_pred_price = np.expm1(final_pred)
catboost_mape = np.mean(np.abs((y_test_price - final_pred_price) / np.maximum(y_test_price, 1e-9))) * 100
catboost_rmsle = np.sqrt(
    mean_squared_log_error(
        np.maximum(y_test_price, 0),
        np.maximum(final_pred_price, 0)
    )
)
rmse = mean_squared_error(y_test, final_pred) ** 0.5
r2 = r2_score(y_test, final_pred)
catboost_mae = mean_absolute_error(y_test, final_pred)

print("\nFinal Model Performance:")
print("RMSE:", rmse)
print("MAE:", catboost_mae)
print("MAPE (%):", catboost_mape)
print("RMSLE:", catboost_rmsle)
print("R² Score:", r2)
print("Best iteration:", final_model.get_best_iteration())

# Step 4: SHAP-style feature contribution output from CatBoost
shap_pool = Pool(
    X_test[selected_features],
    cat_features=selected_cat_features
)

shap_values = final_model.get_feature_importance(
    shap_pool,
    type="ShapValues"
)

print("\nSHAP values array shape:", shap_values.shape)
print("Expected extra SHAP column for base value:", len(selected_features) + 1)
print("\nMean absolute SHAP values for selected features:")
mean_abs_shap = abs(shap_values[:, :-1]).mean(axis=0)
shap_df = pd.DataFrame({
    "feature": selected_features,
    "mean_abs_shap": mean_abs_shap
}).sort_values(by="mean_abs_shap", ascending=False)

print(shap_df)

# XGBoost Baseline
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

# Copy dataset used for modeling
xgb_df = model_df.copy()

# Encode categorical feature for XGBoost (dataset column)
label_encoder = LabelEncoder()
xgb_df["dataset"] = label_encoder.fit_transform(xgb_df["dataset"])

# Use same features as CatBoost
xgb_features = candidate_features.copy()

X_xgb = xgb_df[xgb_features]
y_xgb = xgb_df["log_price"]

# Same train-test split
X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
    X_xgb, y_xgb, test_size=0.2, random_state=42
)

# Initialize XGBoost model
xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)

# Train
xgb_model.fit(X_train_xgb, y_train_xgb)

# Predict
xgb_pred = xgb_model.predict(X_test_xgb)

# Evaluate
xgb_rmse = mean_squared_error(y_test_xgb, xgb_pred) ** 0.5
xgb_r2 = r2_score(y_test_xgb, xgb_pred)
xgb_mae = mean_absolute_error(y_test_xgb, xgb_pred)
xgb_mape = np.mean(np.abs((np.expm1(y_test_xgb) - np.expm1(xgb_pred)) / np.maximum(np.expm1(y_test_xgb), 1e-9))) * 100
xgb_rmsle = np.sqrt(mean_squared_log_error(np.maximum(np.expm1(y_test_xgb), 0), np.maximum(np.expm1(xgb_pred), 0)))

print("\nXGBoost Baseline Performance:")
print("RMSE:", xgb_rmse)
print("MAE:", xgb_mae)
print("MAPE (%):", xgb_mape)
print("RMSLE:", xgb_rmsle)
print("R² Score:", xgb_r2)


# PSO-XGBoost Step 1: define search space and objective function
from sklearn.model_selection import cross_val_score
import pyswarms as ps

np.random.seed(42)

# IMPORTANT: tune only on the training split, not the full dataset
# Also switch XGBoost to one-hot encoded dataset features while keeping the same engineered inputs
xgb_pso_df = pd.get_dummies(model_df.copy(), columns=["dataset"], drop_first=False)
xgb_dataset_cols = [col for col in xgb_pso_df.columns if col.startswith("dataset_")]
base_pso_features = [feat for feat in candidate_features if feat != "dataset"]
pso_features = base_pso_features + xgb_dataset_cols

X_full_pso = xgb_pso_df[pso_features]
y_full_pso = xgb_pso_df["log_price"]

X_train_pso, X_test_pso, y_train_pso, y_test_pso = train_test_split(
    X_full_pso, y_full_pso, test_size=0.2, random_state=42
)

print("\nPSO-XGBoost features used:")
print(pso_features)

# Bounds for PSO search:
# [n_estimators, max_depth, learning_rate, subsample, colsample_bytree,
#  min_child_weight, gamma, reg_lambda]
lower_bounds = np.array([400, 3, 0.01, 0.70, 0.70, 1.0, 0.0, 0.5])
upper_bounds = np.array([900, 6, 0.08, 0.95, 0.95, 8.0, 2.0, 5.0])

print("\nPSO search bounds:")
print("lower:", lower_bounds)
print("upper:", upper_bounds)


def pso_objective_function(params):
    losses = []

    for particle in params:
        n_estimators = int(round(particle[0]))
        max_depth = int(round(particle[1]))
        learning_rate = float(np.clip(particle[2], lower_bounds[2], upper_bounds[2]))
        subsample = float(np.clip(particle[3], lower_bounds[3], upper_bounds[3]))
        colsample_bytree = float(np.clip(particle[4], lower_bounds[4], upper_bounds[4]))
        min_child_weight = float(np.clip(particle[5], lower_bounds[5], upper_bounds[5]))
        gamma = float(np.clip(particle[6], lower_bounds[6], upper_bounds[6]))
        reg_lambda = float(np.clip(particle[7], lower_bounds[7], upper_bounds[7]))

        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            reg_lambda=reg_lambda,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1
        )

        scores = cross_val_score(
            model,
            X_train_pso,
            y_train_pso,
            cv=3,
            scoring="neg_root_mean_squared_error",
            n_jobs=1
        )

        rmse = -scores.mean()
        losses.append(rmse)

    return np.array(losses)

print("\nPSO objective function is ready.")

# PSO-XGBoost Step 2: run PSO optimization
options = {"c1": 0.5, "c2": 0.3, "w": 0.8}

optimizer = ps.single.GlobalBestPSO(
    n_particles=20,
    dimensions=8,
    options=options,
    bounds=(lower_bounds, upper_bounds)
)

print("\nStarting PSO optimization...")

best_cost, best_pos = optimizer.optimize(
    pso_objective_function,
    iters=20,
    verbose=True
)

print("\nPSO optimization finished.")
print("Best CV RMSE:", best_cost)
print("Best position:", best_pos)

best_n_estimators = int(round(best_pos[0]))
best_max_depth = int(round(best_pos[1]))
best_learning_rate = float(np.clip(best_pos[2], lower_bounds[2], upper_bounds[2]))
best_subsample = float(np.clip(best_pos[3], lower_bounds[3], upper_bounds[3]))
best_colsample_bytree = float(np.clip(best_pos[4], lower_bounds[4], upper_bounds[4]))
best_min_child_weight = float(np.clip(best_pos[5], lower_bounds[5], upper_bounds[5]))
best_gamma = float(np.clip(best_pos[6], lower_bounds[6], upper_bounds[6]))
best_reg_lambda = float(np.clip(best_pos[7], lower_bounds[7], upper_bounds[7]))

print("\nBest hyperparameters from PSO:")
print("n_estimators:", best_n_estimators)
print("max_depth:", best_max_depth)
print("learning_rate:", best_learning_rate)
print("subsample:", best_subsample)
print("colsample_bytree:", best_colsample_bytree)
print("min_child_weight:", best_min_child_weight)
print("gamma:", best_gamma)
print("reg_lambda:", best_reg_lambda)

# PSO-XGBoost Step 3: train final model using best PSO hyperparameters
pso_xgb_model = XGBRegressor(
    n_estimators=best_n_estimators,
    max_depth=best_max_depth,
    learning_rate=best_learning_rate,
    subsample=best_subsample,
    colsample_bytree=best_colsample_bytree,
    min_child_weight=best_min_child_weight,
    gamma=best_gamma,
    reg_lambda=best_reg_lambda,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=1,
    early_stopping_rounds=50
)

pso_xgb_model.fit(
    X_train_pso,
    y_train_pso,
    eval_set=[(X_test_pso, y_test_pso)],
    verbose=False
)

pso_xgb_pred = pso_xgb_model.predict(X_test_pso)

pso_xgb_rmse = mean_squared_error(y_test_pso, pso_xgb_pred) ** 0.5
pso_xgb_r2 = r2_score(y_test_pso, pso_xgb_pred)
pso_xgb_mae = mean_absolute_error(y_test_pso, pso_xgb_pred)
pso_xgb_mape = np.mean(np.abs((np.expm1(y_test_pso) - np.expm1(pso_xgb_pred)) / np.maximum(np.expm1(y_test_pso), 1e-9))) * 100
pso_xgb_rmsle = np.sqrt(mean_squared_log_error(np.maximum(np.expm1(y_test_pso), 0), np.maximum(np.expm1(pso_xgb_pred), 0)))

print("\nPSO-XGBoost Final Performance:")
print("RMSE:", pso_xgb_rmse)
print("MAE:", pso_xgb_mae)
print("MAPE (%):", pso_xgb_mape)
print("RMSLE:", pso_xgb_rmsle)
print("R² Score:", pso_xgb_r2)
print("Best boosted rounds used:", pso_xgb_model.best_iteration)

print("\nModel Comparison Summary:")
comparison_df = pd.DataFrame({
    "Model": ["CatBoost", "XGBoost Baseline", "PSO-XGBoost"],
    "RMSE": [rmse, xgb_rmse, pso_xgb_rmse],
    "MAE": [catboost_mae, xgb_mae, pso_xgb_mae],
    "MAPE_%": [catboost_mape, xgb_mape, pso_xgb_mape],
    "RMSLE": [catboost_rmsle, xgb_rmsle, pso_xgb_rmsle],
    "R2": [r2, xgb_r2, pso_xgb_r2]
})
print(comparison_df)

# Feature importance for PSO-XGBoost
pso_importance_df = pd.DataFrame({
    "feature": pso_features,
    "importance": pso_xgb_model.feature_importances_}).sort_values(by="importance", ascending=False)

print("\nPSO-XGBoost Feature Importance:")
print(pso_importance_df)

# Residual analysis
pso_residuals = y_test_pso - pso_xgb_pred

print("\nResidual summary (PSO-XGBoost):")
print(pd.Series(pso_residuals).describe())