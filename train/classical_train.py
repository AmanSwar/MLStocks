import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
import json
import os
from datetime import datetime
import pickle

warnings.filterwarnings("ignore")




def create_sequences(features, targets, seq_length=20):
    """Create sequences for time series prediction"""
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i : i + seq_length].flatten())  # Flatten sequence
        y.append(targets[i + seq_length])  # Predict next value
    return np.array(X), np.array(y)


def create_lagged_features(df, target_col, lags=[1, 2, 3, 5, 10, 20]):
    """Create lagged features for time series"""
    df_lagged = df.copy()
    for lag in lags:
        df_lagged[f"{target_col}_lag_{lag}"] = df_lagged[target_col].shift(lag)

    # Add rolling statistics
    for window in [5, 10, 20]:
        df_lagged[f"{target_col}_rolling_mean_{window}"] = (
            df_lagged[target_col].rolling(window).mean()
        )
        df_lagged[f"{target_col}_rolling_std_{window}"] = (
            df_lagged[target_col].rolling(window).std()
        )

    # Drop NaN values created by lagging
    df_lagged = df_lagged.dropna()
    return df_lagged




class ModelTrainer:
    def __init__(self, model_name, model, save_dir="./checkpoints_classical"):
        self.model_name = model_name
        self.model = model
        self.save_dir = save_dir
        self.metrics = {}
        self.predictions = None

    def train(self, X_train, y_train):
        """Train the model"""
        print(f"\nTraining {self.model_name}...")
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)

    def evaluate(self, X_train, y_train, X_val, y_val):
        """Evaluate model on train and validation sets"""
        train_pred = self.predict(X_train)
        val_pred = self.predict(X_val)

        self.metrics = {
            "train_mse": mean_squared_error(y_train, train_pred),
            "train_rmse": np.sqrt(mean_squared_error(y_train, train_pred)),
            "train_mae": mean_absolute_error(y_train, train_pred),
            "train_r2": r2_score(y_train, train_pred),
            "val_mse": mean_squared_error(y_val, val_pred),
            "val_rmse": np.sqrt(mean_squared_error(y_val, val_pred)),
            "val_mae": mean_absolute_error(y_val, val_pred),
            "val_r2": r2_score(y_val, val_pred),
        }

        self.predictions = {"train": train_pred, "val": val_pred}

        return self.metrics

    def save_model(self, run_dir):
        """Save model to disk"""
        model_path = os.path.join(run_dir, f"{self.model_name}_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"✓ Saved {self.model_name} model")




class ARIMAModel:
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.model = None
        self.model_fit = None

    def fit(self, X_train, y_train):
        """Fit ARIMA model - uses only target variable"""
        # ARIMA works on univariate time series
        self.model = ARIMA(y_train, order=self.order)
        self.model_fit = self.model.fit()

    def predict(self, X):
        """Make predictions"""
        n_periods = len(X)
        forecast = self.model_fit.forecast(steps=n_periods)
        return np.array(forecast)


class SARIMAXModel:
    def __init__(self, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.model_fit = None

    def fit(self, X_train, y_train):
        """Fit SARIMAX model"""
        self.model = SARIMAX(
            y_train, order=self.order, seasonal_order=self.seasonal_order
        )
        self.model_fit = self.model.fit(disp=False)

    def predict(self, X):
        """Make predictions"""
        n_periods = len(X)
        forecast = self.model_fit.forecast(steps=n_periods)
        return np.array(forecast)


class ExponentialSmoothingModel:
    def __init__(self, seasonal_periods=None):
        self.seasonal_periods = seasonal_periods
        self.model = None
        self.model_fit = None

    def fit(self, X_train, y_train):
        """Fit Exponential Smoothing model"""
        self.model = ExponentialSmoothing(
            y_train,
            seasonal_periods=self.seasonal_periods,
            trend="add",
            seasonal="add" if self.seasonal_periods else None,
        )
        self.model_fit = self.model.fit()

    def predict(self, X):
        """Make predictions"""
        n_periods = len(X)
        forecast = self.model_fit.forecast(steps=n_periods)
        return np.array(forecast)




def get_ml_models():
    """Get dictionary of classical ML models"""
    models = {
        # Linear Models
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        "SVR": SVR(kernel="rbf", C=1.0, epsilon=0.1),
    }
    return models


def get_time_series_models():
    """Get dictionary of time series models"""
    models = {
        "ARIMA": ARIMAModel(order=(2, 1, 2)),
        "SARIMAX": SARIMAXModel(order=(1, 1, 1), seasonal_order=(1, 1, 1, 5)),
        "ExpSmoothing": ExponentialSmoothingModel(seasonal_periods=5),
    }
    return models




def train_ml_models(X_train, y_train, X_val, y_val, save_dir):
    """Train all classical ML models"""
    models = get_ml_models()
    results = {}
    trained_models = {}

    print("\n" + "=" * 60)
    print("TRAINING CLASSICAL ML MODELS")
    print("=" * 60)

    for name, model in models.items():
        try:
            trainer = ModelTrainer(name, model, save_dir)
            trainer.train(X_train, y_train)
            metrics = trainer.evaluate(X_train, y_train, X_val, y_val)
            trainer.save_model(save_dir)

            results[name] = metrics
            trained_models[name] = trainer

            print(f"\n{name}:")
            print(
                f"  Train - RMSE: {metrics['train_rmse']:.6f}, MAE: {metrics['train_mae']:.6f}, R²: {metrics['train_r2']:.4f}"
            )
            print(
                f"  Val   - RMSE: {metrics['val_rmse']:.6f}, MAE: {metrics['val_mae']:.6f}, R²: {metrics['val_r2']:.4f}"
            )

        except Exception as e:
            print(f"\n{name}: FAILED - {str(e)}")
            results[name] = None

    return results, trained_models


def train_time_series_models(y_train, y_val, save_dir):
    """Train time series models (univariate)"""
    models = get_time_series_models()
    results = {}
    trained_models = {}

    print("\n" + "=" * 60)
    print("TRAINING TIME SERIES MODELS")
    print("=" * 60)

    for name, model in models.items():
        try:
            trainer = ModelTrainer(name, model, save_dir)
            # Time series models use only target variable
            trainer.train(None, y_train)

            # Make predictions
            train_pred = trainer.predict(np.arange(len(y_train)))
            val_pred = trainer.predict(np.arange(len(y_val)))

            # Calculate metrics
            metrics = {
                "train_mse": mean_squared_error(y_train, train_pred),
                "train_rmse": np.sqrt(mean_squared_error(y_train, train_pred)),
                "train_mae": mean_absolute_error(y_train, train_pred),
                "train_r2": r2_score(y_train, train_pred),
                "val_mse": mean_squared_error(y_val, val_pred),
                "val_rmse": np.sqrt(mean_squared_error(y_val, val_pred)),
                "val_mae": mean_absolute_error(y_val, val_pred),
                "val_r2": r2_score(y_val, val_pred),
            }

            trainer.metrics = metrics
            trainer.predictions = {"train": train_pred, "val": val_pred}
            trainer.save_model(save_dir)

            results[name] = metrics
            trained_models[name] = trainer

            print(f"\n{name}:")
            print(
                f"  Train - RMSE: {metrics['train_rmse']:.6f}, MAE: {metrics['train_mae']:.6f}, R²: {metrics['train_r2']:.4f}"
            )
            print(
                f"  Val   - RMSE: {metrics['val_rmse']:.6f}, MAE: {metrics['val_mae']:.6f}, R²: {metrics['val_r2']:.4f}"
            )

        except Exception as e:
            print(f"\n{name}: FAILED - {str(e)}")
            results[name] = None

    return results, trained_models




def plot_model_comparison(results, save_dir):
    """Plot comparison of all models"""
    # Filter out failed models
    results = {k: v for k, v in results.items() if v is not None}

    if not results:
        print("No successful models to plot")
        return

    models = list(results.keys())

    # Extract metrics
    train_rmse = [results[m]["train_rmse"] for m in models]
    val_rmse = [results[m]["val_rmse"] for m in models]
    train_mae = [results[m]["train_mae"] for m in models]
    val_mae = [results[m]["val_mae"] for m in models]
    train_r2 = [results[m]["train_r2"] for m in models]
    val_r2 = [results[m]["val_r2"] for m in models]

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # RMSE comparison
    ax = axes[0, 0]
    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width / 2, train_rmse, width, label="Train", alpha=0.8)
    ax.bar(x + width / 2, val_rmse, width, label="Validation", alpha=0.8)
    ax.set_xlabel("Model")
    ax.set_ylabel("RMSE")
    ax.set_title("Root Mean Squared Error Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # MAE comparison
    ax = axes[0, 1]
    ax.bar(x - width / 2, train_mae, width, label="Train", alpha=0.8)
    ax.bar(x + width / 2, val_mae, width, label="Validation", alpha=0.8)
    ax.set_xlabel("Model")
    ax.set_ylabel("MAE")
    ax.set_title("Mean Absolute Error Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # R² comparison
    ax = axes[1, 0]
    ax.bar(x - width / 2, train_r2, width, label="Train", alpha=0.8)
    ax.bar(x + width / 2, val_r2, width, label="Validation", alpha=0.8)
    ax.set_xlabel("Model")
    ax.set_ylabel("R² Score")
    ax.set_title("R² Score Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Validation RMSE sorted
    ax = axes[1, 1]
    sorted_idx = np.argsort(val_rmse)
    sorted_models = [models[i] for i in sorted_idx]
    sorted_rmse = [val_rmse[i] for i in sorted_idx]
    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(sorted_models)))
    ax.barh(range(len(sorted_models)), sorted_rmse, color=colors)
    ax.set_yticks(range(len(sorted_models)))
    ax.set_yticklabels(sorted_models)
    ax.set_xlabel("Validation RMSE")
    ax.set_title("Models Ranked by Validation RMSE")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "model_comparison.png"), dpi=300, bbox_inches="tight"
    )
    print(f"\n✓ Saved model comparison plot")
    plt.close()


def plot_predictions_comparison(trained_models, y_val, save_dir, n_samples=200):
    """Plot predictions from top models"""
    # Get top 5 models by validation RMSE
    model_scores = [
        (name, trainer.metrics["val_rmse"])
        for name, trainer in trained_models.items()
        if trainer.metrics is not None
    ]
    model_scores.sort(key=lambda x: x[1])
    top_models = model_scores[:5]

    fig, axes = plt.subplots(len(top_models), 1, figsize=(14, 4 * len(top_models)))
    if len(top_models) == 1:
        axes = [axes]

    plot_len = min(n_samples, len(y_val))

    for i, (name, score) in enumerate(top_models):
        ax = axes[i]
        trainer = trained_models[name]
        val_pred = trainer.predictions["val"]

        ax.plot(y_val[:plot_len], label="Actual", alpha=0.7, linewidth=1.5)
        ax.plot(val_pred[:plot_len], label="Predicted", alpha=0.7, linewidth=1.5)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Value")
        ax.set_title(f"{name} Predictions (Val RMSE: {score:.6f})")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "top_model_predictions.png"),
        dpi=300,
        bbox_inches="tight",
    )
    print(f"✓ Saved top model predictions plot")
    plt.close()


def create_results_table(results, save_dir):
    """Create and save results table"""
    # Filter out failed models
    results = {k: v for k, v in results.items() if v is not None}

    df = pd.DataFrame(results).T
    df = df.sort_values("val_rmse")

    print("\n" + "=" * 80)
    print("MODEL COMPARISON RESULTS (sorted by validation RMSE)")
    print("=" * 80)
    print(df.to_string())
    print("=" * 80)

    # Save to CSV
    df.to_csv(os.path.join(save_dir, "results_comparison.csv"))
    print(f"\n✓ Saved results table")

    return df


# ========================= ABLATION STUDIES =========================


def run_ablation_study(X_train, y_train, X_val, y_val, save_dir):
    """Run ablation studies on feature importance and model configurations"""

    print("\n" + "=" * 60)
    print("ABLATION STUDY: Feature Importance")
    print("=" * 60)

    # Train a Random Forest to get feature importances
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    # Get feature importances
    importances = rf_model.feature_importances_

    # Test with different number of features
    n_features_list = [10, 20, 50, 100, X_train.shape[1]]
    ablation_results = {}

    for n_features in n_features_list:
        if n_features > X_train.shape[1]:
            continue

        # Select top n features
        top_indices = np.argsort(importances)[-n_features:]
        X_train_subset = X_train[:, top_indices]
        X_val_subset = X_val[:, top_indices]

        # Train model with subset
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train_subset, y_train)

        val_pred = model.predict(X_val_subset)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        mae = mean_absolute_error(y_val, val_pred)
        r2 = r2_score(y_val, val_pred)

        ablation_results[f"Top_{n_features}_features"] = {
            "val_rmse": rmse,
            "val_mae": mae,
            "val_r2": r2,
        }

        print(
            f"\nTop {n_features} features: RMSE={rmse:.6f}, MAE={mae:.6f}, R²={r2:.4f}"
        )

    # Save ablation results
    ablation_df = pd.DataFrame(ablation_results).T
    ablation_df.to_csv(os.path.join(save_dir, "ablation_feature_importance.csv"))

    # Plot ablation results
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(ablation_results))
    ax.plot(
        list(ablation_results.keys()),
        [v["val_rmse"] for v in ablation_results.values()],
        "o-",
        linewidth=2,
        markersize=8,
    )
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("Validation RMSE")
    ax.set_title("Ablation Study: Impact of Feature Count on Performance")
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ablation_feature_importance.png"), dpi=300)
    plt.close()

    print(f"\n✓ Saved ablation study results")

    return ablation_results


# ========================= MAIN EXECUTION =========================


def main():
    from data_prep.data_clean import clean_indicator
    from data_prep.data_load import prepare_data

    # Configuration
    config = {
        "data_path": "/home/aman/code/ml_fr/ml_stocks/data/NIFTY_5_years.csv",
        "seq_length": 20,
        "train_split": 0.8,
        "save_dir": "./checkpoints_classical",
        "target_col": "Daily_Return",
    }

    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(config["save_dir"], f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"CLASSICAL ML & TIME SERIES MODEL TRAINING")
    print(f"{'='*60}")
    print(f"Save directory: {save_dir}")
    print(f"{'='*60}\n")

    # Load and prepare data
    print("Loading data...")
    load_df = prepare_data(config["data_path"])
    df = clean_indicator(load_df)

    target_col = config["target_col"]
    feature_cols = [col for col in df.columns if col != target_col]

    # Split data
    train_size = int(len(df) * config["train_split"])
    train_df = df[:train_size]
    val_df = df[train_size:]

    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Number of features: {len(feature_cols)}")

    # Prepare features for ML models (with sequences)
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_df[feature_cols].values)
    val_features = scaler.transform(val_df[feature_cols].values)

    train_targets = train_df[target_col].values
    val_targets = val_df[target_col].values

    # Create sequences
    X_train, y_train = create_sequences(
        train_features, train_targets, config["seq_length"]
    )
    X_val, y_val = create_sequences(val_features, val_targets, config["seq_length"])

    print(f"\nSequence shape: {X_train.shape}")
    print(f"Target shape: {y_train.shape}")

    # Save config
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # Train ML models
    ml_results, ml_models = train_ml_models(X_train, y_train, X_val, y_val, save_dir)

    # Train time series models (using non-sequenced data)
    ts_results, ts_models = train_time_series_models(
        train_targets[config["seq_length"] :],  # Align with ML model targets
        val_targets[config["seq_length"] :],
        save_dir,
    )

    # Combine results
    all_results = {**ml_results, **ts_results}
    all_models = {**ml_models, **ts_models}

    # Create visualizations
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)

    plot_model_comparison(all_results, save_dir)
    plot_predictions_comparison(all_models, y_val, save_dir)
    results_df = create_results_table(all_results, save_dir)

    # Run ablation study
    ablation_results = run_ablation_study(X_train, y_train, X_val, y_val, save_dir)

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"Results saved to: {save_dir}")
    print(f"{'='*60}\n")

    # Print best model
    best_model = results_df.index[0]
    best_rmse = results_df.loc[best_model, "val_rmse"]
    print(f"🏆 Best Model: {best_model}")
    print(f"   Validation RMSE: {best_rmse:.6f}")
    print(f"   Validation MAE: {results_df.loc[best_model, 'val_mae']:.6f}")
    print(f"   Validation R²: {results_df.loc[best_model, 'val_r2']:.4f}")

    return all_results, all_models, save_dir


if __name__ == "__main__":
    results, models, save_dir = main()

    print("\n" + "=" * 60)
    print("All models trained successfully!")
    print("Check the save directory for:")
    print("  - Model comparison plots")
    print("  - Results CSV")
    print("  - Saved model files (.pkl)")
    print("  - Ablation study results")
    print("=" * 60)
