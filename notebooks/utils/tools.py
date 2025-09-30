from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Union
import pandas as pd
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from scipy import stats
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    average_precision_score,
    brier_score_loss,
    PrecisionRecallDisplay,
    make_scorer
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.inspection import permutation_importance
import shap


from pathlib import Path


## ---------------------- ##
## CLASSES
## ---------------------- ##
@dataclass
class DataProfiler:
    """
    Lightweight wrapper around ydata-profiling to generate  a report.

    Usage:
        profiler = DataProfiler.from_csv("/path/to/data.csv", sep=",", title="My Report")
        profiler.generate_report()
        profiler.to_html("report.html")
        # In notebooks:
        # profiler.to_notebook()
    """

    df: pd.DataFrame
    title: str = "Data Profiling Report"
    minimal: bool = False
    explorative: bool = True
    # ydata-profiling accepts extra kwargs like correlations, interactions, etc.
    profile_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Will be created by generate_report; excluded from __init__/repr
    profile: Optional[ProfileReport] = field(init=False, default=None, repr=False)

    def generate_report(self) -> ProfileReport:
        """Generate the profiling report and store it on self.profile."""
        print("Generating profiling report...")
        # Merge core switches with any advanced options passed via profile_kwargs
        kwargs = dict(
            title=self.title,
            minimal=self.minimal,
            explorative=self.explorative,
        )
        kwargs.update(self.profile_kwargs)

        self.profile = ProfileReport(self.df, **kwargs)
        print("Report generated successfully.")
        return self.profile

    def to_html(self, path: str = "report.html") -> None:
        """Export the profiling report to an HTML file."""
        if self.profile is None:
            raise ValueError("Please generate the report first with generate_report().")
        self.profile.to_file(path)
        print(f"Report saved to {path}")

    def to_notebook(self):
        """Display the profiling report inside a Jupyter Notebook."""
        if self.profile is None:
            raise ValueError("Please generate the report first with generate_report().")
        print("Displaying report in Jupyter Notebook...")
        return self.profile.to_notebook_iframe()
    
@dataclass
class FraudPreprocessor:
    """
    Data preprocessing & feature engineering.

    What it does:
      - Parses date column and creates date-based features (year, month).
      - Optionally creates simple behavioral ratios (safe divide).
      - Applies safe log1p on selected positive-skewed features.
      - Scales numeric features and one-hot encodes categoricals.

    Customize:
      - Pass column names according to your dataset schema.
      - Toggle which engineered features to create.
    """ 
    
    df: pd.DataFrame
    
    
    tol: float = 1e-16
    random_state: int = 42
    
    
    
    date_col: str = "data_rif"
    user_col: str = "userid"
    target_col: str = "TARGET"
    
    numeric_cols: List[str] = field(default_factory=lambda: [
        "age",
        "account_balance",
        "num_trx_cd", "num_trx_cc", "num_trx_cp",
        "num_mov_conto",
        "sum_mov_conto_pos", "sum_mov_conto_neg",
        "num_prodotti",
        "f2", "f3", "f4", "f5", "f6", "f7",
    ])
    
    categorical_cols: List[str] = field(default_factory=lambda: [
        "profession", "region"
    ])
    
    create_date_features: bool = False
    create_num_features: bool = True
    apply_log: bool = False       
    apply_log_to: Optional[List[str]] = field(default_factory=lambda: [
        "account_balance", "sum_mov_conto_pos", "sum_mov_conto_neg",
        "num_trx_cd", "num_trx_cc", "num_trx_cp", "num_mov_conto"
    ])
    
    def __post_init__(self):
        self.df = self.df.copy() 
    
    
    def _make_feature_engineering(self) -> pd.DataFrame:
        """Create engineered columns in a copy of df."""
        df = self.df.copy()

        # Parse date and create features
        if self.create_date_features and self.date_col in df.columns:
            print("Creating date-based features (year, month).")
            df[self.date_col] = pd.to_datetime(df[self.date_col], errors="coerce")
            df["year"] = df[self.date_col].dt.year.astype("Int64")
            df["month"] = df[self.date_col].dt.month.astype("Int16")
            # Consider dropping raw date if desired:
            # df = df.drop(columns=[self.date_col])
            self.numeric_cols = list(dict.fromkeys(self.numeric_cols + ["year", "month"]))
            
        if self.create_num_features:
            print("Creating New features:")
            
            if {"sum_mov_conto_pos", "sum_mov_conto_neg"}.issubset(df.columns):
                print(" ratio_pos_neg")
                df['ratio_pos_neg'] = df['sum_mov_conto_pos'] / (abs(df['sum_mov_conto_neg']) + self.tol)
                self.numeric_cols = list(dict.fromkeys(self.numeric_cols + ["ratio_pos_neg"]))
                
            if {"sum_mov_conto_pos", "sum_mov_conto_neg", 'account_balance'}.issubset(df.columns):
                print(" ratio_balance_per_sum")
                df['ratio_balance_per_sum'] = df['account_balance'] / (df['sum_mov_conto_pos'] + abs(df['sum_mov_conto_neg']) + self.tol)
                self.numeric_cols = list(dict.fromkeys(self.numeric_cols + ["ratio_balance_per_sum"]))
                
            if {"num_mov_conto", 'account_balance'}.issubset(df.columns):
                print(" ratio_balance_per_mov")
                df['ratio_balance_per_mov'] = df['account_balance'] / (df['num_mov_conto'] + self.tol)
                
                
                print( " trx_per_balance")
                mean_balance = df.groupby("userid")["account_balance"].transform("mean").abs() 
                df["trx_per_balance"] = df["num_mov_conto"] / (mean_balance + self.tol)
                self.numeric_cols = list(dict.fromkeys(self.numeric_cols + ["ratio_balance_per_mov", "trx_per_balance"]))

            if {"num_prodotti", 'account_balance'}.issubset(df.columns):
                print(" ratio_balance_per_prod")
                df['ratio_balance_per_prod'] = df['account_balance'] / (df['num_prodotti'] + self.tol)
                self.numeric_cols = list(dict.fromkeys(self.numeric_cols + ["ratio_balance_per_prod"]))    

            if {"num_prodotti"}.issubset(df.columns):
                print(" cat_prodotti")
                df['cat_prodotti'] = pd.cut(
                    df['num_prodotti'],
                    bins=[0, 3, 9, np.inf],
                    labels=['low ','medium','high']
                )
                self.categorical_cols = list(dict.fromkeys(self.categorical_cols + ["cat_prodotti"]))              
            
            if {"age"}.issubset(df.columns):
                print(" anomalous_age")
                df = add_anomalous_age(df)
                self.categorical_cols = list(dict.fromkeys(self.categorical_cols + ["anomalous_age"]))
                
                
            return df
        
    
    def split_data(self, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

        
        self.df = self._make_feature_engineering()
        
        if self.target_col not in self.df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in DataFrame.")

        X = self.df.drop(columns=[self.target_col, self.user_col, self.date_col])
        y = self.df[self.target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        print(f"Data split into train and test sets with test size {test_size}.")
        return X_train, X_test, y_train, y_test



@dataclass
class ModelTrain:
    """
    End-to-end training and evaluation for sklearn pipelines.

    What it does:
      - Builds a preprocessing + estimator pipeline (Logistic Regression, Random Forest, Decision Tree).
      - Optionally performs RandomizedSearchCV for hyperparameter tuning.
      - Tunes the decision threshold by maximizing F-beta on a validation grid.
      - Generates a standardized evaluation report:
          * Accuracy
          * Per-class precision/recall/F1
          * Average Precision (class 1)
          * Brier score
          * Confusion matrix
          * Predicted probabilities
      - Provides visualizations:
          * Precision–Recall curve
          * Decision Tree plot (if applicable)
          * Model-based feature importance
          * Permutation importance
          * SHAP-based explanations (waterfall, beeswarm, global bar plot)

    Customize:
      - Pass your own ColumnTransformer as preprocessor or let the class build one internally.
      - Choose the estimator via `model_name`: "logreg", "rf", or "dt".
      - Control randomness with `random_state` for reproducibility.
    """

    X_train: Union[np.ndarray, pd.DataFrame]
    X_test: Union[np.ndarray, pd.DataFrame]
    y_train: Union[np.ndarray, pd.Series]
    y_test: Union[np.ndarray, pd.Series]
    model_name: str = "logreg"
    tag: str = 'simple_logistic_regression'
    random_state: int = 42
    n_iter_search: int = 30
    cv_splits: int = 5
    scoring: str = "recall"
    do_random_search: bool = True
    beta_for_threshold: float = 2.0
    threshold_grid: np.ndarray = field(default_factory=lambda: np.linspace(0.05, 0.95, 50))
    preprocessor: Optional[ColumnTransformer] = None

    pipeline_: Optional[Pipeline] = field(init=False, default=None, repr=False)
    best_estimator_: Optional[BaseEstimator] = field(init=False, default=None, repr=False)
    best_threshold_: float = field(init=False, default=0.5)

    def __post_init__(self):
        self.X_train = self.X_train.copy() if isinstance(self.X_train, pd.DataFrame) else self.X_train
        self.X_test = self.X_test.copy() if isinstance(self.X_test, pd.DataFrame) else self.X_test
        self.y_train = self.y_train.copy() if isinstance(self.y_train, pd.Series) else self.y_train
        self.y_test = self.y_test.copy() if isinstance(self.y_test, pd.Series) else self.y_test
        self.is_lr = self.model_name.lower() in ("logreg", "logistic", "logisticregression")

    def _make_preprocessing(self) -> ColumnTransformer:
        numeric_features = self.X_train.select_dtypes(include=["int64", "float64", "int32"]).columns.tolist()
        categorical_features = self.X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="constant", fill_value=0)), ("scaler", StandardScaler())])
        categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="constant", fill_value="sconosciuto")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
        preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, numeric_features), ("cat", categorical_transformer, categorical_features)])
        return preprocessor

    def _make_estimator(self) -> BaseEstimator:
        if self.model_name.lower() in ("logreg", "logistic", "logisticregression"):
            return LogisticRegression(solver="liblinear", class_weight="balanced", random_state=self.random_state, max_iter=1000)
        elif self.model_name.lower() in ("rf", "randomforest", "random_forest"):
            return RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=self.random_state, n_jobs=-1)
        elif self.model_name.lower() in ("dt", "decisiontree"):
            return DecisionTreeClassifier(class_weight="balanced", random_state=self.random_state)
        raise ValueError("model_name must be one of: 'logreg', 'rf', 'dt'.")

    def _default_param_distributions(self) -> Dict[str, List[Any]]:
        if self.model_name.lower().startswith("log"):
            return {"clf__C": np.logspace(-3, 2, 20).tolist(), "clf__penalty": ["l1", "l2"], "clf__solver": ["liblinear"]}
        elif self.model_name.lower() in ("rf", "randomforest", "random_forest"):
            return {"clf__n_estimators": [200, 300, 400], "clf__max_depth": [3, 6, 9, None], "clf__min_samples_split": [2, 5]}
        elif self.model_name.lower() in ("dt", "decisiontree"):
            return {"clf__max_depth": [3, 5, 8, 10, None], "clf__min_samples_split": [2, 10, 20], "clf__criterion": ["gini", "entropy"]}
        return {}

    def fit(self, param_distributions: Optional[Dict[str, List[Any]]] = None) -> "ModelTrainer":
        preprocessor = self.preprocessor if self.preprocessor else self._make_preprocessing()
        estimator = self._make_estimator()
        base_pipe = Pipeline(steps=[("prep", preprocessor), ("clf", estimator)])
        print(f"Building estimator and pipeline for {self.model_name}...")
        if self.do_random_search is False:
            print("Fitting pipeline without RandomizedSearchCV...")
            base_pipe.fit(self.X_train, self.y_train)
            self.pipeline_ = base_pipe
            self.best_estimator_ = self.pipeline_.named_steps["clf"]
            return self
        else:
            print(f"Running RandomizedSearchCV (n_iter={self.n_iter_search}, scoring='{self.scoring}')...")
            search = RandomizedSearchCV(estimator=base_pipe, param_distributions=param_distributions or self._default_param_distributions(), n_iter=self.n_iter_search, scoring=self.scoring, cv=self.cv_splits, random_state=self.random_state, n_jobs=-1, verbose=0, refit=True)
            search.fit(self.X_train, self.y_train)
            self.pipeline_ = search.best_estimator_
            self.best_estimator_ = self.pipeline_.named_steps["clf"]
            best_params = {k.replace('clf__', ''): v for k, v in search.best_params_.items()}
            print(f"Best params found by RandomizedSearchCV: {best_params}")
        print("Fit complete.")
        return self

    def _tune_threshold(self, y_true: Union[np.ndarray, pd.Series], y_prob: np.ndarray) -> float:
        best_thr = 0.5
        best_score = -np.inf
        beta2 = self.beta_for_threshold ** 2
        for thr in self.threshold_grid:
            y_hat = (y_prob >= thr).astype(int)
            precision, recall, _, _ = precision_recall_fscore_support(y_true, y_hat, average="binary", pos_label=1, zero_division=0)
            if (beta2 * precision + recall) > 0:
                fbeta = (1 + beta2) * (precision * recall) / (beta2 * precision + recall)
            else:
                fbeta = 0.0
            if fbeta > best_score:
                best_score = fbeta
                best_thr = thr
        return best_thr

    def evaluate(self, tune_threshold: bool = True) -> Dict[str, Any]:
        if self.pipeline_ is None:
            raise RuntimeError("Call fit() before evaluate().")
        print("Computing predicted probabilities...")
        proba = self.pipeline_.predict_proba(self.X_test)[:, 1]
        if tune_threshold:
            self.best_threshold_ = self._tune_threshold(self.y_test, proba)
        else:
            self.best_threshold_ = 0.5
        y_pred = (proba >= self.best_threshold_).astype(int)
        acc = accuracy_score(self.y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(self.y_test, y_pred, average=None, labels=[0, 1], zero_division=0)
        cm = confusion_matrix(self.y_test, y_pred, labels=[0, 1])
        report = {
            "model": self.model_name,
            "tag": self.tag,
            "best_threshold": float(self.best_threshold_),
            "accuracy": float(acc),
            "metrics_per_class": {
                "non-frode": {"precision": float(precision[0]), "recall": float(recall[0]), "f1": float(f1[0]), "support": int(support[0])},
                "frode": {"precision": float(precision[1]), "recall": float(recall[1]), "f1": float(f1[1]), "support": int(support[1])},
            },
            "average_precision_class1": float(average_precision_score(self.y_test, proba)),
            "brier_score": float(brier_score_loss(self.y_test, proba)),
            "confusion_matrix": cm.tolist(),
            "y_prob": proba,
            "y_pred": y_pred,
            "params": self.best_estimator_.get_params()
        }
        print("Evaluation complete.")
        return report

    def plot_precision_recall(self, y_prob: np.ndarray, show_threshold: Optional[float] = None, save: Optional[bool] = False, filename: str = f"precision_recall_curve.png") -> str:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=110)
        PrecisionRecallDisplay.from_predictions(self.y_test, y_prob, name=self.model_name.upper(), plot_chance_level=True, ax=ax)
        ax.set_title(f"Precision–Recall Curve (class 1) • {self.model_name.upper()}")
        if show_threshold is not None:
            thr = show_threshold
            y_pred = (y_prob >= thr).astype(int)
            p1, r1, _, _ = precision_recall_fscore_support(self.y_test, y_pred, average="binary", pos_label=1, zero_division=0)
            ax.scatter([r1], [p1], s=40, marker="o", color='red', zorder=5)
            ax.annotate(f"Thr={thr:.3f}", (r1, p1), textcoords="offset points", xytext=(5, -10), color='red')
        plt.tight_layout()
        plt.show()
        if save:
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(filename))
        plt.close(fig)
        return ax

    def plot_decision_tree(self, max_depth: Optional[int] = 5, save: bool = False, filename: str = "decision_tree.png") -> None:
        if self.pipeline_ is None:
            raise RuntimeError("Call fit() before plot_decision_tree().")
        clf = self.pipeline_.named_steps["clf"]
        if not isinstance(clf, DecisionTreeClassifier):
            print(f"WARNING: The fitted model is {self.model_name}, not a Decision Tree.")
            print("This method is available only when model_name is 'dt'.")
            return
        names = self._get_feature_names()
        class_names = ["Non-Frode (0)", "Frode (1)"]
        print(f"Rendering Decision Tree (Max Depth: {max_depth})...")
        fig = plt.figure(figsize=(20, 10), dpi=100)
        plot_tree(decision_tree=clf, feature_names=names, class_names=class_names, filled=True, rounded=True, max_depth=max_depth, fontsize=8)
        plt.title(f"Decision Tree (visualized max depth: {max_depth})", fontsize=14)
        plt.tight_layout()
        plt.show()
        if save:
            try:
                Path(filename).parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(str(filename))
                print(f"Tree saved to: {filename}")
            except Exception as e:
                print(f"Error while saving file: {e}")
        plt.close(fig)

    def _get_feature_names(self) -> list[str]:
        if self.pipeline_ is None:
            raise RuntimeError("Call fit() before getting feature names.")
        if "prep" in self.pipeline_.named_steps:
            try:
                names = self.pipeline_.named_steps["prep"].get_feature_names_out()
                return list(names)
            except Exception:
                pass
        if isinstance(self.X_train, pd.DataFrame):
            return list(self.X_train.columns)
        return [f"feat_{i}" for i in range(self.pipeline_.named_steps["clf"].n_features_in_)]

    def feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        if self.pipeline_ is None:
            raise RuntimeError("Call fit() before feature_importance().")
        clf = self.pipeline_.named_steps["clf"]
        names = self._get_feature_names()
        if hasattr(clf, "coef_"):
            coefs = np.squeeze(clf.coef_)
            imp = np.abs(coefs)
            kind = "abs(coef_)"
        elif hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
            kind = "feature_importances_"
        else:
            raise ValueError("Classifier does not expose coef_ or feature_importances_.")
        df_imp = pd.DataFrame({"feature": names, "importance": imp}).sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
        print(f"Top-{top_n} feature importance by {kind}:")
        return df_imp

    def permutation_importance(self, scoring: str = "average_precision", n_repeats: int = 10, random_state: int = 42, n_jobs: int = -1, top_n: int = 20) -> pd.DataFrame:
        if self.pipeline_ is None:
            raise RuntimeError("Call fit() before permutation_importance().")
        print("Computing permutation importance on X_test...")
        result = permutation_importance(self.pipeline_, self.X_test, self.y_test, scoring=scoring, n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs)
        names = self._get_feature_names()
        df_perm = pd.DataFrame({"feature": names, "importance_mean": result.importances_mean, "importance_std": result.importances_std}).sort_values("importance_mean", ascending=False).head(top_n).reset_index(drop=True)
        print(f"Top-{top_n} permutation importance (scoring='{scoring}'):")
        return df_perm

    def _plot_waterfall(self, explanation: shap.Explanation, title: str, y_prob: float, save: bool, path: str) -> None:
        if len(explanation.shape) == 3:
            explanation_for_plot = explanation[0, :, 1]
        else:
            explanation_for_plot = explanation
        fig = plt.figure(figsize=(8, 6), dpi=100)
        shap.plots.waterfall(explanation_for_plot, show=False)
        plt.title(f"SHAP Waterfall: {title} ({self.model_name.upper()})\nPredicted Probability (Class 1): {y_prob:.4f}")
        plt.tight_layout()
        if save:
            out_dir = Path(path)
            out_dir.mkdir(parents=True, exist_ok=True)
            safe_title = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in title).replace(' ', '_')
            filename = out_dir / f"shap_waterfall_{safe_title}_{self.tag.lower()}.png"
            fig.savefig(str(filename), bbox_inches='tight')
            print(f"Waterfall plot saved to: {filename}")
        plt.show()
        plt.close(fig)

    def shap_importance(self, X_for_shap: Optional[Union[pd.DataFrame, np.ndarray]] = None, nsamples: int = 500, top_n: int = 20, save: bool = False, path: str = "shap_plots") -> Dict[str, Any]:
        if self.pipeline_ is None:
            raise RuntimeError("Call fit() before shap_importance().")
        X_test = self.X_test
        y_test = self.y_test
        y_pred = self.pipeline_.predict(X_test)
        clf = self.pipeline_.named_steps["clf"]
        names = self._get_feature_names()
        X_shap_source = X_for_shap if X_for_shap is not None else X_test
        ns = min(nsamples, len(X_shap_source))
        X_sample = X_shap_source.sample(ns, random_state=self.random_state) if isinstance(X_shap_source, pd.DataFrame) else X_shap_source[:ns]
        preprocessor = self.pipeline_.named_steps.get("prep")
        if preprocessor:
            X_trans_sample = preprocessor.transform(X_sample)
        else:
            X_trans_sample = X_sample.values if isinstance(X_sample, pd.DataFrame) else X_sample
        X_trans_df = pd.DataFrame(X_trans_sample, columns=names)
        is_tree_model = self.model_name.lower() in ("rf", "randomforest", "random_forest", "dt", "decisiontree")
        base_value_pos_class = None
        if is_tree_model:
            explainer = shap.TreeExplainer(clf, data=X_trans_df)
            shap_vals = explainer(X_trans_df)
            shap_matrix = shap_vals.values[:, :, 1]
            base_value_pos_class = explainer.expected_value[1]
        else:
            try:
                explainer = shap.LinearExplainer(clf, X_trans_df)
            except:
                explainer = shap.KernelExplainer(clf.predict_proba, X_trans_df)
            shap_matrix = explainer.shap_values(X_trans_df)[1]
            shap_vals = None
            expected_val = explainer.expected_value
            if np.ndim(expected_val) > 0:
                base_value_pos_class = expected_val[1]
            else:
                base_value_pos_class = expected_val
        print("\n--- Beeswarm Plot (Effect and Distribution) ---")
        fig_beeswarm = plt.figure(figsize=(10, 8))
        if shap_vals is not None:
            shap.plots.beeswarm(shap_vals[:, :, 1], show=False)
        else:
            shap.summary_plot(shap_matrix, X_trans_df.values, feature_names=names, plot_type="beeswarm", show=False, max_display=top_n)
        plt.title(f"SHAP Beeswarm Plot ({self.tag.upper()})")
        plt.tight_layout()
        if save:
            out_dir = Path(path)
            out_dir.mkdir(parents=True, exist_ok=True)
            filename = out_dir / f"shap_beeswarm_{self.tag.lower()}.png"
            fig_beeswarm.savefig(str(filename), bbox_inches='tight')
            print(f"Beeswarm plot saved to: {filename}")
        plt.show()
        plt.close(fig_beeswarm)
        global_importance = pd.DataFrame({'feature': names, 'mean_abs_shap': np.abs(shap_matrix).mean(axis=0)}).sort_values('mean_abs_shap', ascending=False)
        top_features = global_importance.head(top_n)
        print("\n--- Global Feature Importance (Mean Absolute SHAP) ---")
        fig_bar = plt.figure(figsize=(10, len(top_features) * 0.4))
        plt.barh(top_features['feature'], top_features['mean_abs_shap'], color='skyblue')
        plt.xlabel("Mean Absolute SHAP Value (Global Importance)")
        plt.title(f"Top {top_n} Feature Importance ({self.tag.upper()})")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        if save:
            out_dir = Path(path)
            out_dir.mkdir(parents=True, exist_ok=True)
            filename = out_dir / f"shap_global_bar_{self.tag.lower()}.png"
            fig_bar.savefig(str(filename), bbox_inches='tight')
            print(f"Global importance bar plot saved to: {filename}")
        plt.show()
        plt.close(fig_bar)
        fn_indices = y_test[(y_test == 1) & (y_pred == 0)].index.tolist()
        tp_indices = y_test[(y_test == 1) & (y_pred == 1)].index.tolist()
        fn_sample_raw = X_test.loc[[fn_indices[0]]] if fn_indices else None
        tp_sample_raw = X_test.loc[[tp_indices[0]]] if tp_indices else None
        if fn_sample_raw is not None:
            if preprocessor:
                fn_trans = preprocessor.transform(fn_sample_raw)
            else:
                fn_trans = fn_sample_raw.values
            if is_tree_model:
                fn_explanation = explainer(fn_trans, check_additivity=False)
            else:
                fn_base_value = base_value_pos_class
                fn_shap_val_sample = explainer.shap_values(fn_trans)[1][0]
                fn_explanation = shap.Explanation(values=fn_shap_val_sample, base_values=fn_base_value, data=fn_trans[0], feature_names=names)
            fn_y_prob = self.pipeline_.predict_proba(fn_sample_raw)[0, 1]
            print("\n--- Waterfall: False Negative (why it failed) ---")
            self._plot_waterfall(fn_explanation, "False Negative (y=1, p=0)", fn_y_prob, save=save, path=path)
        if tp_sample_raw is not None:
            if preprocessor:
                tp_trans = preprocessor.transform(tp_sample_raw)
            else:
                tp_trans = tp_sample_raw.values
            if is_tree_model:
                tp_explanation = explainer(tp_trans, check_additivity=False)
            else:
                tp_base_value = base_value_pos_class
                tp_shap_val_sample = explainer.shap_values(tp_trans)[1][0]
                tp_explanation = shap.Explanation(values=tp_shap_val_sample, base_values=tp_base_value, data=tp_trans[0], feature_names=names)
            tp_y_prob = self.pipeline_.predict_proba(tp_sample_raw)[0, 1]
            print("\n--- Waterfall: True Positive (why it succeeded) ---")
            self._plot_waterfall(tp_explanation, "True Positive (y=1, p=1)", tp_y_prob, save=save, path=path)
        return {"global_importance": global_importance, "shap_values": shap_matrix, "X_used": X_trans_df}


## ---------------------- ##
## FUNCTIONS
## ---------------------- ##

def auto_fill_na(df: pd.DataFrame, cols: List[str] | None = None) -> pd.DataFrame:
    
    report = {}
    
    for col in cols:
        before = df[col].isna().sum()
        fill_map = (
        df.dropna(subset=[col])
          .groupby('userid')[col]
          .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
        )
    
        df[col] = df.apply(
            lambda row: fill_map[row['userid']] if pd.isna(row[col]) and row['userid'] in fill_map else row[col],
            axis=1
        )
    
        after = df[col].isna().sum()
        fixed = before - after
    
        report[col] = {
            "nan_before": int(before),
            "nan_fixed": int(fixed),
            "nan_remaining": int(after)
        }
        
    print("\nReport subsitutions:")
    for col, stats in report.items():
        print(f"{col}: NaN before={stats['nan_before']}, filled={stats['nan_fixed']}, after={stats['nan_remaining']}")
        
    return df



def analyze_nan_fraud_rate(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Analyze how missing values relate to the fraud rate and test independence.

    For each feature column (excluding the target), the function:
      1) Creates a boolean flag ``is_nan_<feature>`` marking missing values.
      2) Computes the fraud rate among rows with NaN vs. non-NaN.
      3) Computes the difference in fraud rates (NaN minus non-NaN).
      4) Runs a Chi-square test of independence on the 2x2 table
         between ``is_nan`` and the binary target, storing the p-value.
      5) Plots a bar chart comparing ``fraud_rate_nan`` vs. ``fraud_rate_non``
         for all features that contain at least one NaN.

    The function returns a summary DataFrame with one row per feature that has NaNs:
    ``feature``, ``nan_count``, ``nan_perc``, ``fraud_rate_nan``,
    ``fraud_rate_non``, ``diff``, ``p_value``.

    Notes:
        * **Mutation:** This function adds columns named ``is_nan_<feature>`` to
          the provided DataFrame as a side effect. If you need to avoid mutation,
          pass a copy of the DataFrame.
        * **Target expectation:** ``target_col`` is expected to be binary and
          encoded as 0/1 or boolean; the function uses the mean to compute rates.
        * **Statistical caution:** The Chi-square test assumes sufficient expected
          counts (commonly ≥ 5 in each cell). Very sparse features may yield shaky
          p-values.

    Args:
        df: Input DataFrame containing the features and the target.
        target_col: Name of the binary target column indicating fraud (1) vs. non-fraud (0).

    Returns:
        A DataFrame summarizing NaN-related fraud rates and the Chi-square p-value,
        sorted by ``diff`` in descending order.

    Raises:
        KeyError: If ``target_col`` is not present in ``df``.
        ValueError: If ``target_col`` is not binary-like (contains values other than {0,1,True,False}).
        ImportError: If required dependencies (``scipy``, ``matplotlib``) are not available.

    Example:
        >>> summary = analyze_nan_fraud_rate(df, target_col="TARGET")
        >>> summary.head()
        feature  nan_count  nan_perc  fraud_rate_nan  fraud_rate_non      diff   p_value
        0     age        123      0.01            0.07            0.03  0.04000  0.002345
    """
    
    nan_summary = []

    for col in df.columns:
        if col == target_col:
            continue
        if df[col].isna().any():
            flag_col = f"is_nan_{col}"
            df[flag_col] = df[col].isna()

            fraud_rate_nan = df.loc[df[flag_col], target_col].mean()
            fraud_rate_non = df.loc[~df[flag_col], target_col].mean()

            # Tabella di contingenza per test chi-quadrato
            contingency = pd.crosstab(df[flag_col], df[target_col])
            chi2, p_value, dof, expected = chi2_contingency(contingency)

            nan_summary.append({
                "feature": col,
                "nan_count": int(df[col].isna().sum()),
                "nan_perc": df[col].isna().mean(),
                "fraud_rate_nan": fraud_rate_nan,
                "fraud_rate_non": fraud_rate_non,
                "diff": fraud_rate_nan - fraud_rate_non,
                "p_value": p_value
            })

    nan_summary_df = pd.DataFrame(nan_summary).sort_values("diff", ascending=False)

    print("NaN summary (fraud rate confronto NaN vs Non-NaN con test statistico):")


    if not nan_summary_df.empty:
        ax = nan_summary_df.set_index("feature")[["fraud_rate_nan", "fraud_rate_non"]].plot(
            kind="bar", figsize=(10, 6)
        )
        plt.title("Fraud rate per NaN vs Non-NaN")
        plt.ylabel("Fraud rate")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    return nan_summary_df



def add_anomalous_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a flag for anomalous age values in the dataset.

    The function identifies suspicious or inconsistent values of the `age` 
    column by applying three rules:
      1. Age decreases compared to the previous month for the same user.
      2. Age differs from the user's most common age (mode) by more than 2 years.
      3. Age differs from the previous month's age by more than 2 years.

    Args:
        df (pd.DataFrame): Input DataFrame containing at least the columns
            `userid`, `data_rif`, and `age`.

    Returns:
        pd.DataFrame: A copy of the input DataFrame with an additional boolean
        column `anomalous_age` set to True when any of the anomaly rules apply.
        Temporary helper columns are dropped before returning.

    Raises:
        KeyError: If `userid`, `data_rif`, or `age` are missing from the input DataFrame.

    Example:
        >>> df = pd.DataFrame({
        ...     "userid": [1, 1, 1],
        ...     "data_rif": ["2023-01-31", "2023-02-28", "2023-03-31"],
        ...     "age": [30, 29, 35]
        ... })
        >>> add_anomalous_age(df)["anomalous_age"].tolist()
        [False, True, True]
    """

    df = df.sort_values(by=["userid", "data_rif"]).copy()
    
   
    mode_age = df.groupby("userid")["age"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    df["mode_age"] = df["userid"].map(mode_age)


    df["prev_age"] = df.groupby("userid")["age"].shift(1)
    
  
    cond1 = df["age"] < df["prev_age"]
    

    cond2 = (df["age"] - df["mode_age"]).abs() > 2
    
  
    cond3 = (df["age"] - df["prev_age"]).abs() > 2
    

    df["anomalous_age"] = cond1 | cond2 | cond3
    
    return df.drop(columns=["mode_age", "prev_age"])