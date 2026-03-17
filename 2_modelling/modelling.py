import pandas as pd
import mlflow
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

DAGSHUB_USER = "fandadefchristian"
DAGSHUB_REPO = "Fraud-Detection-Experiment-2"
EXPERIMENT_NAME = "Fraud_Detection_Experiment"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../1_preprocessing/fraud_data_preprocessing", "preprocessed_fraud_dataset.csv")

def train_model():
    print("=== Memulai Proses Training Model (Advanced) ===")
    
    # 1. Setup DagsHub & MLflow
    dagshub.init(repo_owner=DAGSHUB_USER, repo_name=DAGSHUB_REPO, mlflow=True)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 2. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"Error: File tidak ditemukan di {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    TARGET_COLUMN = 'is_fraud' 
    
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Hyperparameter Tuning (Kriteria: Skilled/Advanced)
    # gunakan grid kecil agar cepat
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10]
    }
    
    with mlflow.start_run(run_name="RandomForest_Tuning_Advanced"):
        print("Training dimulai...")
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='f1_weighted')
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        # --- Log Parameters ---
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_param("model_type", "Random Forest Tuned")

        # --- Log Metrics ---
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred, average='weighted'),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted')
        }
        mlflow.log_metrics(metrics)

        # --- Log Artifacts (Min. 2 Tambahan) ---
        # 1. Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens')
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        
        # 2. Feature Importance (Artefak Tambahan ke-2)
        feat_importances = pd.Series(best_model.feature_importances_, index=X.columns)
        feat_importances.nlargest(10).plot(kind='barh')
        plt.title("Top 10 Feature Importances")
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")

        # 3. Classification Report (Artefak Tambahan ke-3)
        with open("classification_report.txt", "w") as f:
            f.write(classification_report(y_test, y_pred))
        mlflow.log_artifact("classification_report.txt")

        # --- Log Model ---
        mlflow.sklearn.log_model(best_model, "model")
        
        # Cleanup
        for f in ["confusion_matrix.png", "feature_importance.png", "classification_report.txt"]:
            if os.path.exists(f): os.remove(f)

    print("=== Selesai! Cek hasil di DagsHub. ===")

if __name__ == "__main__":
    train_model()