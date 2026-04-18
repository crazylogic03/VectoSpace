import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Configure Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(SCRIPT_DIR, "..", "..", "datasets", "train_cleaned.csv")
TEST_PATH = os.path.join(SCRIPT_DIR, "..", "..", "datasets", "test_cleaned.csv")
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def plot_feature_importance(model, feature_names, save_path):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette="viridis")
    plt.title("Random Forest - Feature Importance")
    plt.xlabel("Relative Importance")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Feature Importance Plot saved to {save_path}")

def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"Grade {i}" for i in labels])
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap="Blues", ax=ax, values_format="d")
    plt.title("Random Forest - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Confusion Matrix Plot saved to {save_path}")

def main():
    print(f"Loading data...")
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        print(f"❌ Error: Required datasets not found at {TRAIN_PATH} or {TEST_PATH}.")
        return

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    target_col = "final_grade"
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    # Model
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1)
    
    print("Performing 5-Fold Cross Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"CV Accuracy Scores: {scores}")
    print(f"Mean CV Accuracy: {scores.mean() * 100:.2f}% (+/- {scores.std() * 2 * 100:.2f}%)")
    
    print("\nTraining final model on full train set...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    labels = sorted(y_train.unique())
    target_names = [f"Grade {i}" for i in labels]
    print(f"\nFinal Test Accuracy: {acc*100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
    
    # Save visualizations
    plot_feature_importance(model, X_train.columns, os.path.join(MODEL_DIR, "feature_importance.png"))
    plot_confusion_matrix(y_test, y_pred, labels, os.path.join(MODEL_DIR, "confusion_matrix.png"))
    
    # Save model
    rf_path = os.path.join(MODEL_DIR, "random_forest.pkl")
    with open(rf_path, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Random Forest model saved to {rf_path}")
    
if __name__ == "__main__":
    main()
