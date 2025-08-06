import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate(model_path, X_test_path, y_test_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    logging.info(f"Accuracy: {acc:.4f}")
    logging.info(f"Confusion Matrix:\n{cm}")

def main():
    evaluate('./models/modelForPrediction.pkl', './data/X_test_scaled.csv', './data/y_test.csv')

if __name__ == '__main__':
    main()