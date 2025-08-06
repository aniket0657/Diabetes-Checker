import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pickle
import logging
import os

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def train_and_tune(X_train_path, y_train_path, model_save_path):
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()

    param_grid = {
        'penalty': ['l1', 'l2'],
        'C': np.logspace(-3, 3, 7),
        'solver': ['newton-cg', 'lbfgs', 'liblinear']
    }

    clf = GridSearchCV(LogisticRegression(), param_grid, scoring='accuracy', cv=10)
    clf.fit(X_train, y_train)

    best_model = clf.best_estimator_

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with open(model_save_path, 'wb') as f:
        pickle.dump(best_model, f)

    logging.info("Best model trained and saved.")

def main():
    train_and_tune('./data/X_train_scaled.csv', './data/y_train.csv', './models/modelForPrediction.pkl')

if __name__ == '__main__':
    main()