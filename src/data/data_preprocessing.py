import os
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def scale_and_save(X_train_path, X_test_path, save_path):
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    with open(save_path, 'wb') as f:
        pickle.dump(scaler, f)

    pd.DataFrame(X_train_scaled).to_csv('./data/interim/X_train_scaled.csv', index=False)
    pd.DataFrame(X_test_scaled).to_csv('./data/interim/X_test_scaled.csv', index=False)

    logging.info("Scaler saved and scaled data written to disk.")

def main():
    scale_and_save('./data/X_train.csv', './data/X_test.csv', './models/standardScalar.pkl')

if __name__ == '__main__':
    main()


