import pandas as pd
import logging
import os
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logging.info("Data loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def save_split_data(X_train, X_test, y_train, y_test, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    X_train.to_csv(os.path.join(save_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(save_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(save_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(save_dir, 'y_test.csv'), index=False)
    logging.info("Train-test split data saved.")

def main():
    df = load_data()
    df['BMI'] = df['BMI'].replace(0, df['BMI'].mean())
    df['BloodPressure'] = df['BloodPressure'].replace(0, df['BloodPressure'].mean())
    df['Glucose'] = df['Glucose'].replace(0, df['Glucose'].mean())
    df['Insulin'] = df['Insulin'].replace(0, df['Insulin'].mean())
    df['SkinThickness'] = df['SkinThickness'].replace(0, df['SkinThickness'].mean())

    X = df.drop('Outcome', axis=1)
    y = df[['Outcome']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    save_split_data(X_train, X_test, y_train, y_test, save_dir='./data')

if __name__ == '__main__':
    main()
