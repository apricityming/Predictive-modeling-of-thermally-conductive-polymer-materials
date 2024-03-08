import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

warnings.filterwarnings("ignore")

def load_data(file):
    Thermal_data = pd.read_csv(file)
    Thermal_data = Thermal_data.iloc[:, :]

    return Thermal_data

def preprocess_and_discretize(file, target_column):
    Thermal_data = load_data(file)
    Thermal_data_df = pd.DataFrame(Thermal_data, columns=Thermal_data.columns)
    X = Thermal_data.drop(target_column, axis=1)
    y = Thermal_data[target_column]

    scaler_minmax = MinMaxScaler()
    X_normalized = scaler_minmax.fit_transform(X)

    scaler_standard = StandardScaler()
    X_standardized = scaler_standard.fit_transform(X_normalized)

    y_discretized = y ** (2/5)

    return X_normalized, X_standardized, y_discretized, scaler_minmax, scaler_standard, Thermal_data_df

def split_train_test(X, y, test_size=0.3, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X_standardized, y_discretized, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test