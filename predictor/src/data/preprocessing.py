"""
Utilities per il preprocessing dei dati del macchinario
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle


class DataPreprocessor:
    """
    Classe per preprocessare i dati del macchinario.

    Funzionalità:
    - Normalizzazione/Standardizzazione
    - Gestione valori mancanti
    - Split train/validation/test
    - Salvataggio/Caricamento scaler per inferenza

    Args:
        scaling_method (str): 'standard' o 'minmax' (default: 'standard')
    """

    def __init__(self, scaling_method='standard'):
        self.scaling_method = scaling_method
        self.input_scaler = None
        self.output_scaler = None

        if scaling_method == 'standard':
            self.input_scaler = StandardScaler()
            self.output_scaler = StandardScaler()
        elif scaling_method == 'minmax':
            self.input_scaler = MinMaxScaler()
            self.output_scaler = MinMaxScaler()
        else:
            raise ValueError("scaling_method deve essere 'standard' o 'minmax'")

    def fit_transform(self, X, y):
        """
        Fit degli scaler sui dati e trasformazione.

        Args:
            X (np.ndarray o pd.DataFrame): Features di input
            y (np.ndarray o pd.DataFrame): Target di output

        Returns:
            tuple: (X_scaled, y_scaled)
        """
        X = self._to_numpy(X)
        y = self._to_numpy(y)

        # Gestione valori mancanti
        X = self._handle_missing_values(X)
        y = self._handle_missing_values(y)

        # Fit e trasformazione
        X_scaled = self.input_scaler.fit_transform(X)
        y_scaled = self.output_scaler.fit_transform(y)

        return X_scaled, y_scaled

    def transform(self, X, y=None):
        """
        Trasforma i dati usando scaler già fittati.

        Args:
            X (np.ndarray o pd.DataFrame): Features di input
            y (np.ndarray o pd.DataFrame, optional): Target di output

        Returns:
            np.ndarray o tuple: X_scaled o (X_scaled, y_scaled)
        """
        if self.input_scaler is None:
            raise ValueError("Scaler non ancora fittato. Usa fit_transform prima.")

        X = self._to_numpy(X)
        X = self._handle_missing_values(X)
        X_scaled = self.input_scaler.transform(X)

        if y is not None:
            y = self._to_numpy(y)
            y = self._handle_missing_values(y)
            y_scaled = self.output_scaler.transform(y)
            return X_scaled, y_scaled

        return X_scaled

    def inverse_transform_output(self, y_scaled):
        """
        Riporta gli output predetti alla scala originale.

        Args:
            y_scaled (np.ndarray): Output scalati

        Returns:
            np.ndarray: Output nella scala originale
        """
        if self.output_scaler is None:
            raise ValueError("Output scaler non fittato.")

        return self.output_scaler.inverse_transform(y_scaled)

    def split_data(self, X, y, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
        """
        Split dei dati in train, validation e test set.

        Args:
            X (np.ndarray): Features
            y (np.ndarray): Targets
            train_size (float): Proporzione training set
            val_size (float): Proporzione validation set
            test_size (float): Proporzione test set
            random_state (int): Seed per riproducibilità

        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
            "Le proporzioni devono sommare a 1.0"

        # Prima split: train vs (val+test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(1 - train_size), random_state=random_state
        )

        # Seconda split: val vs test
        val_ratio = val_size / (val_size + test_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_ratio), random_state=random_state
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def save_scalers(self, filepath):
        """Salva gli scaler per uso futuro"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'input_scaler': self.input_scaler,
                'output_scaler': self.output_scaler,
                'scaling_method': self.scaling_method
            }, f)
        print(f"Scalers salvati in: {filepath}")

    def load_scalers(self, filepath):
        """Carica scaler precedentemente salvati"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.input_scaler = data['input_scaler']
            self.output_scaler = data['output_scaler']
            self.scaling_method = data['scaling_method']
        print(f"Scalers caricati da: {filepath}")

    @staticmethod
    def _to_numpy(data):
        """Converte DataFrame o liste in numpy array"""
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            return data.values
        return np.array(data)

    @staticmethod
    def _handle_missing_values(data):
        """Gestisce valori mancanti sostituendoli con la media"""
        if np.any(np.isnan(data)):
            print("Warning: Valori mancanti trovati. Sostituiti con la media.")
            col_mean = np.nanmean(data, axis=0)
            inds = np.where(np.isnan(data))
            data[inds] = np.take(col_mean, inds[1])
        return data


def load_csv_data(filepath, input_columns, output_columns):
    """
    Carica dati da file CSV.

    Args:
        filepath (str): Path al file CSV
        input_columns (list): Nomi delle colonne di input
        output_columns (list): Nomi delle colonne di output

    Returns:
        tuple: (X, y) come numpy arrays
    """
    df = pd.read_csv(filepath)

    X = df[input_columns].values
    y = df[output_columns].values

    print(f"Dati caricati: {X.shape[0]} campioni")
    print(f"Input features: {X.shape[1]}")
    print(f"Output features: {y.shape[1]}")

    return X, y
