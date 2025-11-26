#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
import os
import gzip
import json
import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

CONFIG = {
    "data": {
        "train": "files/input/train_data.csv.zip",
        "test": "files/input/test_data.csv.zip",
    },
    "artifacts": {
        "model": "files/models/model.pkl.gz",
        "metrics": "files/output/metrics.json",
    },
    "features": {
        "drop": ["Year", "Car_Name"],
        "target": "Present_Price",
        "categorical": ["Fuel_Type", "Selling_type", "Transmission"],
    },
}

class VehiclePricePredictor:

    def __init__(self):
        self.grid_search = None
    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lógica de transformación compacta."""
        return (
            df.copy()
            .assign(Age=lambda x: 2021 - x["Year"])
            .drop(columns=CONFIG["features"]["drop"])
        )
    def load_datasets(self):
        """Carga y preprocesa train y test en un solo paso."""
        paths = CONFIG["data"]
        dfs = {
            k: pd.read_csv(v, compression="zip") 
            for k, v in paths.items()
        }
        
        self.train_df = self._process_data(dfs["train"])
        self.test_df = self._process_data(dfs["test"])
    def build_pipeline(self, feature_columns):
        """Construye el pipeline scikit-learn."""
        categorical_cols = CONFIG["features"]["categorical"]
        numerical_cols = [c for c in feature_columns if c not in categorical_cols]
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(), categorical_cols),
                ("num", MinMaxScaler(), numerical_cols),
            ]
        )

        return Pipeline([
            ("preprocesador", preprocessor),
            ("selector", SelectKBest(score_func=f_regression)),
            ("regresor", LinearRegression()),
        ])
    def train(self):
        target = CONFIG["features"]["target"]
        X_train = self.train_df.drop(columns=target)
        y_train = self.train_df[target]
        pipeline = self.build_pipeline(X_train.columns)
        param_grid = {
            "selector__k": range(1, 15),
            "regresor__fit_intercept": [True, False],
            "regresor__positive": [True, False],
        }

        self.grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=10,
            scoring="neg_mean_absolute_error",
            n_jobs=-1
        )
        
        self.grid_search.fit(X_train, y_train)
    def export_model(self):
        path = CONFIG["artifacts"]["model"]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with gzip.open(path, "wb") as f:
            pickle.dump(self.grid_search, f)
    def compute_metrics(self):
        target = CONFIG["features"]["target"]
        metrics_path = CONFIG["artifacts"]["metrics"]
        datasets = {
            "train": (self.train_df.drop(columns=target), self.train_df[target]),
            "test": (self.test_df.drop(columns=target), self.test_df[target]),
        }

        results = []
        for name, (X, y) in datasets.items():
            y_pred = self.grid_search.predict(X)
            results.append({
                "type": "metrics",
                "dataset": name,
                "r2": float(r2_score(y, y_pred)),
                "mse": float(mean_squared_error(y, y_pred)),
                "mad": float(median_absolute_error(y, y_pred)),
            })
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            for res in results:
                f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    predictor = VehiclePricePredictor()
    predictor.load_datasets()
    predictor.train()
    predictor.export_model()
    predictor.compute_metrics()

