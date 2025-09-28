import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

def load_data(file_path):
    """
    Загрузка данных из CSV файла с проверкой существования файла.
    :param file_path: Путь к CSV файлу.
    :return: DataFrame с загруженными данными.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден!")
    
    df = pd.read_csv(file_path)
    print(f"Данные успешно загружены. Размер: {df.shape}")
    print("Пропущенные значения по столбцам:")
    print(df.isnull().sum())
    print("Типы данных по столбцам:")
    print(df.dtypes)
    return df

def preprocess_data(df, target_column):
    """
    Предобработка данных: разделение на признаки и целевую переменную, 
    автоматическое определение типов признаков.
    :param df: DataFrame с данными.
    :param target_column: Имя столбца с целевой переменной.
    :return: Обработанные признаки, целевая переменная, препроцессор.
    """
    # Проверка наличия целевого столбца
    if target_column not in df.columns:
        raise ValueError(f"Столбец {target_column} не найден в данных. Доступные столбцы: {list(df.columns)}")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Автоматическое определение типов признаков
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Числовые признаки: {numeric_features}")
    print(f"Категориальные признаки: {categorical_features}")
    
  
    # Создание препроцессора
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Применение препроцессора к данным
    X_processed = preprocessor.fit_transform(X)
    print(f"Размер данных после обработки: {X_processed.shape}")
    
    return X_processed, y, preprocessor

def train_model(X, y):
    """
    Обучение модели линейной регрессии.
    :param X: Признаки.
    :param y: Целевая переменная.
    :return: Обученная модель.
    """
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict(model, X):
    """
    Предсказание на новых данных.
    :param model: Обученная модель.
    :param X: Признаки.
    :return: Предсказанные значения.
    """
    return model.predict(X)

def evaluate_model(y_true, y_pred):
    """
    Оценка модели с использованием метрик MSE и R^2.
    :param y_true: Истинные значения.
    :param y_pred: Предсказанные значения.
    :return: MSE, R^2.
    """
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"Среднеквадратичная ошибка: {mse:.2f}")
    print(f"Коэффициент детерминации R^2: {r2:.2f}")
    return mse, r2