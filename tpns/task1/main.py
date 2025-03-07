import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_data() -> pd.DataFrame:
    """Получение датасета из csv файлов"""
    df1 = pd.read_csv("data/winequality-red.csv", sep=";",
                      encoding="utf-8", header=0, na_values=["NA", "N/A"])
    df2 = pd.read_csv("data/winequality-white.csv", sep=";",
                      encoding="utf-8", header=0, na_values=["NA", "N/A"])

    df1.insert(len(df1.columns) - 1, 'color', 0)
    df2.insert(len(df2.columns) - 1, 'color', 1)

    combined_df = pd.concat([df1, df2], axis=0)
    shuffled_df = combined_df.sample(frac=1).reset_index(drop=True)
    return shuffled_df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Чистка и восстановление данных"""
    # Заполняем пропущенные данные (mean - мат ожидание)
    df.fillna(df.mean(), inplace=True)

    # Удаляем дубликаты
    df.drop_duplicates(inplace=True)

    # Нормализация всех числовых столбцов
    df = (df - df.min()) / (df.max() - df.min())
    return df


def show_density(s: pd.Series, name: str = "") -> None:
    """Построение графика плотности"""
    sns.kdeplot(s, bw_method=0.1)
    plt.title(name)
    plt.show()


def show_heatmap(df: pd.DataFrame) -> None:
    """Построение тепловой матрицы"""
    # Вычисляем матрицу корреляции
    corr_matrix = df.corr()

    # Строим тепловую карту
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm',
                fmt='.2f', linewidths=0.5)
    plt.title('Тепловая матрица корреляции')
    plt.show()


if __name__ == "__main__":
    df = get_data()
    df = clean_data(df)
    df = df.drop('color', axis=1)
    df = df.drop('total sulfur dioxide', axis=1)
    show_heatmap(df)
    # print(df)
