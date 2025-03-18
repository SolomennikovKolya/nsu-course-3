import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import KBinsDiscretizer
import math
from tabulate import tabulate


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
    # df = (df - df.min()) / (df.max() - df.min())

    # Дискретизация некатегориальных данных
    chunks_num = 10
    for feature in df.columns.drop(['color', 'quality']):
        normalized_column = (df[feature] - df[feature].min()) / \
            (df[feature].max() - df[feature].min())
        df[feature] = np.floor(normalized_column * chunks_num).astype(int)
    # show_chunks_distribution(df, chunks_num)

    # Удаляем сильно коррелирующие признаки
    # df.drop('color', axis=1, inplace=True)
    # df.drop('total sulfur dioxide', axis=1, inplace=True)
    return df


def show_density(s: pd.Series, name: str = "") -> None:
    """Построение графика плотности"""
    sns.kdeplot(s, bw_method=0.1)
    plt.title(name)
    plt.show()


def show_chunks_distribution(df: pd.DataFrame, chunks_num: int) -> None:
    """Построение графика распределения данных по чанкам для каждого признака"""
    feature_bin_counts = pd.DataFrame(
        index=df.columns, columns=range(chunks_num))
    for feature in df.columns:
        for bin in range(chunks_num):
            feature_bin_counts.loc[feature, bin] = (df[feature] == bin).mean()

    plt.figure(figsize=(10, 6))
    sns.heatmap(feature_bin_counts.astype(float), annot=True,
                fmt=".2f", cmap="YlGnBu", cbar=True)
    plt.title("Доля данных в каждом чанке для каждого признака")
    plt.xlabel("Чанк")
    plt.ylabel("Признак")
    plt.show()


def standard_deviation(x: pd.Series) -> float:
    """Вычисляет стандартное отклонение величины x"""
    mean_x = np.mean(x)
    std = np.sqrt(np.sum((x - mean_x) ** 2) / (len(x) - 1))
    return std


def covariance(x: pd.Series, y: pd.Series) -> float:
    """Вычисляет ковариацию между двумя величинами x и y"""
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    cov = np.sum((x - mean_x) * (y - mean_y)) / (len(x) - 1)
    return cov


def pearson_correlation(x: pd.Series, y: pd.Series) -> float:
    """Вычисляет корреляцию Пирсона"""
    cov = covariance(x, y)
    std_x = standard_deviation(x)
    std_y = standard_deviation(y)
    return cov / (std_x * std_y)


def custom_corr(df: pd.DataFrame) -> pd.DataFrame:
    """Построение матрицы корреляции"""
    columns = df.columns
    corr_matrix = pd.DataFrame(index=columns, columns=columns)

    for col1 in columns:
        for col2 in columns:
            corr_matrix.loc[col1, col2] = pearson_correlation(
                df[col1], df[col2])
    corr_matrix = corr_matrix.astype(float)
    return corr_matrix


def show_heatmap(corr_matrix: pd.DataFrame) -> None:
    """Построение тепловой матрицы"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm',
                fmt='.2f', linewidths=0.5)
    plt.title('Тепловая матрица корреляции')
    plt.show()


def entropy(target_col: pd.Series) -> float:
    """Вычисляет энтропию для целевого столбца"""
    # Получаем уникальные значения и их частоты
    values, counts = np.unique(target_col, return_counts=True)
    # Вычисляем вероятности
    probabilities = counts / len(target_col)
    # Вычисляем энтропию
    return -np.sum(probabilities * np.log2(probabilities))


def information_gain(data: pd.DataFrame, feature: str, target: str) -> float:
    """Вычисляет информационный выигрыш для конкретного признака feature относительно целевого признака target"""
    # Энтропия исходного набора данных
    total_entropy = entropy(data[target])
    # Вычисляем средневзвешенную энтропию после разделения
    values, counts = np.unique(data[feature], return_counts=True)
    weighted_entropy = 0.0
    for v, c in zip(values, counts):
        subset = data[data[feature] == v]
        weighted_entropy += (c / len(data)) * entropy(subset[target])
    # Information Gain
    return total_entropy - weighted_entropy


def split_information(data: pd.DataFrame, feature: str) -> float:
    """Вычисляет Split Information для признака feature"""
    values, counts = np.unique(data[feature], return_counts=True)
    probabilities = counts / len(data)
    return -np.sum(probabilities * np.log2(probabilities))


def gain_ratio(data: pd.DataFrame, feature: str, target: str) -> float:
    """Вычисляет Gain Ratio для признака feature относительно целевого признака target"""
    ig = information_gain(data, feature, target)
    si = split_information(data, feature)
    return ig / si if si != 0 else 0.0


def most_important_features(data: pd.DataFrame) -> None:
    """Находит наиболее важные признаки для разделения по целевой переменной"""
    # Вычисляем Gain Ratio для каждого признака
    # Целевая переменная - качество вина
    target = 'quality'
    all_gr = []
    for feature in df.columns.drop([target]):
        gr = gain_ratio(df, feature, target)
        all_gr.append((gr, feature))
    all_gr.sort(reverse=True)

    table_data = [(feature, f"{gr:.4f}") for gr, feature in all_gr]
    print(tabulate(table_data, headers=[
          "Признак", "Gain Ratio"], tablefmt="pretty"))


if __name__ == "__main__":
    df = get_data()
    df = clean_data(df)

    # print(df)
    # show_density(df['fixed acidity'])
    # show_heatmap(df.corr())
    # show_heatmap(custom_corr(df))
    most_important_features(df)
