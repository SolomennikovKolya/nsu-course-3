import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer


def get_data_from_csv() -> pd.DataFrame:
    """Получение датасета из csv файлов"""
    data_part1 = pd.read_csv("data/winequality-red.csv", sep=";",
                             encoding="utf-8", header=0, na_values=["NA", "N/A"])
    data_part2 = pd.read_csv("data/winequality-white.csv", sep=";",
                             encoding="utf-8", header=0, na_values=["NA", "N/A"])

    data_part1.insert(len(data_part1.columns) - 1, 'color', 0)
    data_part2.insert(len(data_part2.columns) - 1, 'color', 1)
    data = pd.concat([data_part1, data_part2], axis=0)
    return data


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Чистка и восстановление данных"""
    data.fillna(data.mean(), inplace=True)  # Заполнение пропусков
    data.drop_duplicates(inplace=True)      # Удаление дубликатов

    # Нормализация
    cols_to_normalize = data.columns.drop('quality')
    preprocessor = ColumnTransformer(
        transformers=[
            ('norm', MinMaxScaler(), cols_to_normalize)
        ],
        remainder='passthrough'
    )
    norm_data = preprocessor.fit_transform(data)
    data = pd.DataFrame(
        norm_data, columns=cols_to_normalize.append(pd.Index(['quality'])))
    return data


def get_data(shuffle=False, remove_unnecessary_features=False) -> pd.DataFrame:
    """Получение очищенного датасета"""
    data = get_data_from_csv()
    data = clean_data(data)

    # Удаление сильно коррелирующих признаков
    if (remove_unnecessary_features):
        data.drop('color', axis=1, inplace=True)
        data.drop('total sulfur dioxide', axis=1, inplace=True)

    # Перемешивание строк
    if (shuffle):
        data = data.sample(frac=1).reset_index(drop=True)

    return data
