import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE


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
    # Заполнение пропусков
    data.fillna(data.mean(), inplace=True)

    # Удаление дубликатов
    data.drop_duplicates(inplace=True)

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


def balance_data(data: pd.DataFrame) -> pd.DataFrame:
    threshold = 100
    data_grouped = data.groupby('quality')
    balanced_data = data_grouped.apply(
        lambda x: x.sample(n=min(threshold, len(x))))
    return balanced_data.reset_index(drop=True)


def oversample_with_smote(data: pd.DataFrame, sampling_strategy: dict[float, int]) -> pd.DataFrame:
    """Выполняет oversampling с помощью SMOTE, игнорируя классы с недостаточным числом сэмплов"""
    sampling_strategy_cleaned = {
        i: sampling_strategy[i] for i in sampling_strategy if sampling_strategy[i] > data['quality'].value_counts()[i]}

    target_col = 'quality'
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Отбираем классы, где есть хотя бы 2 сэмпла (иначе SMOTE не сможет их использовать)
    class_counts = y.value_counts()
    valid_classes = class_counts[class_counts >= 2].index.tolist()

    # Фильтрация по допустимым классам
    mask = y.isin(valid_classes)
    X_filtered = X[mask]
    y_filtered = y[mask]

    # Подбор допустимого k_neighbors
    min_class_count = y_filtered.value_counts().min()
    k_neighbors = min(min_class_count - 1, 5)

    smote = SMOTE(sampling_strategy=sampling_strategy_cleaned,
                  random_state=54, k_neighbors=k_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X_filtered, y_filtered)

    return pd.DataFrame(X_resampled, columns=X.columns).assign(**{target_col: y_resampled})


def undersample_data(data: pd.DataFrame, sampling_strategy: dict[float, int]) -> pd.DataFrame:
    """Андерсемплинг данных по целевой переменной"""
    data_grouped = data.groupby('quality')  # Разбиваем данные по классам
    undersampled_data = []  # Список для хранения андерсемплированных данных

    for class_value, size in sampling_strategy.items():
        class_data = data_grouped.get_group(class_value)
        if len(class_data) > size:
            class_data_resampled = resample(
                class_data, n_samples=size, random_state=54)
        else:
            class_data_resampled = class_data
        undersampled_data.append(class_data_resampled)

    undersampled_data = pd.concat(undersampled_data, axis=0)
    return undersampled_data.reset_index(drop=True)


def get_data(clean_features=False, shuffle=False, oversample=False, undersample=False) -> pd.DataFrame:
    """Получение очищенного датасета"""
    data = get_data_from_csv()
    data = clean_data(data)

    # Удаление сильно кореллирующих признаков
    if clean_features:
        data.drop('color', axis=1, inplace=True)
        data.drop('total sulfur dioxide', axis=1, inplace=True)

    # Перемешивание сэмплов
    if shuffle:
        data = data.sample(frac=1).reset_index(drop=True)

    # Выравнивание датасета
    sampling_strategy = {
        3.0: 40,    # Изначально - 30
        4.0: 400,   # Изначально - 206
        5.0: 500,   # Изначально - 1752
        6.0: 500,   # Изначально - 2323
        7.0: 500,   # Изначально - 856
        8.0: 200,   # Изначально - 148
        9.0: 10,    # Изначально - 5
    }
    if undersample:
        data = undersample_data(data, sampling_strategy)
    if oversample:
        data = oversample_with_smote(data, sampling_strategy)
    return data
