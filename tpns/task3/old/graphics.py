import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
import seaborn as sns


def show_data_distribution(X: pd.DataFrame, y: pd.Series) -> None:
    """2D проекция датасета по направлению наилучшего разделения по целевой переменной"""
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(X, y)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap="Spectral", alpha=0.7)
    plt.colorbar(label="Quality")
    plt.title("LDA-проекция")
    plt.show()


def show_graphics(y_test, y_pred) -> None:
    """Отрисовка графиков"""
    plt.figure(figsize=(12, 6))

    # График 1: предсказания vs настоящие значения
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5, color=(0, 0.33, 1, 1))
    plt.plot([1, 10], [1, 10], 'r--')
    plt.xticks(np.arange(1, 11, 1))
    plt.yticks(np.arange(1, 11, 1))
    plt.grid(True, alpha=0.3)
    plt.xlabel("Настоящее качество")
    plt.ylabel("Предсказанное качество")
    plt.title("Предсказание vs Реальность")

    # График 2: ошибка
    plt.subplot(1, 2, 2)
    errors = y_test - y_pred
    plt.hist(errors, bins=20, color='salmon', edgecolor='black')
    plt.xlabel("Ошибка")
    plt.ylabel("Частота")
    plt.title("Распределение ошибок")

    plt.tight_layout()
    plt.show()


def show_loss_curve(model):
    """График изменения ошибки"""
    plt.figure(figsize=(6, 6))
    plt.plot(model.loss_curve_)
    plt.xlabel("Эпоха")
    plt.ylabel("Loss (MSE)")
    plt.title("Изменение ошибки по эпохам")
    plt.grid(True)
    plt.show()


def show_confusion_matrix(y_test, y_pred):
    y_pred_rounded = np.clip(np.round(y_pred), 1, 10)

    cm = confusion_matrix(y_test, y_pred_rounded, labels=range(1, 11))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=range(1, 11), yticklabels=range(1, 11))
    plt.xlabel("Предугаданное качество")
    plt.ylabel("Реальное качество")
    plt.title("Матрица ошибок")
    plt.show()
