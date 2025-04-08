from sklearn.neural_network import MLPRegressor


def train_regressor(X_train, y_train):
    """Обучение персептрона для линейной регрессии с помощью sklearn"""
    model = MLPRegressor(
        hidden_layer_sizes=(),  # Без скрытых слоёв
        activation='identity',  # Линейная активация для регрессии
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model
