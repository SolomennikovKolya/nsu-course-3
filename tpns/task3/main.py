from data_preprocess import load_and_preprocess_data, create_sequences, get_dataloaders
from model_common import train_model, evaluate_model
from info import plot_predictions, print_regression_metrics_table

from lib_rnn import LibRNN
from lib_gru import LibGRU
from lib_lstm import LibLSTM
from man_rnn import ManRNN
from man_gru import ManGRU
from man_lstm import ManLSTM


SEQ_LEN: int = 24             # длина окна
BATCH_SIZE: int = 64          # размер батча
HIDDEN_SIZE: int = 64         # размер скрытых слоёв
NUM_LAYERS: int = 3           # кол-во скрытых слоёв
NUM_EPOCHS: int = 10          # кол-во эпох
LEARNING_RATE: float = 0.001  # скорость обучения


def main() -> None:
    X, y, _, scaler_y = load_and_preprocess_data()
    X_seq, y_seq = create_sequences(X, y, SEQ_LEN)
    train_dl, test_dl, y_test_scaled, y_test_orig = get_dataloaders(X_seq, y_seq, BATCH_SIZE)

    model = LibRNN(input_size=X.shape[1], hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
    # model = LibGRU(input_size=X.shape[1], hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
    # model = LibLSTM(input_size=X.shape[1], hidden_size=HIDDEN_SIZE, num_layers=NUM_LsSAYERS)

    # model = ManRNN(input_size=X.shape[1], hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
    # model = ManGRU(input_size=X.shape[1], hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
    # model = ManLSTM(input_size=X.shape[1], hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)

    train_model(model, train_dl, learning_rate=LEARNING_RATE, num_epochs=NUM_EPOCHS)

    preds_scaled = evaluate_model(model, test_dl)
    preds = scaler_y.inverse_transform(preds_scaled)
    y_true = scaler_y.inverse_transform(y_test_orig)

    plot_predictions(y_true, preds)
    print_regression_metrics_table(y_true, preds)


if __name__ == "__main__":
    main()
