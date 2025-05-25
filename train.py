from load_data import load_heart_data
from preprocess import preprocess_data
from model_neuron import create_nn_model

# Załadowanie i wstępne przetworzenie danych
data = load_heart_data("heart.csv")
X_train, X_test, y_train, y_test = preprocess_data(data)

# Stworzenie i trening modelu
model = create_nn_model(X_train.shape[1])
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Zapisanie modelu i danych
model.save("heart_failure_nn_model.h5")
import joblib
joblib.dump((X_test, y_test), "test_data.pkl")