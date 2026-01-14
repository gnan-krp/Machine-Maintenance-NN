from src.data_loader import load_data
from src.preprocess import preprocess
from src.model import build_model
from src.train import train_model
from src.evaluate import plot_loss, plot_predictions
from src.predict import predict_single

# 1. Load Data
X, y = load_data()

# 2. Preprocess
X_train, X_test, y_train, y_test, scaler = preprocess(X, y)

# 3. Build Model
model = build_model(X_train.shape[1])

# 4. Train Model
history = train_model(model, X_train, y_train, epochs=200, batch_size=8)

# 5. Evaluate
plot_loss(history)
plot_predictions(model, X_test, y_test)

# 6. Predict Single Sample
sample_machine = [6, 210, 4.2, 78]
predict_single(model, scaler, sample_machine)
