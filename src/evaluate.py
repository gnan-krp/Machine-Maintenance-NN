import matplotlib.pyplot as plt
import os

def plot_loss(history, save_path='visuals/loss_curve.png'):
    os.makedirs("visuals", exist_ok=True)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.show()

def plot_predictions(model, X_test, y_test, save_path='visuals/prediction_vs_actual.png'):
    os.makedirs("visuals", exist_ok=True)
    y_pred = model.predict(X_test)
    plt.scatter(y_test, y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])
    plt.xlabel('Actual Maintenance Cost (₹)')
    plt.ylabel('Predicted Maintenance Cost (₹)')
    plt.title('Actual vs Predicted Maintenance Cost')
    plt.savefig(save_path)
    plt.show()
