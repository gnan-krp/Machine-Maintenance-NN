import numpy as np

def predict_single(model, scaler, sample_features):
    sample_scaled = scaler.transform(np.array([sample_features]))
    pred = model.predict(sample_scaled)
    print(f"Predicted Maintenance Cost: ₹{pred[0][0]:.2f}")
    return pred
