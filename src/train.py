def train_model(model, X_train, y_train, epochs=100, batch_size=8):
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )
    return history
