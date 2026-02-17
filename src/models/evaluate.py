from sklearn.metrics import classification_report,confusion_matrix
def evaluate_model(model, X_test, y_test):
    """
    Evaluates an XGBoost model on test data.
    Args:
        model: Trained model.
        X_test: Test features.
        y_test : Test Labels.

    """
    y_preds = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_preds))