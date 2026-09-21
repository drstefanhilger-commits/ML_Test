from sklearn.neural_network import MLPClassifier

def build_model(learning_rate):
    return MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        learning_rate_init=learning_rate,
        max_iter=400
    )
