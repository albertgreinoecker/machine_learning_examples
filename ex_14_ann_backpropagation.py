import numpy as np


# Aktivierungsfunktionen und ihre Ableitungen
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Neuronales Netzwerk
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialisierung der Gewichte
        self.weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))

        self.bias_hidden = np.random.uniform(-1, 1, (1, hidden_size))
        self.bias_output = np.random.uniform(-1, 1, (1, output_size))

    def forward(self, inputs):
        # Vorwärtsdurchlauf
        self.input_layer = inputs
        self.hidden_layer = sigmoid(
            np.dot(self.input_layer, self.weights_input_hidden) + self.bias_hidden
        )
        self.output_layer = sigmoid(
            np.dot(self.hidden_layer, self.weights_hidden_output) + self.bias_output
        )
        return self.output_layer

    def backward(self, expected_output, learning_rate):
        # Fehlerberechnung
        output_error = expected_output - self.output_layer
        output_delta = output_error * sigmoid_derivative(self.output_layer)

        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_layer)

        # Gewichts- und Bias-Updates
        self.weights_hidden_output += np.dot(self.hidden_layer.T, output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

        self.weights_input_hidden += np.dot(self.input_layer.T, hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(self, inputs, targets, epochs, learning_rate):
        for epoch in range(epochs):
            outputs = self.forward(inputs)
            self.backward(targets, learning_rate)

            # Fehleranzeige alle 100 Epochen
            if epoch % 100 == 0:
                loss = np.mean(np.square(targets - outputs))
                print(f"Epoch {epoch}, Loss: {loss}")


# Beispiel: XOR-Problem
if __name__ == "__main__":
    # Eingaben und erwartete Ausgaben
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([[0], [1], [1], [0]])

    # Initialisiere das Netzwerk
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)

    # Trainiere das Netzwerk
    nn.train(inputs, targets, epochs=10000, learning_rate=0.1)

    # Teste das Netzwerk
    print("\nErgebnisse nach dem Training:")
    for input_data in inputs:
        output = nn.forward(input_data)
        print(f"Eingabe: {input_data}, Ausgabe: {output}")
