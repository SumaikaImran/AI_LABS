# import math
# import random
# # Sigmoid activation function and its derivative
# def sigmoid(x):
#     return 1 / (1 + math.exp(-x))
# def sigmoid_derivative(x):
#     return x * (1 - x)
#
# # Feedforward Neural Network with Backpropagation (using math lib)
# class NeuralNetwork:
#     def __init__(self, input_size, hidden_size, output_size, learning_rate):
#         # Initialize weights and biases
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.learning_rate = learning_rate
#
#         # Take initial weights from user
#         print("Enter initial weights for Input to Hidden Layer ({}x{} matrix):".format(input_size, hidden_size))
#         self.W_input_hidden = [[float(input(f"W[{i}][{j}]: ")) for j in range(hidden_size)] for i in range(input_size)]
#
#         print("Enter initial weights for Hidden to Output Layer ({}x{} matrix):".format(hidden_size, output_size))
#         self.W_hidden_output = [[float(input(f"W[{j}][{k}]: ")) for k in range(output_size)] for j in
#                                 range(hidden_size)]
#         self.bias_hidden = [random.uniform(-1, 1) for _ in range(hidden_size)]
#         self.bias_output = [random.uniform(-1, 1) for _ in range(output_size)]
#     def feedforward(self, inputs):
#         # Input to hidden layer
#         self.hidden_layer = []
#         for j in range(self.hidden_size):
#             hidden_sum = sum(inputs[i] * self.W_input_hidden[i][j] for i in range(self.input_size)) + self.bias_hidden[
#                 j]
#             self.hidden_layer.append(sigmoid(hidden_sum))
#         # Hidden to output layer
#         self.output_layer = []
#         for k in range(self.output_size):
#             output_sum = sum(self.hidden_layer[j] * self.W_hidden_output[j][k] for j in range(self.hidden_size)) + \
#                          self.bias_output[k]
#             self.output_layer.append(sigmoid(output_sum))
#
#         return self.output_layer
#
#     def backpropagation(self, inputs, expected_output):
#         # Compute the output layer error
#         output_error = [expected_output[k] - self.output_layer[k] for k in range(self.output_size)]
#
#         # Only proceed if error is not zero
#         if any(e != 0 for e in output_error):
#             output_delta = [output_error[k] * sigmoid_derivative(self.output_layer[k]) for k in range(self.output_size)]
#
#             # Compute the hidden layer error
#             hidden_error = [sum(output_delta[k] * self.W_hidden_output[j][k] for k in range(self.output_size)) for j in
#                             range(self.hidden_size)]
#             hidden_delta = [hidden_error[j] * sigmoid_derivative(self.hidden_layer[j]) for j in range(self.hidden_size)]
#
#             # Update weights for Hidden to Output layer
#             for j in range(self.hidden_size):
#                 for k in range(self.output_size):
#                     self.W_hidden_output[j][k] += self.learning_rate * output_delta[k] * self.hidden_layer[j]
#
#             # Update biases for output layer
#             for k in range(self.output_size):
#                 self.bias_output[k] += self.learning_rate * output_delta[k]
#
#             # Update weights for Input to Hidden layer
#             for i in range(self.input_size):
#                 for j in range(self.hidden_size):
#                     self.W_input_hidden[i][j] += self.learning_rate * hidden_delta[j] * inputs[i]
#
#             # Update biases for hidden layer
#             for j in range(self.hidden_size):
#                 self.bias_hidden[j] += self.learning_rate * hidden_delta[j]
#
#     def train(self, training_data, expected_outputs, epochs):
#         for epoch in range(epochs):
#             for i in range(len(training_data)):
#                 self.feedforward(training_data[i])
#                 self.backpropagation(training_data[i], expected_outputs[i])
#
#             # Print error every 1000 epochs for monitoring
#             if epoch % 1000 == 0:
#                 total_error = 0
#                 for i in range(len(training_data)):
#                     output = self.feedforward(training_data[i])
#                     total_error += sum((expected_outputs[i][k] - output[k]) ** 2 for k in range(self.output_size))
#                 print(f"Epoch {epoch}, Error: {total_error}")
#
#     def predict(self, inputs):
#         # Get the continuous output from the network
#         output = self.feedforward(inputs)
#
#         # Convert the continuous output to binary (0 or 1) using a threshold of 0.5
#         binary_output = [1 if o >= 0.5 else 0 for o in output]
#
#         return binary_output
# # Fixed values for input size, hidden size, output size
# input_size = 2  # Number of inputs (e.g., for XOR)
# hidden_size = 2  # Fixed number of hidden neurons
# output_size = 1  # Single output neuron for binary classification
#
# # User inputs for training data
# print(f"\nEnter training data (4 input combinations):")
# training_data = []
# for i in range(4):
#     data = list(map(int, input(f"Input {i + 1} (two values, e.g., '0 1'): ").split()))
#     training_data.append(data)
#
# # User inputs for expected outputs
# print(f"\nEnter expected outputs for the training data:")
# expected_outputs = []
# for i in range(4):
#     output = int(input(f"Expected Output for Input {training_data[i]} (0 or 1): "))
#     expected_outputs.append([output])
#
# # User inputs for learning rate and epochs
# learning_rate = float(input("\nEnter learning rate (e.g., 0.1): "))
# epochs = int(input("Enter number of epochs (e.g., 10000): "))
#
# # Initialize and train the network
# nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)
# nn.train(training_data, expected_outputs, epochs)
#
# # Testing phase loop
# while True:
#     print("\nYou can now test the network.")
#     print("Enter 'exit' to quit the testing phase.")
#
#     user_input = input("Enter test input (two values, e.g., '1 0') or 'exit' to quit: ")
#
#     if user_input.lower() == 'exit':
#         print("Exiting testing phase.")
#         break
#
#     # Convert input into integers
#     try:
#         test_input = list(map(int, user_input.split()))
#         if len(test_input) != input_size:
#             raise ValueError(f"Invalid input length. Please enter {input_size} integers.")
#     except ValueError as ve:
#         print(f"Error: {ve}")
#         continue
#
#     # Predict the output for the given input
#     output = nn.predict(test_input)
#     print(f"Predicted Output: {output}")
def dls(node, target, depth):
    if node == target:
        return True
    if depth <= 0:
        return False
    for neighbor in graph.get(node, []):
        if dls(neighbor, target, depth - 1):
            return True
    return False
def ids(start, target, max_depth):
    for depth in range(max_depth):
        print(f"Searching at depth: {depth}")
        if dls(start, target, depth):
            print(f"Target {target} found at depth {depth}!\n")
            return True
        else:
            print(f"Target {target} not found at depth {depth}.\n")
    return False
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}
# Example usage
start_node = 'A'
target_node = 'F'
max_depth = 3
if ids(start_node, target_node, max_depth):
    print(f"Target {target_node} found in the graph!")
else:
    print(f"Target {target_node} not found in the graph.")

