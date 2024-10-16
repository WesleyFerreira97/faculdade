import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    # Calcula a função sigmoid, que é uma função de ativação comumente usada em redes neurais.
    # Ela transforma a entrada em um valor entre 0 e 1, o que é útil para modelar probabilidade.

    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # Calcula a derivada da função sigmoid, que é usada durante o processo de backpropagation
    # para ajustar os pesos da rede neural.
    return x * (1 - x)

def initialize_weights(num_inputs):
    # Inicia os pesos da rede neural com valores aleatórios entre -1 e 1.
    np.random.seed(1)
    return 2 * np.random.random((num_inputs, 1)) - 1

def train_network(inputs, targets, num_iterations, learning_rate):
    # Treina a rede neural ajustando os pesos com base nos dados de entrada e nos alvos
    num_samples, num_inputs = inputs.shape
    weights = initialize_weights(num_inputs)  # Inicia os pesos
    errors_list = []  # Armazena os erros

    # Loop de treinamento
    for iteration in range(num_iterations):
        layer_output = sigmoid(np.dot(inputs, weights))  # Calcula a saída da rede
        error = targets - layer_output  # Calcula o erro
        mean_squared_error = np.mean(error ** 2)  # Calcula o erro médio ao quadrado
        errors_list.append(mean_squared_error)  # Armazena o erro

        # Exibe o erro a cada 1000 iterações
        if iteration % 1000 == 0:
            print(f"Erro médio ao quadrado na iteração {iteration}: {mean_squared_error}")

        adjustments = error * sigmoid_derivative(layer_output)  # Calcula os ajustes
        weights += learning_rate * np.dot(inputs.T, adjustments)  # Atualiza os pesos

    return weights, errors_list  # Retorna os pesos treinados e a lista de erros

def predict(inputs, weights):
    # Faz previsões com base nos dados de entrada e nos pesos treinados da rede.
    return sigmoid(np.dot(inputs, weights))

# Dados de entrada
input_data = np.array([[1, 0, 0],
                       [0, 1, 1],
                       [1, 1, 0],
                       [0, 0, 1]])

# Dados alvo
target_data = np.array([[1],
                        [0],
                        [1],
                        [0]])

# Valores de iterações e taxa de aprendizado
iterations = 10000
learning_rate = 0.05

# Treinamento
trained_weights, training_errors = train_network(input_data, target_data, iterations, learning_rate)
output = predict(input_data, trained_weights)

# Gráfico 1: Erro Médio
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(training_errors, label='Erro Médio ', color='blue')
plt.xlabel("Número de Iterações")
plt.ylabel("Erro Médio")
plt.title("Erro de Treinamento ao Longo do Tempo")
plt.legend()

# Gráfico 2: Comparação entre Saídas e Alvos
plt.subplot(1, 2, 2)
plt.plot(target_data, marker='o', label='Alvo Esperado', color='green')
plt.plot(output, marker='x', label='Saída da Rede', color='red')
plt.xlabel("Exemplos")
plt.ylabel("Valores")
plt.title("Comparação entre Saídas da Rede e Alvos")
plt.xticks(ticks=[0, 1, 2, 3], labels=['Exemplo 1', 'Exemplo 2', 'Exemplo 3', 'Exemplo 4'])
plt.legend()

plt.tight_layout()
plt.show()  # Exibe os gráficos
