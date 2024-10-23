import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))     ## Função sigmoide usada como função de ativação


def sigmoid_d(z):
    return sigmoid(z) * (1 - sigmoid(z))    ## Derivada da função sigmoide, utilizada para ajustar
                                            ## o gradiente de erro, dando maior precisão

def create_XyW():
    # Função que cria valores de Entrada (X)
    # valores de Saída (y)
    # e os pesos (W)
    
    X = np.array([[0,0,0],
                  [0,0,1],
                  [0,1,0],
                  [1,0,0]])
    
    y = np.array([[0],
                  [1],
                  [1],
                  [0]])
    
    W = 2 * np.random.random((3,1)) - 1
    # São criados os pesos para cada sample de X
    # Essa definição faz com que sejam valores aleatorios entre -1 e +1

    return X, y, W


def training(X, y, W, iterations, bias=0):
    # Função do treinamento da rede neural
    # Ela irá iterar 10.000 vezes, ajustando os pesos e bias
    
    for i in range(iterations):
        
        z = np.dot(X, W) + bias     ## multiplica a matriz com seus respectivos pesos e soma os bias
        A = sigmoid(z)              ## calcula os valores previstos para y
        
        error = y - A               ## calcula a margem de error comparando os valores reais com os previstos
        delta = error * sigmoid_d(A)   ## calcula o delta
        
        W += np.dot(X.T, delta)     ## define novos pesos levando em conta o delta
        bias += np.sum(delta)       ## define uma nova bias também levando em conta o delta
        
    z = np.dot(X, W) + bias     
    A = sigmoid(z)      
    ## após terminar o treinamento e encontrar os valores otimizados depois de 10000 iterações,
    ## calcula os valores finais e os retorna
    
    return A


if __name__ == '__main__':
    
    X, y, W = create_XyW()
    ## inicializa as variaveis principais: Entrada, Saída e Pesos
    iterations = 10000
    ## numero de iterações a ser realizada
    
    A = training(X, y, W, iterations)   ## valores previstos para y
    
    
    print("Valores esperados: \n", y)   ## imprime os valores reais
    
    np.set_printoptions(suppress=True)  ## mostra os valores por inteiro sem a notação científica
    print("\nValores previstos: \n", A) ## imprime os valores previstos


    ## Quanto mais próximo os valores previstos (A) dos valores esperados (y)
    ## melhor foi o algoritmo. Nesse caso, tendo 10.000 iterações para um problema simples
    ## como classificação binária significa que é muito provável que os valores previstos sejam
    ## muito próximos dos valores esperados.
    
