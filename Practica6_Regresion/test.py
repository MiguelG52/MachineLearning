import pandas as pd
import numpy as np

# Leer datos
casas_dataset = pd.read_csv("./casas.csv")
j_casas = pd.read_csv("./j.csv")

# Separar en x y
x_casas, y_casas = casas_dataset.iloc[:, :3], casas_dataset.iloc[:, 3]

# Añadir la columna de dummys
def create_dummy(dataframe):
    df = dataframe.copy()
    dummy_column = pd.Series([1] * len(df), name='sesgo')
    df = pd.concat([dummy_column, df], axis=1)
    return df

def update_weight(w, x, y, a):
    w_nuevo = w.copy()
    # Calcular las sumas utilizando la operación dot para cada elemento de w
    for i in range(len(w)):
        suma_i = np.dot((w[i] * x[:, i] - y), x[:, i])
        w_nuevo[i] = w[i] - 2 * a * suma_i
    return w_nuevo

def update_weight_sgd(w, x, y, a, j):
    w_nuevo = w.copy()
    for i in range(0,len(w)):
        suma = ( w[i] * x[j][i] - y[j]) * x[j][i]
        w_nuevo[i] = w[i] - 2 * a * suma 
    return w_nuevo 

def bgd(num_iterations, weight, alpha, x, y):
    for iteracion in range(num_iterations):
        # Actualizar los pesos utilizando la función
        w_nuevo = update_weight(weight, x, y, alpha)

    # Imprimir los nuevos pesos al final de todas las iteraciones
    print("Pesos finales:", w_nuevo)

def sgd(num_iterations, weights, alpha, x, y, j):
    for iteracion in range(0,num_iterations):
        index_value = j[iteracion] 
        # Actualizar los pesos utilizando la función
        weights = update_weight_sgd(weights, x, y, alpha, index_value[0])
    # Imprimir los nuevos pesos al final de todas las iteraciones
    print("Pesos finales:", weights)


# transformamos los dataframes a arreglos
x_casas_sesgo = create_dummy(x_casas)
x_casas_sesgo = np.array(x_casas_sesgo,dtype=float)
x_casas = np.array(x_casas,dtype=float)
y_casas = np.array(y_casas,dtype=float)
j_casas = np.array(j_casas)

print("BGD")
bgd(60, [0, 0,0], 0.01, x_casas, y_casas)
print("BGD sesgo")
bgd(60, [0,0,0,0], 0.01, x_casas_sesgo, y_casas )
print("SGD")
sgd(100,[0,0,0], 0.01,x_casas.copy(),y_casas,j_casas)
print("SGD sesgo")
sgd(100,[0,0,0,0], 0.01,x_casas_sesgo.copy(),y_casas,j_casas)