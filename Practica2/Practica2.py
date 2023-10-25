#Gonzalez Hipolito Miguel Angel
#lee conjunto de datos y crea dataframe
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold     
from sklearn.utils import resample    

datos = pd.read_csv("peleteria.csv")

def crearConjunto():
    # Dividir el conjunto de datos en entrenamiento (60%) y prueba (40%)
    train_data, test_data = train_test_split(datos, test_size=0.4, shuffle=False)

    # Imprimir la cantidad de datos en cada conjunto
    print("Numero de filas en el conjunto de entrenamiento:", len(train_data))
    print("Numero de filas en el conjunto de prueba:", len(test_data))

    # Imprimir los subconjuntos de entrenamiento y prueba
    print("Subconjunto de Entrenamiento:")
    print(train_data)
    print("\nSubconjunto de Prueba:")
    print(test_data)

def validacionCruzada():
    K = 4
    cruzado = KFold(n_splits=K, shuffle=False)
    train_size = int(0.6 * len(datos))
    train_data = datos[:train_size]
    test_data = datos[train_size:]



    for fold, (train_idx, val_idx) in enumerate(cruzado.split(train_data)):
        val_data = train_data.iloc[val_idx]
        print("Pliegue {} - Tamano del conjunto de validacion: {}".format(fold + 1, len(val_data)))
        print("Conjunto de entrenamiento:")
        print(train_data.iloc[train_idx])
        print("Conjunto de validacion:")
        print(val_data)
        print("Conjunto de prueba:")
        print(test_data)
        print("\n")

def conjuntoValidacion():
    # Inicializar el objeto LeaveOneOut
    loo = LeaveOneOut()

    for train_idx, val_idx in loo.split(datos):
        train_data = datos.iloc[train_idx]
        val_data = datos.iloc[val_idx]
        print("Conjunto de entrenamiento:")
        print(train_data)
        
        print("Conjunto de validacion:")
        print(val_data)
        
        print("\n")

def conjuntosValidacionBootstrap():

    for i in range(4):
        # Genera un conjunto de entrenamiento con reemplazo
        conjunto_entrenamiento = resample(datos, n_samples=12)
        
        # El conjunto de validación es el complemento del conjunto de entrenamiento
        conjunto_validacion = datos.drop(conjunto_entrenamiento.index)
        
        print("Conjunto {}: ".format(i + 1))
        
        print("Conjunto de entrenamiento:")
        print(conjunto_entrenamiento)
        
        print("Conjunto de validacion:")
        print(conjunto_validacion)
        
        print("\n")

print("Validacion cruzada\n")
validacionCruzada()