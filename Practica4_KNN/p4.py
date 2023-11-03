#Gonzalez Hipolito Miguel Angel

import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from fpdf import FPDF

#lee conjunto de datos
datosIris = pd.read_csv('./iris.csv')
datosEmails = pd.read_csv('./emails.csv')

#elimina primer columna del conjunto
datosEmails = datosEmails.drop(['Email No.'], axis=1)

# Separamos los datos en x y iris
x1, y1 = datosIris.iloc[:, : 4], datosIris.iloc[:, 4] #df_iris[['species']]

# Separa los datos en x y para el conjunto de datos email
x_email, y_email  = datosEmails.iloc[:, :-1], datosEmails.iloc[:, -1] # La última columna (indicación de si el correo es spam o no)


# Conjuntos de entrenamiento (70%) y prueba (30%)
x_iris_train, x_iris_test, y_iris_train, y_iris_test = train_test_split(x1,y1, test_size=0.3,  shuffle=True,random_state=0,)
x_train_email, x_test_email, y_train_email, y_test_email = train_test_split(x_email, y_email, test_size=0.3,shuffle=True, random_state=0)


#arreglos para knn
iris = []
emails = []

knn_accuracies_email = []
knn_accuracies_iris = []

#mejor accuracy = 0
accuracyIris = 0
accuracyEmails = 0

kf = KFold(n_splits=3, shuffle=False, random_state=None)

#Busca la mejor configuracion mediante parametro
def getBestConfig(data, bestAccuracy):
    bestConfig = None
    for element in data:
        a = element.get('accuracy')
        if(a > bestAccuracy):
            bestAccuracy = a
            bestConfig = element
    return bestConfig

def createAccuracyTable(data,pdf):
    for index,item in enumerate(data):
        if(index == 0):
            pdf.cell(35, 10, str(item['dataset']), border=1)
        else:
            pdf.cell(35, 10, str(""), border=1)    
        pdf.cell(35, 10, str(item['neighbors']), border=1)
        pdf.cell(35, 10, str(item['weight']), border=1)    
        pdf.cell(35, 10, str(item['fold']), border=1)
        pdf.cell(35, 10, str(item['accuracy']), border=1)

        pdf.ln() 

def createMeanAccuracyTable(data, pdf):
    for index,item in enumerate(data):
        if(index == 0):
            pdf.cell(35, 10, str(item['dataset']), border=1)
        else:
            pdf.cell(35, 10, str(""), border=1)
        pdf.cell(35, 10, str(item['neighbors']), border=1)
        pdf.cell(35, 10, str(item['weight']), border=1)
        pdf.cell(35, 10, str(item['accuracy']), border=1)
        pdf.cell(35, 10, str(""), border=1)
        
        pdf.ln()
def createFoldAccuracy(neighbor,weight, accuracy,dataset,fold):
    data={ 
            "neighbors": neighbor,
            "weight": weight,
            "accuracy": accuracy,
            "dataset": dataset,
            "fold": fold
        }
    return data

def makeKnn(xTrain, yTrain,neighbors, weight,dataset, infoKnn, knn_mean):
    global kf
    # Funcion para realizar la validacion cruzada y calcular la precision
    knn_classifier = (KNeighborsClassifier(n_neighbors=neighbors))if neighbors == 1 else KNeighborsClassifier(n_neighbors=neighbors, weights=weight)
    accuracies = []  # Almacenar las precisiones por pliegue para obtener el promedio 
    pliegue = 0
    for train_index, val_index in kf.split(xTrain):    
        pliegue+=1
        x_train_fold, x_val_fold = xTrain.iloc[train_index], xTrain.iloc[val_index]
        y_train_fold, y_val_fold = yTrain.iloc[train_index], yTrain.iloc[val_index]

        # Entrena y evalua el modelo knn
        knn_classifier.fit(x_train_fold, y_train_fold)
        iris_knn_prediction = knn_classifier.predict(x_val_fold)
        accuracy_knn = accuracy_score(y_val_fold, iris_knn_prediction)
        accuracies.append(accuracy_knn)

        #añade los datos al arreglo
        data = createFoldAccuracy(neighbors, weight, accuracy_knn,dataset, pliegue)
        infoKnn.append(data)

    knn_mean.append({
        "dataset":dataset,
        "neighbors":neighbors,
        "weight":weight if weight is not None else "",
        "accuracy": np.mean(accuracies), 
    })

def createConfusionMatrix(name,y_test, y_prediction, labels):
    cm = confusion_matrix(y_test, y_prediction)
    # Imprimir la matriz de confusión utilizando ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues')
    disp.ax_.set_title('Matriz de Confusión')
    disp.ax_.set_xlabel('Etiquetas Predichas')
    disp.ax_.set_ylabel('Etiquetas Reales')
    plt.title("Matriz de Confusión")
    # Guardar la visualización de la matriz de confusión en una imagen
    plt.savefig(f"confusion_matrix_{name}.png", format="png")
    plt.close()

#Genera los accuracies de los pliegues para añadirlos a un arreglo y calcular el accuracy promedio.ss de cada configuracion
makeKnn(xTrain=x_iris_train, yTrain=y_iris_train,neighbors=1, dataset="iris",infoKnn=iris, knn_mean=knn_accuracies_iris,weight="----")
makeKnn(xTrain=x_iris_train, yTrain=y_iris_train, neighbors=10,dataset="iris",infoKnn=iris, knn_mean= knn_accuracies_iris, weight="uniform")
makeKnn(xTrain=x_iris_train, yTrain=y_iris_train, neighbors=10,dataset="iris",infoKnn=iris, knn_mean= knn_accuracies_iris, weight="distance")

makeKnn(xTrain=x_train_email, yTrain=y_train_email,neighbors=1, dataset="email",infoKnn=emails, knn_mean=knn_accuracies_email,weight="----")
makeKnn(xTrain=x_train_email, yTrain=y_train_email, neighbors=10,dataset="email",infoKnn=emails, knn_mean= knn_accuracies_email, weight="uniform")
makeKnn(xTrain=x_train_email, yTrain=y_train_email, neighbors=10,dataset="email",infoKnn=emails, knn_mean= knn_accuracies_email, weight="distance")


#establece la mejor configuración para el dataset de email e iris
configIris = getBestConfig(knn_accuracies_iris, accuracyIris)
configEmails = getBestConfig(knn_accuracies_email, accuracyEmails)


iris_prediction = None
emails_prediction = None


#decide cual configuracion usarpara crear el entrenamiento de iris
if(configIris.get("neighbors") == 1):
    knn = KNeighborsClassifier(n_neighbors=configIris["neighbors"])
else:
    knn = KNeighborsClassifier(n_neighbors=configIris["neighbors"], weights=configIris["weight"])

knn.fit(x_iris_train, y_iris_train)
iris_prediction = knn.predict(x_iris_test)
accuracyIrisFinal = accuracy_score(y_iris_test, iris_prediction)

#decide cual configuracion usarpara crear el entrenamiento e emails
if(configEmails.get("neighbors") == 1):
    knn = KNeighborsClassifier(n_neighbors=configEmails["neighbors"])
else:
    knn = KNeighborsClassifier(n_neighbors=configEmails["neighbors"], weights=configEmails["weight"])
knn.fit(x_train_email, y_train_email)
emails_prediction = knn.predict(x_test_email)
accuracyEmailFinal = accuracy_score(y_test_email, emails_prediction)

# Imprimir la matriz de confusión utilizando createConbfusionMatrix
labels =["Iris setosa","iris versicolor","iris-verginica"]
createConfusionMatrix("iris",y_iris_test,iris_prediction, labels)
# Generar el informe de clasificación
reporte = classification_report(y_iris_test, iris_prediction)

# Imprimir la matriz de confusión utilizando createConfusionMatrix
labels=["No es spam", "Es spam"]
createConfusionMatrix("emails",y_test_email,emails_prediction, labels)
# Generar el informe de clasificación
reporte2 = classification_report(y_test_email, emails_prediction)

pdf = FPDF()
pdf.set_font("Helvetica", size=9)
pdf.add_page()

# Encabezados de la tabla
pdf.cell(35, 10, "Dataset", border=1)
pdf.cell(35, 10, "Vecinos", border=1)
pdf.cell(35, 10, "Peso", border=1)
pdf.cell(35, 10, "Pliegue", border=1)
pdf.cell(35, 10, "Accuracy", border=1)
pdf.ln()
# Llenar la tabla con datos del arreglo de diccionarios
createAccuracyTable(iris, pdf)
createAccuracyTable(emails,pdf)

pdf.add_page()
pdf.cell(35, 10, "Dataset", border=1)
pdf.cell(35, 10, "Vecinos", border=1)
pdf.cell(35, 10, "Pesos", border=1)
pdf.cell(35, 10, "Promedio Accuracy", border=1)
pdf.cell(35, 10, str(""), border=1)
pdf.ln()
createMeanAccuracyTable(knn_accuracies_iris,pdf)
createMeanAccuracyTable(knn_accuracies_email,pdf)


pdf.add_page()
pdf.cell(30, 10, "Dataset", border=1)
pdf.cell(30, 10, "Clasificador", border=1)
pdf.cell(30, 10, "Vecinos", border=1)
pdf.cell(30, 10, "Pesos", border=1)
pdf.cell(30, 10, "Distribucion", border=1)
pdf.cell(30, 10, "Accuracy", border=1)
pdf.ln()
pdf.cell(30, 10, str("iris"), border=1)
pdf.cell(30, 10, str("Naive Bayes"), border=1)
pdf.cell(30, 10, str("------"), border=1)
pdf.cell(30, 10, str("------"), border=1)
pdf.cell(30, 10, str("Normal"), border=1)
pdf.cell(30, 10, str("1.0"), border=1)
pdf.ln()
pdf.cell(30, 10, str(""), border=1)
pdf.cell(30, 10, str("K-NN"), border=1)
pdf.cell(30, 10, str(configIris["neighbors"]), border=1)
pdf.cell(30, 10, str(configIris["weight"]), border=1)
pdf.cell(30, 10, str("------"), border=1)
pdf.cell(30, 10, str(accuracyIrisFinal), border=1)
pdf.ln()
pdf.cell(30, 10, str("emails"), border=1)
pdf.cell(30, 10, str("Naive Bayes"), border=1)
pdf.cell(30, 10, str("------"), border=1)
pdf.cell(30, 10, str("------"), border=1)
pdf.cell(30, 10, str("Normal"), border=1)
pdf.cell(30, 10, str(".9502762430939227"), border=1)
pdf.ln()
pdf.cell(30, 10, str(""), border=1)
pdf.cell(30, 10, str("K-NN"), border=1)
pdf.cell(30, 10, str(configEmails["neighbors"]), border=1)
pdf.cell(30, 10, str(configEmails["weight"]), border=1)
pdf.cell(30, 10, str("------"), border=1)
pdf.cell(30, 10, str(configEmails['accuracy']), border=1)

pdf.add_page()
pdf.multi_cell(0, 10, reporte,align="C")
pdf.image("confusion_matrix_iris.png", x=10, y=pdf.get_y(), w=190)

# Agregar una nueva página para la siguiente matriz de confusión
pdf.add_page()
pdf.multi_cell(0, 10, reporte2,align="C")
pdf.image("confusion_matrix_emails.png", x=10, y=pdf.get_y(), w=190)


# Guardar el PDF en un archivo
pdf.output("tabla.pdf")
