#Gonzalez Hipolito Miguel Angel

import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay 
import matplotlib.pyplot as plt
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


# Configura los clasificadores Bayesiano GaussianNB y MultinomialNB
gaussian_nb = GaussianNB()
multinomial_nb = MultinomialNB()
#aarreglos para nb y mnb
k_values = [3, 5]
datosIris = []
datosEmails = []

a_i = []
a_e = []
#mejor accuracy = 0
accuracyIris = 0
accuracyEmails = 0
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
        pdf.cell(35, 10, str(item['folds']), border=1)
        pdf.cell(35, 10, str(item['model']), border=1)    
        pdf.cell(35, 10, str(item['fold']), border=1)
        pdf.cell(35, 10, str(item['accuracy']), border=1)

        pdf.ln() 

def createMeanAccuracyTable(data, pdf):
    for index,item in enumerate(data):
        if(index == 0):
            pdf.cell(35, 10, str(item['dataset']), border=1)
        else:
            pdf.cell(35, 10, str(""), border=1)
        pdf.cell(35, 10, str(item['folds']), border=1)
        pdf.cell(35, 10, str(item['model']), border=1)
        pdf.cell(35, 10, str(item['accuracy']), border=1)
        pdf.cell(35, 10, str(""), border=1)
        
        pdf.ln()
def createFoldAccuracy(k,model, accuracy, dataset, fold, y_real, y_predict):
    data={ 
            "folds": k,
            "model": model,
            "accuracy": accuracy,
            "dataset": dataset,
            "fold": fold,
            "y_real": y_real, 
            "y_predict": y_predict
        }
    return data


# Funcion para realizar la validacion cruzada y calcular la precision
for k in k_values:
    kf = KFold(n_splits=k, shuffle=False, random_state=None)
    
    accuracies_gnb = []  # Almacenar las precisiones para Gaussian Naive Bayes
    accuracies_mnb = []  # Almacenar las precisiones para Multinomial Naive Bayes
    pliegue = 0
    for train_index, val_index in kf.split(x_iris_train):
        
        pliegue+=1
        x_train_fold, x_val_fold = x_iris_train.iloc[train_index], x_iris_train.iloc[val_index]
        y_train_fold, y_val_fold = y_iris_train.iloc[train_index], y_iris_train.iloc[val_index]

        # Entrena y evalua el modelo Gaussian Naive Bayes
        gaussian_nb.fit(x_train_fold, y_train_fold)
        iris_nb_prediction = gaussian_nb.predict(x_val_fold)
        accuracy_nb = accuracy_score(y_val_fold, iris_nb_prediction)
        accuracies_gnb.append(accuracy_nb)
        
        #añade los datos al arreglo
        data = createFoldAccuracy(k,"normal", accuracy_nb,"iris", pliegue,y_val_fold, iris_nb_prediction)
        datosIris.append(data)
        
        # Entrena y evalua el modelo Multinomial Naive Bayes
        multinomial_nb.fit(x_train_fold, y_train_fold)
        iris_mnb_prediction = multinomial_nb.predict(x_val_fold)
        accuracy_mnb = accuracy_score(y_val_fold, iris_mnb_prediction)
        accuracies_mnb.append(accuracy_mnb)

        data_mnb = createFoldAccuracy(k,"multinomial", accuracy_mnb,"iris", pliegue,y_val_fold, iris_mnb_prediction)
        datosIris.append(data_mnb)

    a_i.append({
        "dataset":"iris",
        "model":"normal",
        "folds":k,
        "accuracy": np.mean(accuracies_gnb), 
    })
    a_i.append({
        "dataset":"iris",
        "model":"multinomial",
        "folds":k,
        "accuracy": np.mean(accuracies_mnb), 
    })
configIris = getBestConfig(a_i, accuracyIris)


# Funcion para realizar la validacion cruzada y calcular la precision
for k in k_values:
    kf1 = KFold(n_splits=k, shuffle=False, random_state=None)
    
    accuraciesE_gnb = []  # Almacenar las precisiones para Gaussian Naive Bayes para email
    accuraciesE_mnb = []  # Almacenar las precisiones para Multinomial Naive Bayes para email
    pliegue=0
    for train_index, val_index in kf1.split(x_train_email):
        pliegue+=1

        x_train_fold, x_val_fold = x_train_email.iloc[train_index], x_train_email.iloc[val_index]
        y_train_fold, y_val_fold = y_train_email.iloc[train_index], y_train_email.iloc[val_index]

        # Entrena y evalua el modelo Gaussian Naive Bayes
        gaussian_nb.fit(x_train_fold, y_train_fold)
        email_nb_prediction = gaussian_nb.predict(x_val_fold)
        accuracyE_nb = accuracy_score(y_val_fold, email_nb_prediction)
        accuraciesE_gnb.append(accuracyE_nb)
        
        #añade los datos al arreglo
        data = createFoldAccuracy(k,"normal", accuracyE_nb,"email", pliegue,y_val_fold, email_nb_prediction)

        datosEmails.append(data)        
        # Entrena y evalua el modelo Multinomial Naive Bayes
        multinomial_nb.fit(x_train_fold, y_train_fold)
        email_mnb_prediction = multinomial_nb.predict(x_val_fold)
        accuracyE_mnb = accuracy_score(y_val_fold, email_mnb_prediction)
        accuraciesE_mnb.append(accuracyE_mnb)

        data_mnb = createFoldAccuracy(k,"multinomial", accuracyE_mnb,"iris", pliegue,y_val_fold, email_mnb_prediction)
        datosEmails.append(data_mnb)
    a_e.append({
        "dataset":"email",
        "model":"normal",
        "folds":k,
        "accuracy": np.mean(accuraciesE_gnb), 
    })
    a_e.append({
        "dataset":"email",
        "model":"multinomial",
        "accuracy": np.mean(accuraciesE_mnb), 
        "folds":k,
    })
    #establece la mejor configuración para el dataset de email
configEmails = getBestConfig(a_e, accuracyEmails)

iris_prediction = None
emails_prediction = None

iris_accuracy_final = None

if(configIris.get("model") == "normal"):
    gaussian_nb.fit(x_iris_train, y_iris_train)
    iris_prediction = gaussian_nb.predict(x_iris_test)
    iris_accuracy_final = accuracy_score(y_iris_test, iris_prediction)
else:
    multinomial_nb.fit(x_iris_train, y_iris_train)
    iris_prediction = multinomial_nb.predict(x_iris_test)
    iris_accuracy_final = accuracy_score(y_iris_test, iris_prediction)

if(configEmails.get("model") == "normal"):
    gaussian_nb.fit(x_train_email, y_train_email)
    emails_prediction = gaussian_nb.predict(x_test_email)
else:
    multinomial_nb.fit(x_train_email, y_train_email)
    emails_prediction = multinomial_nb.predict(x_test_email)    

print(iris_accuracy_final)
cm = confusion_matrix(y_iris_test, iris_prediction)

# Imprimir la matriz de confusión utilizando ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Iris setosa","iris versicolor","iris-verginica"])
disp.plot(cmap='Blues')
disp.ax_.set_title('Matriz de Confusión')
disp.ax_.set_xlabel('Etiquetas Predichas')
disp.ax_.set_ylabel('Etiquetas Reales')
plt.title("Matriz de Confusión")
# Guardar la visualización de la matriz de confusión en una imagen
plt.savefig("confusion_matrix_iris.png", format="png")
plt.close()

# Generar el informe de clasificación
reporte = classification_report(y_iris_test, iris_prediction)

cm2 = confusion_matrix(y_test_email, emails_prediction)

# Imprimir la matriz de confusión utilizando ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=["No es spam", "Es spam"])
disp.plot(cmap='Blues')
disp.ax_.set_title('Matriz de Confusión')
disp.ax_.set_xlabel('Etiquetas Predichas')
disp.ax_.set_ylabel('Etiquetas Reales')
plt.title("Matriz de Confusión")
# Guardar la visualización de la matriz de confusión en una imagen
plt.savefig("confusion_matrix_email.png", format="png")
plt.close()
# Generar el informe de clasificación
reporte2 = classification_report(y_test_email, emails_prediction)

pdf = FPDF()
pdf.set_font("Helvetica", size=9)
pdf.add_page()

# Encabezados de la tabla
pdf.cell(35, 10, "Dataset", border=1)
pdf.cell(35, 10, "No Pliegues", border=1)
pdf.cell(35, 10, "Distribuicion", border=1)
pdf.cell(35, 10, "Pliegue", border=1)
pdf.cell(35, 10, "Accuracy", border=1)
pdf.ln()


# Llenar la tabla con datos del arreglo de diccionarios
createAccuracyTable(datosIris, pdf)
pdf.ln()
createAccuracyTable(datosEmails, pdf)

pdf.ln()
pdf.cell(35, 10, "Dataset", border=1)
pdf.cell(35, 10, "No Pliegues", border=1)
pdf.cell(35, 10, "Distribuicion", border=1)
pdf.cell(35, 10, "Promedio Accuracy", border=1)
pdf.cell(35, 10, str(""), border=1)
pdf.ln()
createMeanAccuracyTable(a_i,pdf)
createMeanAccuracyTable(a_e,pdf)

pdf.add_page()
pdf.cell(35, 10, "Dataset", border=1)
pdf.cell(35, 10, "Distribuicion", border=1)
pdf.cell(35, 10, "Accuracy", border=1)
pdf.ln()
pdf.cell(35, 10, str(configIris['dataset']), border=1)
pdf.cell(35, 10, str(configIris['model']), border=1)
pdf.cell(35, 10, str(configIris['accuracy']), border=1)
pdf.ln()
pdf.cell(35, 10, str(configEmails['dataset']), border=1)
pdf.cell(35, 10, str(configEmails['model']), border=1)
pdf.cell(35, 10, str(configEmails['accuracy']), border=1)

pdf.add_page()
pdf.multi_cell(0, 10, reporte,align="C")
pdf.image("confusion_matrix_iris.png", x=10, y=pdf.get_y(), w=190)

# Agregar una nueva página para la siguiente matriz de confusión
pdf.add_page()
pdf.multi_cell(0, 10, reporte2,align="C")
pdf.image("confusion_matrix_email.png", x=10, y=pdf.get_y(), w=190)


# Guardar el PDF en un archivo
pdf.output("tabla.pdf")