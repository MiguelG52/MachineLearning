#Gonzalez Hipolito Miguel Angel

import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay 
from tabulate import tabulate
import numpy as np

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
#aarreglos datosIris para nb y mnb
k_values = [3, 5]
datosIris = []
#aarreglos datosEmails para nb y mnb
datosEmails = []

#mejor accuracy = 0
accuracy = 0
#Busca la mejor configuracion mediante parametro
def getBestConfig(data):
    global accuracy
    bestConfig = next((d for d in data if d.get('accuracy') >= accuracy),None)
    return bestConfig




# Funcion para realizar la validacion cruzada y calcular la precision
print("iris")
for k in k_values:
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    
    accuracies_gnb = []  # Almacenar las precisiones para Gaussian Naive Bayes
    accuracies_mnb = []  # Almacenar las precisiones para Multinomial Naive Bayes

    for train_index, val_index in kf.split(x_iris_train):
        
        x_train_fold, x_val_fold = x_iris_train.iloc[train_index], x_iris_train.iloc[val_index]
        y_train_fold, y_val_fold = y_iris_train.iloc[train_index], y_iris_train.iloc[val_index]

        # Entrena y evalua el modelo Gaussian Naive Bayes
        gaussian_nb.fit(x_train_fold, y_train_fold)
        iris_nb_prediction = gaussian_nb.predict(x_val_fold)
        accuracy_nb = accuracy_score(y_val_fold, iris_nb_prediction)
        accuracies_gnb.append(accuracy_nb)
        
        #añade los datos al arreglo
        data={ 
            "folds": k,
            "model": "normal",
            "accuracy": accuracy_nb,
            "dataset": "iris",
            "y_real": y_val_fold,
            "y_predict":iris_nb_prediction
        }
        print("\nmodel: normal")
        print(k)
        print(accuracy_nb)
        
        datosIris.append(data)
        
        #print(data)
        #print("\n")

        # Entrena y evalua el modelo Multinomial Naive Bayes
        multinomial_nb.fit(x_train_fold, y_train_fold)
        iris_mnb_prediction = multinomial_nb.predict(x_val_fold)
        accuracy_mnb = accuracy_score(y_val_fold, iris_mnb_prediction)
        accuracies_mnb.append(accuracy_mnb)

        data_mnb = {
            "folds": k,
            "model": "multinomial",
            "accuracy": accuracy_mnb,
            "dataset": "iris",
            "y_real":y_val_fold,
            "y_predict": iris_mnb_prediction
        }
        print("\nmodel: multinominal")
        print(k)
        print(accuracy_mnb)
        datosIris.append(data_mnb)

configIris = getBestConfig(datosIris)


# Funcion para realizar la validacion cruzada y calcular la precision
print("emails")
for k in k_values:
    kf1 = KFold(n_splits=k, shuffle=True, random_state=0)
    
    accuraciesE_gnb = []  # Almacenar las precisiones para Gaussian Naive Bayes para email
    accuraciesE_mnb = []  # Almacenar las precisiones para Multinomial Naive Bayes para email

    for train_index, val_index in kf1.split(x_train_email):
        
        x_train_fold, x_val_fold = x_train_email.iloc[train_index], x_train_email.iloc[val_index]
        y_train_fold, y_val_fold = y_train_email.iloc[train_index], y_train_email.iloc[val_index]

        # Entrena y evalua el modelo Gaussian Naive Bayes
        gaussian_nb.fit(x_train_fold, y_train_fold)
        email_nb_prediction = gaussian_nb.predict(x_val_fold)
        accuracyE_nb = accuracy_score(y_val_fold, email_nb_prediction)
        accuraciesE_gnb.append(accuracyE_nb)
        
        #añade los datos al arreglo
        data={ 
            "folds": k,
            "model": "normal",
            "accuracy": accuracyE_nb,
            "dataset": "email",
            "y_real": y_val_fold, 
            "y_predict": email_nb_prediction
        }
        print("\nmodel: normal")
        print(k)
        print(accuracyE_nb)
        datosEmails.append(data)        
        # Entrena y evalua el modelo Multinomial Naive Bayes
        multinomial_nb.fit(x_train_fold, y_train_fold)
        email_mnb_prediction = multinomial_nb.predict(x_val_fold)
        accuracyE_mnb = accuracy_score(y_val_fold, email_mnb_prediction)
        accuraciesE_mnb.append(accuracyE_mnb)

        data_mnb = {
            "folds": k,
            "model": "multinomial",
            "accuracy": accuracyE_mnb,
            "dataset": "email",
            "y_real":y_val_fold,
            "y_predict":email_mnb_prediction
        }
        print("\nmodel: multinominal")
        print(k)
        print(accuracyE_mnb)
        datosEmails.append(data_mnb)

    #establece la mejor configuración para el dataset de email
configEmails = getBestConfig(datosEmails)

iris_prediction = None
emails_prediction = None


if(configIris.get("model") == "normal"):
    gaussian_nb.fit(x_iris_train, y_iris_train)
    iris_prediction = gaussian_nb.predict(x_iris_test)
else:
    multinomial_nb.fit(x_iris_train, y_iris_train)
    iris_prediction = multinomial_nb.predict(x_iris_test)

if(configEmails.get("model") == "normal"):
    gaussian_nb.fit(x_train_email, y_train_email)
    emails_prediction = gaussian_nb.predict(x_test_email)
else:
    multinomial_nb.fit(x_train_email, y_train_email)
    emails_prediction = multinomial_nb.predict(x_test_email)

    
cm = confusion_matrix(y_iris_test, iris_prediction)

# Imprimir la matriz de confusión utilizando ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Iris setosa","iris versicolor","iris-verginica"])
disp.plot(cmap='Blues')
disp.ax_.set_title('Matriz de Confusión')
disp.ax_.set_xlabel('Etiquetas Predichas')
disp.ax_.set_ylabel('Etiquetas Reales')

# Generar el informe de clasificación
reporte = classification_report(y_iris_test, iris_prediction)
print(reporte)


cm = confusion_matrix(y_test_email, emails_prediction)

# Imprimir la matriz de confusión utilizando ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No es spam", "Es spam"])
disp.plot(cmap='Blues')
disp.ax_.set_title('Matriz de Confusión')
disp.ax_.set_xlabel('Etiquetas Predichas')
disp.ax_.set_ylabel('Etiquetas Reales')

# Generar el informe de clasificación
reporte = classification_report(y_test_email, emails_prediction)
print(reporte)
