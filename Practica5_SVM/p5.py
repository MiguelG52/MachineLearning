import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay 
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from fpdf import FPDF

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

def calculate_total_coordinates(data, label_column, label_value):
    subset = data[data[label_column] == label_value]
    total_x = subset['sepal_length'].sum()/len(subset)
    total_y = subset['sepal_width'].sum()/len(subset)
    return total_x, total_y

def get_suport_vectors(data_x, data_y, positive_label, negative_label): 
    
    temp = pd.DataFrame(data_x, columns=["sepal_length", "sepal_width"])
    temp['species'] = data_y

    positive_vector = calculate_total_coordinates(temp, 'species', positive_label)
    negative_vector = calculate_total_coordinates(temp, 'species', negative_label) 

    total_positive_instances = len(temp[temp['species'] == positive_label])

    #invierte el vector en caso de que el el vector positivo tenga una coordenada y menor
    if negative_vector[1] > positive_vector[1]:
        positive_vector, negative_vector = negative_vector, positive_vector
        positive_label, negative_label = negative_label, positive_label

    perpendicular_vector = [(positive_vector[0] + negative_vector[0]) / 2,
                            (positive_vector[1] + negative_vector[1]) / 2]

    proba = total_positive_instances/120

    result = {
        "c-_vector": negative_vector,
        "c+_vector": positive_vector,
        "perpendicular_vector": perpendicular_vector,
        "c-_label": negative_label,
        "c+_label": positive_label,
        "proba": proba
    }
    return result

def get_projections(data, perpendicular_vector):
    projections = []
    for index, row in data.iterrows():
        vector_v = np.array([row['sepal_length'], row['sepal_width']])
        projection = (np.dot(vector_v, perpendicular_vector)) / (np.linalg.norm(perpendicular_vector))
        projections.append(projection)
    return projections

def predict_svm(data, info_vector, isInverted):
    prediction = []
    probabilities = []
    
    perpendicular_vector = info_vector["perpendicular_vector"]
    positive_c_label, negative_c_label = info_vector["c+_label"], info_vector["c-_label"]
    magnitude = np.linalg.norm(perpendicular_vector)
    probability = info_vector["proba"]

    print(magnitude)
    print(pd.DataFrame(data))

    for projection in data:
        prediction.append(positive_c_label if projection < magnitude else negative_c_label)
        probabilities.append(probability if projection < magnitude else 0) if isInverted == False else probabilities.append(0 if projection > magnitude else probability)
        
    prediction_df = pd.DataFrame({"species": prediction, "probability": probabilities})
    return prediction_df

def final_prediction(p_setosa, p_versicolor, p_virginica):
    predictions = []
    
    for index, row in p_setosa.iterrows():
        proba_setosa = row["probability"]
        proba_versicolor = p_versicolor.at[index, "probability"]
        proba_virginica = p_virginica.at[index, "probability"]
        
        # Modificación para agregar el nombre de la clase con la probabilidad más alta
        if proba_setosa == proba_versicolor == proba_virginica == 0:
            predictions.append("Iris-setosa")  # Si todas las probabilidades son 0, agregamos "Iris-setosa"
        else:
            class_probabilities = {"Iris-setosa": proba_setosa, "Iris-versicolor": proba_versicolor, "Iris-virginica": proba_virginica}
            highest_proba_class = max(class_probabilities, key=class_probabilities.get)
            predictions.append(highest_proba_class)

    p = pd.DataFrame({"species": predictions})
    return p

#lee conjunto de datos
iris_dataset = pd.read_csv('./iris.csv')
iris_dataset = iris_dataset.drop(['petal_length','petal_width'], axis=1)

#creamos conjuntos con estrategia one vs all
setosa_dataset = iris_dataset.copy()
versicolor_dataset= iris_dataset.copy()
virginica_dataset = iris_dataset.copy()

setosa_dataset['species'] = setosa_dataset['species'].apply(lambda x: "No-Iris-setosa" if x != 'Iris-setosa' else x)
versicolor_dataset['species'] = versicolor_dataset['species'].apply(lambda x: "No-Iris-versicolor" if x != 'Iris-versicolor' else x)
virginica_dataset['species'] = virginica_dataset['species'].apply(lambda x: "No-Iris-virginica" if x != 'Iris-virginica' else x)

#separamps en x 
x_setosa, y_setosa = setosa_dataset.iloc[:, :2], setosa_dataset.iloc[:, 2]

x_versicolor, y_versicolor = versicolor_dataset.iloc[:, :2], versicolor_dataset.iloc[:, 2]

x_virginica, y_virginica = virginica_dataset.iloc[:, :2], virginica_dataset.iloc[:, 2]

x_iris, y_iris = iris_dataset.iloc[:, :2], iris_dataset.iloc[:, 2]


#creamos los conjuntos de prueba y entrenamiento
x_train_setosa, x_test_setosa, y_train_setosa, y_test_setosa = train_test_split(x_setosa,y_setosa, test_size=.2, shuffle=True, random_state=0)

x_train_versicolor, x_test_versicolor, y_train_versicolor, y_test_versicolor = train_test_split(x_versicolor,y_versicolor, test_size=.2, shuffle=True, random_state=0)

x_train_virginica, x_test_virginica, y_train_virginica, y_test_virginica = train_test_split(x_virginica,y_virginica, test_size=.2, shuffle=True, random_state=0)

x_train, x_test, y_train, y_test = train_test_split(x_iris,y_iris, test_size=.2, shuffle=True, random_state=0)

#obtiene los vectores de soporte de los conjuntos de setosa, versicolor y virginica
setosa_data = get_suport_vectors(x_train_setosa, y_train_setosa,"Iris-setosa", "No-Iris-setosa")
versicolor_data = get_suport_vectors(x_train_versicolor, y_train_versicolor, "Iris-versicolor", "No-Iris-versicolor")
virginica_data = get_suport_vectors(x_train_virginica, y_train_virginica, "Iris-virginica", "No-Iris-virginica")

#obtenemos las proyeccciones
setosa_projections = get_projections(x_test_setosa, setosa_data['perpendicular_vector'])
versicolor_projections = get_projections(x_test_versicolor, versicolor_data['perpendicular_vector'])
virginica_projections = get_projections(x_test_virginica, virginica_data['perpendicular_vector'])


#calculamos svm
setosa_prediction = predict_svm(setosa_projections, setosa_data, False)
versicolor_prediction = predict_svm(versicolor_projections, versicolor_data,True)
virginica_prediction = predict_svm(virginica_projections, virginica_data, True)

prediction = final_prediction(setosa_prediction, versicolor_prediction, virginica_prediction)
labels =["Iris setosa","iris versicolor","iris-verginica"]
createConfusionMatrix("iris",y_test,prediction,labels)
