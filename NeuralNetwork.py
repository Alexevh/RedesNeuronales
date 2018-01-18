
# Import pandas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Import `train_test_split` from `sklearn.model_selection`
from sklearn.model_selection import train_test_split
# Import `StandardScaler` from `sklearn.preprocessing`
from sklearn.preprocessing import StandardScaler
# Import `Sequential` from `keras.models`
from keras.models import Sequential
# Import the modules from `sklearn.metrics`
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
# Import `Dense` from `keras.layers`
from keras.layers import Dense


"""
Voy a cargar en un objeto CSV de panda los valores de los csv de vinos blanco y rojos

"""
white = pd.read_csv("data/winequality-white.csv", sep=';')
red = pd.read_csv("data/winequality-red.csv", sep=';')

#Inicializo un seed random por que voy usar luego un muestreo aleatorio de datos
np.random.seed(570)

"""
Voy a transformar los dos CSV en uno solo, para eso primero les voy a agregar a los objetos pandas (csv) un
nuevo campo, se va a llamar tipo, al vino tinto le pongo 1 y al blanco 0, luego voy a unir los CSV en uno solo
con la funcion append de pandas
"""
red['type'] = 1
white['type'] = 0

#Sumo los vectores, con esto me queda un unico CSV que es la suma de los dos con un solo indice
wines = red.append(white, ignore_index=True)


"""
wines es un objeto pandas, voy a obtener una porcion de la 'lista' el metodo ix me permite hacer un slice o 
seleccion parcial de la lista por indice y por posicion, en este caso el primer slice que es el indice esta en
: por que voy a tomar todos los indices, y luego 11 columnas desde la cero.


Ejemplo, si este es el CSV y quiero traer solo hasta el indice c y la columna cuatro haria:

  x   y   z   8   9
a NaN NaN NaN NaN NaN
b NaN NaN NaN NaN NaN
c NaN NaN NaN NaN NaN
d NaN NaN NaN NaN NaN
e NaN NaN NaN NaN NaN

Solucion al corte:

 wines.ix[:'c', :4]
    x   y   z   8
a NaN NaN NaN NaN
b NaN NaN NaN NaN
c NaN NaN NaN NaN
"""
X=wines.ix[:,0:11]

"""
La funcion ravel nos aplana el array, nos devuelve un array de una dimension y seteamos el valor objetivo
valor=etiqueta en este caso el tipo que es un binario
"""
y=np.ravel(wines.type)

"""
Vamos a dividir nuestros datos en 4 conjuntos, dos de test y dos de entrenamiento, para eso usamos la funcion
split, a ella le pasamos X que es la variable que guarda nuestros datos, y que es el vector aplanado con la 
etiqueta que queremos controlar, luego el tama;o del test que va a ser del 33% y el random state
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

"""
Ahora voy a estandarizar los datos, para limpiar los outliners, primero defino el escalador, luego escalo
el set de entrenamiento y el set de pruebas

"""
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

"""
Ahora, con los datos preparados voy a generar una red de perceptrones usando Keras, el sequencial() es un 
modelo simple de stack de capas
"""
model = Sequential()

"""
Un perceptron puede tener X capas, yo voy a usar solo 3, una de entrada, una oculta y una de salida

Para la capa de entrada voy a usar una densa, que es un tipo de capa completamente conectada, la capa de entrada
tiene un input de 12 como primer parametro de Dense, esto representa la dimensionalidad del array de salida 
actualmente 12 hidden units. O sea el modelo va a tirar arrays de tama;o (*, 12).

El valor de input shape indica el tamanio del array que espera como entrada, en este caso tenemos 11 columnas asi
que el valor de entrada es 11

La capa oculta o de intermedio tambien usa la funcion relu pero su resultado son arrays de 8

La capa final va a usar una funcion de activacion sogmoid asi que la salida va a ser una probabilidad
Significa que el resultado va a ser un numero entre 0 y 1 indicando cuan alta es la probabilidad de que el
objetivo sea de valor 1 (vino tinto)
"""

#Capa de Entrada
model.add(Dense(12, activation='relu', input_shape=(11,)))
#Capa oculta
model.add(Dense(8, activation='relu'))
#Capa de salida
model.add(Dense(1, activation='sigmoid'))

"""
Voy a compilar el modelo, en loss usamos binary_crossentropy debido a que estamos evaluando valores binarios
0 o 1 y asi es mejor, si fuera un problema de regresion usaria Mean Squared Error (MSE) y si fuera un problema
de categorizacion de multiples clases usaria  categorical_crossentropy.

"""

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

"""
Entrenamos el modelo y luego de esto esta listo para ser usado
"""
model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1)

"""
Vamos a probarlo
"""
#Obtengo una prediccion y tengo como parametro una porcion de datos
y_pred = model.predict(X_test)

#Vemos el resultado de la prediccion
print("Prediccion :")
print(y_pred[:5])

"""
Recordemos que en y_test tenemos una porcion de los datos que nos reservamos en el momento del split, y que
deberia coincidir con el resultado que obtengamos de la prediccion, por eso vamos a usar esas matrices para
comparar si el modelo hace las cosas bien

"""
print("Control :")
print(y_test[:5])
type(y_test[:5])

"""
Como los valores son probabilidades estan en 0.999877 por ejemplo, recorro el resultado y lo redondeo para que
me retorne 0 o uno y lo pueda comparar correctamente con el grupo de control, ademas como algunos valores son
tan chicos me dan notaciones como 9.001-2 por ejemplo
"""
lista = y_pred[:5]
for numero in lista:
        for item in numero:
            print(round(item))

"""
Vamos a medir el score que nos da el modelo, con esto obtenemos el resultado mas o menos de loss y acuracy
"""
score = model.evaluate(X_test, y_test,verbose=1)
print("El score dio :")
print(score)

"""
Voy a usar otras herramientas de control solo por que este modelo es de autoaprendizaje

Matriz de confusion: es una descomposicion de predicciones en una tabla mostrando las predicciones correctas y el
tipo de predicciones incorrectas hechas, idealmente solo deberiamos ver numero en la diagonal que significa que
todas las predicciones son correctas

Precision: es una medida de exactitud en porcentajes 

Recall: es una medida de completitud del clasificador, cuanto mayor sea mas casos se cubren

F-score: Es una medida promedio de precision y recall

Cohen's Kappa: clasificacion de exactitud (accuracy) normalizada mediante el desbalance de las clases en los datos
"""

# Confusion matrix
#print("Matriz de confusion :")
#print(confusion_matrix(y_test, y_pred))

# Precision
#print("Precision :")
#precision_score(y_test, y_pred)

# Recall
#print("Recall :")
#recall_score(y_test, y_pred)

# F1 score
#print("F1 :")
#f1_score(y_test,y_pred)

# Cohen's kappa
#print("Cohen's Kappa :")
#cohen_kappa_score(y_test, y_pred)