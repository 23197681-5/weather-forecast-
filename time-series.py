# encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')#windows remove formatação utf8
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
from sklearn import metrics

plt.style.use('fivethirtyeight')

headers = ["data","hora","temperaturaInst","temperaturaMax", "temperaturaMin"]
dadosTemperatura = pd.read_csv("./TemperaturaDoisVizinhos2.csv", names=headers)

# Exclui todas as linhas com valores nulos
dadosTemperaturaFiltrados = dadosTemperatura.dropna(thresh=5)

# Remove cabecalho da lista
dadosTemperaturaSemCabecalho = dadosTemperaturaFiltrados[1:13131]
#131)

#Pega Coluna de Temperatura Maxima
dadosTemperaturaMaxima =  dadosTemperaturaSemCabecalho["temperaturaMax"]

#Pega coluna com data
dadosData = pd.to_datetime( dadosTemperaturaSemCabecalho["data"] )

dadosY = np.array(dadosTemperaturaMaxima)

dadosX = dadosData.map(dt.datetime.toordinal)
dadosX = np.array(dadosX).reshape((len(dadosX), 1))

# Listas para armazenar as previsões de cada modelo
y_pred = []
y_true = []

end = dadosY.shape[0]
window = 10
totalItems, mre, x = 0, 0, 0
for i in range(1, end-window):

    print ("Iteração = " + str(i))
    X_train = dadosX[i:i+window]
    y_train = dadosY[i:i+window]

    x_test = dadosX[i+window]
    y_test = dadosY[i+window]

    model = MLPRegressor(activation="relu", solver='lbfgs',
                         hidden_layer_sizes=(10, 35, 53, 35),
                         max_iter=100, learning_rate = 'adaptive', shuffle=True, random_state=1)

    model.fit(X_train, y_train)
    x = model.predict([x_test])
    z = float(x) - float(y_test)
    if z < 0:
        z = z * - 1
    mre += z/float(x)
    totalItems += 1
    y_pred.append(x)
    y_true.append(y_test)

# Transforma as listas em arrays numpy para facilitar os cálculos

y_pred = np.array(y_pred)
y_true = np.array(y_true)

print "MMRE(Mean Magnitude of Relative Error): " + str(mre/totalItems)


fig, ax = plt.subplots(figsize=(12,7))
fig.subplots_adjust(left=0.07, right=0.95,bottom=0.17,top=0.95)
ax.plot(y_pred, label='Prediction')
ax.plot(y_true, label='MaxTemperature')
ax.set_xlim(right=450)
ax.legend()

classes = dadosTemperaturaSemCabecalho["data"]
classes = classes[-1009:]
ax.set_xticks(np.arange(len(classes))[2::24])
_ = ax.set_xticklabels(classes[2::24], rotation=45)

plt.show()