# Classificação vetorial de Suporte Linear
# Classificação por Regressão Vetorial de Suporte


import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.svm import SVR


atletico_g = cv2.imread('atletico.jpg')
corinthians_g = cv2.imread('corinthians.jpg')
flamengo_g = cv2.imread('flamengo.jpg')
palmeiras_g = cv2.imread('palmeiras.jpg') # 200, 200, 3
teste1 = cv2.imread('teste.png')

cv2.waitKey(0)


atletico = cv2.resize(atletico_g, (10,10)) # 10,10,3
corinthians = cv2.resize(corinthians_g, (10,10))
flamengo = cv2.resize(flamengo_g, (10,10))
palmeiras = cv2.resize(palmeiras_g, (10,10))
teste = cv2.resize(teste1, (10,10))


print(atletico.shape)


X = np.concatenate((atletico, corinthians, flamengo, palmeiras), axis = 0)

 

y = [1, 2, 3, 4]

y = np.array(y)
Y = y.reshape(-1)


X = X.reshape(len(y), -1)

print(X.shape)


clf_lin = SVC(kernel='linear')
svr_lin = SVR(kernel='linear')


print('----------------------------------------------------------')
print('Inicio do treinamento do Modelo SVC')

clf_lin.fit(X,Y)

print('Término do treinamento do modelo SVC')

predicao = clf_lin.predict(flamengo.reshape(1,-1))

score = clf_lin.score(X,Y)


print(predicao)
print(score)

if predicao == 1:
   resultado = atletico_g
if predicao == 2:
   resultado = corinthians_g
if predicao == 3:
   resultado = flamengo_g
if predicao == 4:
   resultado = palmeiras_g


cv2.imshow("resultado", resultado)
cv2.imshow("teste", teste1)
cv2.waitKey(0)

print('----------------------------------------------------------')

print('Início do treinamento do modelo SVR')

svr_lin.fit(X,Y)

print('Término do treinamento do modelo SVR')

predicao = svr_lin.predict(flamengo.reshape(1,-1))

score = svr_lin.score(X,Y)


print(predicao)
print(score)

if predicao == 1:
   resultado = atletico_g
if predicao == 2:
   resultado = corinthians_g
if predicao == 3:
   resultado = flamengo_g
if predicao == 4:
   resultado = palmeiras_g


cv2.imshow("resultado", resultado)
cv2.imshow("teste", teste1)
cv2.waitKey(0)
