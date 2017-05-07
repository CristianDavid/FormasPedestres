#!/usr/bin/python2
# -*- coding: utf-8 -*-
from __future__ import division # División de punto flotante por defecto
from __future__ import print_function
import numpy
import random
import copy
import cv2
import svmutil

def cross_validation(entradas, salidas, k):
    indices = [i for i in range(len(entradas))]
    random.shuffle(indices)
    entradas_reordenadas = [entradas[i] for i in indices]
    salidas_reordenadas  = [salidas[i]  for i in indices]
    promedios = [];
    offset = int(len(entradas) / k);
    i = 0
    while i < len(entradas):
        fin = i + offset
        entradas_prueba = entradas_reordenadas[i:fin]
        salidas_prueba  = salidas_reordenadas[i:fin]
        entradas_entrenamiento = [entradas_reordenadas[j]\
            for j in range(len(entradas_reordenadas)) if j not in range(i,fin)]
        salidas_entrenamiento  = [salidas_reordenadas[j]\
            for j in range(len(salidas_reordenadas))  if j not in range(i,fin)]
        modelo_svm = svmutil.svm_train(
            salidas_entrenamiento, entradas_entrenamiento, '-t 0 -s 0')
        (_, precision, _) =\
            svmutil.svm_predict(salidas_prueba, entradas_prueba, modelo_svm)
        promedio = precision[0]
        promedios.append(promedio)
        i = fin
    media = numpy.mean(promedios) 
    desviacion_estandar = numpy.std(promedios)
    return media, desviacion_estandar
        

def obtener_descriptor(nombre_imagen):
    winSize     = (64,64)
    blockSize   = (16,16)
    blockStride = (16,16) # 8,8 por defecto
    cellSize = (16,16) # 8,8 por defecto
    nbins    = 3 # 9 por defecto
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold  = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins,
                            derivAperture, winSigma, histogramNormType,
                            L2HysThreshold, gammaCorrection, nlevels)
    img = cv2.imread(nombre_imagen)
    img = cv2.resize(img, (64, 128))
    return [h[0] for h in hog.compute(img)]

def obtener_descriptores(directorio):
    import os
    return [obtener_descriptor(directorio + archivo)\
            for archivo in os.listdir(directorio) if archivo.endswith(".png")]

def main():
    import sys
    carpeta_pedestres = 'pedestres/'
    carpeta_no_pedestres = 'no_pedestres/'
    descriptores_positivos = obtener_descriptores(carpeta_pedestres)
    descriptores_negativos = obtener_descriptores(carpeta_no_pedestres)
    entradas = descriptores_positivos + descriptores_negativos
    salidas = [1]*len(descriptores_positivos) + [0]*len(descriptores_negativos)
    (promedio, desviacion_estandar) = cross_validation(entradas, salidas, 10)
    print('Promedio:', promedio)
    print('Desviación estándar:', desviacion_estandar)
    sys.exit(0)

if __name__ == '__main__':
    main()
