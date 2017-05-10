#!/usr/bin/python2
# -*- coding: utf-8 -*-
from __future__ import division # División de punto flotante por defecto
from __future__ import print_function
import numpy
import random
import copy
import os
import sys
import cv2
import svmutil

def cross_validation(entradas, salidas, k):
    indices = [i for i in range(len(entradas))]
    random.shuffle(indices)
    entradas_reordenadas = [entradas[i] for i in indices]
    salidas_reordenadas  = [salidas[i]  for i in indices]
    promedios = [];
    offset = int(len(entradas) / k)
    resto = len(entradas) % k
    i = 0
    while i < len(entradas):
        fin = i + offset + (1 if resto > 0 else 0)
        resto = resto - 1 if resto > 0 else 0
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
    return media, desviacion_estandar, modelo_svm

def clasificar_imagen(modelo_svm, descriptor, salida):
    (prediccion,_,_) = svmutil.svm_predict([salida], [descriptor], modelo_svm)
    return prediccion[0]

def obtener_imagenes(directorio):
   return [cv2.imread(directorio + archivo)\
            for archivo in os.listdir(directorio)\
            if archivo.endswith(".png") or archivo.endswith('.jpg')]

def obtener_ruta_imagenes(directorio):
    return [directorio + archivo \
            for archivo in os.listdir(directorio)\
            if archivo.endswith(".png") or archivo.endswith(".jpg")]

def obtener_descriptor(imagen):
    winSize     = (64,64)
    blockSize   = (16,16)
    blockStride = (16,16) # 8,8 por defecto
    cellSize = (16,16) # 8,8 por defecto
    nbins    = 9  # por defecto
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold  = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins,
                            derivAperture, winSigma, histogramNormType,
                            L2HysThreshold, gammaCorrection, nlevels)
    imagen = cv2.resize(imagen, (64, 128)) #El tamaño depende de los parametros
    return [h[0] for h in hog.compute(imagen)]

def obtener_descriptores(imagenes):
    return [obtener_descriptor(imagen) for imagen in imagenes]

def main():
    carpeta_pedestres = '../pedestres/'
    carpeta_no_pedestres = '../no_pedestres/'
    carpeta_no_pedestres = '../floresRecortadas/'
    carpeta_salida = '../salidas/'
    archivo_modelo_svm = '../modelo_svm.model'
    num_positivas = 600
    num_negativas = 600
    k_val = 10
    print('leyendo imagenes')
    imagenes_positivas = obtener_imagenes(carpeta_pedestres)
    imagenes_negativas = obtener_imagenes(carpeta_no_pedestres)
    imagenes_positivas = imagenes_positivas[:num_positivas]
    imagenes_negativas = imagenes_negativas[:num_negativas]
    print('obteniendo_descriptores')
    descriptores_positivos = obtener_descriptores(imagenes_positivas)
    descriptores_negativos = obtener_descriptores(imagenes_negativas)
    imagenes = imagenes_positivas + imagenes_negativas
    entradas = descriptores_positivos + descriptores_negativos
    salidas  = [1]*len(descriptores_positivos) + [0]*len(descriptores_negativos)
    print('intentando cargar modelo desde archivo')
    modelo_svm = svmutil.svm_load_model(archivo_modelo_svm)
    if modelo_svm is None:
        print('ejecutando cross validation')
        (promedio, desviacion_estandar, modelo_svm) =\
                cross_validation(entradas, salidas, k_val)
        print('Promedio:', promedio)
        print('Desviación estándar:', desviacion_estandar)
        print('Guardando modelo en archivo')
        svmutil.svm_save_model(archivo_modelo_svm, modelo_svm)
    #Aquí tengo las pruebas de imagenes
    print('Clasificando imagenes')
    for i in range(len(imagenes)):
        #pos = random.randint(0, len(imagenes))
        pos = i
        imagen = imagenes[pos].copy()
        entrada = entradas[pos]
        salida_esperada = salidas[pos]
        salida = clasificar_imagen(modelo_svm, entrada, salida_esperada)
        val_pixel = [0,255,0] if salida == salida_esperada else [0,0,255]
        for j in range(len(imagen)-5, len(imagen)):
            for k in range(len(imagen[j])):
                imagen[j][k][:] = val_pixel;
        if salida != salida_esperada:
            nombre_archivo = carpeta_salida + str(i+1).zfill(3) + '.jpg'
            cv2.imwrite(nombre_archivo, imagen)
#        cv2.imshow('imagen', imagen)
#        cv2.waitKey()
    sys.exit(0)

if __name__ == '__main__':
    main()
