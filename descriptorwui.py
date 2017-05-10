#!/usr/bin/python2
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division # División de punto flotante por defecto
from threading import Thread
from kivy.app import App
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.image import Image
import time
import numpy
import random
import copy
import os
import sys
import cv2
import svmutil


def on_thread(function):
    """
    Decorator to execute a function on a thread
    :param function:
    :return:
    """
    def decorator(*args, **kwargs):
        t = Thread(target=function, args=args, kwargs=kwargs)
        t.daemon = True
        t.start()
    return decorator


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
    return media, desviacion_estandar, modelo_svm


def clasificar_imagen(modelo_svm, descriptor, salida):
    (prediccion,_,_) = svmutil.svm_predict([salida], [descriptor], modelo_svm)
    print('Salida: ', salida)
    print('Prediccion: ', prediccion[0])
    return prediccion[0]


def obtener_ruta_imagenes(directorio):
    return [directorio + archivo \
            for archivo in os.listdir(directorio) if archivo.endswith(".png")]


def obtener_imagenes(directorio):
   return [cv2.imread(directorio + archivo)\
            for archivo in os.listdir(directorio) if archivo.endswith(".png")]       


def obtener_descriptor(imagen):
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
    imagen = cv2.resize(imagen, (64, 128)) #El tamano depende de los parametros
    return [h[0] for h in hog.compute(imagen)]


def obtener_descriptores(imagenes):
    return [obtener_descriptor(imagen) for imagen in imagenes]


def start_train(pedestrian_path, not_pedestrian_path, popup_instance):
    carpeta_pedestres = pedestrian_path + "/"
    carpeta_no_pedestres = not_pedestrian_path + "/"
    popup_instance.content.text = 'Leyendo imagenes'
    imagenes_positivas = obtener_imagenes(carpeta_pedestres)
    imagenes_negativas = obtener_imagenes(carpeta_no_pedestres)
    imagenes_positivas = imagenes_positivas[:600] # Para no tardar tanto
    imagenes_negativas = imagenes_negativas[:600]
    popup_instance.content.text = 'obteniendo_descriptores'
    descriptores_positivos = obtener_descriptores(imagenes_positivas)
    descriptores_negativos = obtener_descriptores(imagenes_negativas)
    imagenes = imagenes_negativas + imagenes_positivas
    entradas = descriptores_positivos + descriptores_negativos
    salidas  = [1]*len(descriptores_positivos) + [0]*len(descriptores_negativos)
    popup_instance.content.text = 'ejecutando cross validation'
    (promedio, desviacion_estandar, modelo_svm) =\
            cross_validation(entradas, salidas, 10)
    popup_instance.content.text = 'Promedio: {}\nDesviacion estandar: {}\nDa click fuera para continuar.'.format(promedio, desviacion_estandar)
    popup_instance.auto_dismiss = True
    return modelo_svm
    # print('Clasificando la imagen 300')
    # clasificar_imagen(modelo_svm, entradas[300], salidas[300])
    # sys.exit(0)


class MainScreen(GridLayout):


    def __init__(self, **kwargs):
        super(MainScreen, self).__init__(**kwargs)
        self.cols = 2

        # Image Widget
        self.test_image = Image()
        self.test_image_result = Label(font_size=40)

        # File and Label Widget
        self.first_folder_label = Label(text="Carpeta pedestres", font_size=40)
        self.second_folder_label = Label(text="Carpeta no pedestres", font_size=40)
        self.test_folder_label = Label(text="Carpeta de pruebas", font_size=40)
        self.first_folder = FileChooserListView(path='.', dirselect=True)
        self.second_folder = FileChooserListView(path='.', dirselect=True)
        self.test_folder = FileChooserListView(path='.', dirselect=True)

        # Button Widget
        self.start_training = Button(text='Iniciar entrenamiento', width=200)
        self.start_training.bind(on_press=self.__on_click_train__)
        self.read_images = Button(text='Iniciar pruebas', width=200)
        self.read_images.bind(on_press=self.__on_click_test__)
        self.next_image = Button(text='Siguiente imagen', width=200)
        self.next_image.bind(on_press=self.__on_click_next_image__)

        # Add Widgets
        self.add_widget(self.first_folder_label)
        self.add_widget(self.first_folder)
        self.add_widget(self.second_folder_label)
        self.add_widget(self.second_folder)
        self.add_widget(self.start_training)

    def __get_path__(self, instance, selection):
        print(selection)

    def __on_click_train__(self, instance):
        if len(self.first_folder.selection) < 1 or len(self.second_folder.selection) < 1:
            popup = Popup(
                title='Error',
                content=Label(text='Necesitas seleccionar una carpeta para iniciar.'), 
                size=(400,100), 
                size_hint=(None, None)
            )
            popup.open()
        else:
            popup = Popup(
                title='Entrenando',
                content=Label(text='Realizando entrenamiento...'), 
                size=(400,150), 
                size_hint=(None, None),
                auto_dismiss=False,
            )
            popup.open()
            self.__start_train__(popup)
            # print(self.first_folder.selection)
            # print(self.second_folder.selection)

    @on_thread
    def __start_train__(self,popup):
        """
        Starts global function start_train on a thread
        :param popup:
        :return:
        """
        self.svm = start_train(
          pedestrian_path=self.first_folder.selection[0],
          not_pedestrian_path=self.second_folder.selection[0],
          popup_instance=popup,
        )

        self.remove_widget(self.first_folder_label)
        self.remove_widget(self.first_folder)
        self.remove_widget(self.second_folder_label)
        self.remove_widget(self.second_folder)
        self.remove_widget(self.start_training)

        self.add_widget(self.test_folder_label)
        self.add_widget(self.test_folder)
        self.add_widget(self.read_images)


    def __on_click_test__(self, instance):
        if len(self.test_folder.selection) < 1:
            popup = Popup(
                title='Error',
                content=Label(text='Necesitas seleccionar una carpeta para iniciar.'), 
                size=(400,100), 
                size_hint=(None, None)
            )
            popup.open()
        else:
            self.images_route = obtener_ruta_imagenes(self.test_folder.selection[0] + '/')
            self.images_desc = obtener_imagenes(self.test_folder.selection[0] + '/')
            self.images_desc = obtener_descriptores(self.images_desc)

            self.test_image.source = self.images_route.pop()
            res = clasificar_imagen(self.svm, self.images_desc.pop(), 0)
            self.test_image_result = 'Pedestre' if res == 1 else 'No pedestre'

            self.remove_widget(self.test_folder_label)
            self.remove_widget(self.test_folder)
            self.remove_widget(self.read_images)

            self.add_widget(self.test_image_result)
            self.add_widget(self.test_image)
            self.add_widget(self.next_image)

    def __on_click_next_image__(self, instance):
        if len(self.images_route) == 0:
            self.remove_widget(self.test_image_result)
            self.remove_widget(self.test_image)
            self.remove_widget(self.next_image)

            popup = Popup(
                title='Finalizado',
                content=Label(text='Pruebas finalizadas.\nDa click fuera para hacer más pruebas'), 
                size=(400,150), 
                size_hint=(None, None),
                auto_dismiss=True,
            )

            popup.open()
            self.add_widget(self.test_folder_label)
            self.add_widget(self.test_folder)
            self.add_widget(self.read_images)
        else:
            self.test_image.source = self.images_route.pop()
            res = clasificar_imagen(self.svm, self.images_desc.pop(), 0)
            self.test_image_result = 'Pedestre' if res == 1 else 'No pedestre'


class MyApp(App):

    def build(self):
        self.title = 'Testerino'
        return MainScreen()

if __name__ == '__main__':
    MyApp().run()
