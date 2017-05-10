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
from descriptor import cross_validation, clasificar_imagen, obtener_descriptor,\
                       obtener_imagenes, obtener_descriptores,\
                       obtener_ruta_imagenes, obtener_imagen, obtener_descriptor
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

class MainScreen(GridLayout):


    def __init__(self, **kwargs):
        super(MainScreen, self).__init__(**kwargs)
        self.cols = 2

        # Image Widget
        self.test_image = Image()

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
        if len(self.first_folder.selection) > 0:
            if self.first_folder.selection[0].endswith('.model'):
                self.svm = svmutil.svm_load_model(self.first_folder.selection[0])
                #self.remove_widget(self.first_folder_label)
                self.remove_widget(self.first_folder)
                self.remove_widget(self.second_folder_label)
                self.remove_widget(self.second_folder)
                self.remove_widget(self.start_training)

                self.first_folder_label.text = "Carpeta de pruebas"
                #self.add_widget(self.test_folder_label)
                self.add_widget(self.test_folder)
                self.add_widget(self.read_images)
                return
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
        self.first_folder_label.text = "Carpeta de pruebas"
        # self.remove_widget(self.first_folder_label)
        self.remove_widget(self.first_folder)
        self.remove_widget(self.second_folder_label)
        self.remove_widget(self.second_folder)
        self.remove_widget(self.start_training)

        #self.add_widget(self.test_folder_label)
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
            if self.test_folder.selection[0].endswith('.png') or self.test_folder.selection[0].endswith('.jpg'):
                self.images_route = [self.test_folder.selection[0]]
                self.images_desc = [obtener_imagen(self.test_folder.selection[0])]
            else:
                self.images_route = obtener_ruta_imagenes(self.test_folder.selection[0] + '/')
                self.images_desc = obtener_imagenes(self.test_folder.selection[0] + '/')

            self.images_desc = obtener_descriptores(self.images_desc)

            self.test_image.source = self.images_route.pop()
            res = clasificar_imagen(self.svm, self.images_desc.pop(), 0)
            self.first_folder_label.text = 'Pedestre' if res == 1 else 'No pedestre'

            #self.remove_widget(self.test_folder_label)
            self.remove_widget(self.test_folder)
            self.remove_widget(self.read_images)

            #self.add_widget(self.test_image_result)
            self.add_widget(self.test_image)
            self.add_widget(self.next_image)

    def __on_click_next_image__(self, instance):
        if len(self.images_route) == 0 or len(self.images_desc) == 0:
            #self.remove_widget(self.test_image_result)
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
            self.first_folder_label.text = "Carpeta de pruebas"
            #self.add_widget(self.test_folder_label)
            self.add_widget(self.test_folder)
            self.add_widget(self.read_images)
        else:
            self.test_image.source = self.images_route.pop()
            res = clasificar_imagen(self.svm, self.images_desc.pop(), 0)
            self.first_folder_label.text = 'Pedestre' if res == 1 else 'No pedestre'


class MyApp(App):

    def build(self):
        self.title = 'Formas pedestres'
        return MainScreen()

if __name__ == '__main__':
    MyApp().run()
