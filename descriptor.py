from __future__ import print_function
import cv2

def validate(model, test_set):
   hit_count  = 0
   for instance in test_set:
      if model.classify(instance) == instance.class_label:
         hit_count += 1
   return hit_count / len(test_set)
   
def train_and_test(model, training_set, test_set):
   model.train(training_set)
   return validate(model, test_set)

def cross_validation(model, training_set, k):
   import random
   import statistics
   from copy import deepcopy
   shuffled_set = deepcopy(training_set)
   random.shuffle(shuffled_set)
   offset = int(len(training_set) / k)
   i = 0
   mean_list = []
   while i < len(training_set):
      range_end = i + offset
      test_set = shuffled_set[i:range_end]
      current_training_set = deepcopy(shuffled_set)
      del current_training_set[i:range_end]
      mean_list.append(train_and_test(model, current_training_set, test_set))
      i = range_end
   return statistics.mean(mean_list), statistics.stdev(mean_list)

def abrir_imagenes(nombre_directorio):
    pass

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

def entrenar_y_probar(xe, ye, xp, yp):
    import svmutil
    modelo_svm = svmutil.svm_train(ye, xe, '-t 0 -s 0')
    indices_vs = [i-1 for i in modelo_svm.get_sv_indices()]
#    print "Vectores soporte: "
#    for i in indices_vs:
#        print xe[i] + [ye[i]]
    svmutil.svm_predict(yp, xp, modelo_svm)
    return indices_vs

def main():
    import sys
    carpeta_pedestres = 'pedestres/'
    carpeta_no_pedestres = 'no_pedestres/'
    descriptores_positivos = obtener_descriptores(carpeta_pedestres)
    descriptores_negativos = obtener_descriptores(carpeta_no_pedestres)
    entradas = descriptores_positivos + descriptores_negativos
    salidas = [1]*len(descriptores_positivos) + [0]*len(descriptores_negativos)
    entrenar_y_probar(entradas, salidas, entradas, salidas)    

    sys.exit(0)
    h = obtener_descriptor('ejemplo.png')
    print('longitud:',len(h))
    print('caracteristicas=[',end='')
    for i in range(len(h)):
        print(h[i][0], end=',\n' if (i+1) % 8 == 0\
                                else ']\n' if i == len(h)-1\
                                else ', ')

if __name__ == '__main__':
    main()
