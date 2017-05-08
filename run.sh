#DIR=src/
#descomentar si las carpetas de imágenes están donde mismo que el código
DIR=./
export PYTHONPATH=$PYTHONPATH:"${DIR}libsvm-3.22/libsvm-3.22/python/"
python "${DIR}descriptor.py"
#python "${DIR}descriptorwui.py"
