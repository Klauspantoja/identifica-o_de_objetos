O projeto é uma ferramenta avançada de reconhecimento de imagem baseada em inteligência artificial. 
Ele é capaz de analisar e identificar objetos, textos, padrões e características em imagens fornecidas, transformando dados visuais em informações categorizadas
#######################################################################

Funcionalidades Principais:

Reconhecimento de Objetos: Identifica e classifica objetos em uma imagem.
Reconhecimento Facial: Identifica faces e, em cenários avançados, pode categorizar expressões emocionais.

#######################################################################

Instruções de Instalação: para sua execução vamos precisar de algumas biblioticas 
Importação das Bibliotecas:  

from keras.models import load_model  
import cv2  
import numpy as np


'load_model' da Keras é usado para carregar um modelo pré-treinado.
'cv2' é a biblioteca OpenCV, uma ferramenta amplamente utilizada para processamento de imagens e visão computacional.
'numpy' é uma biblioteca para manipulação numérica de dados.

Em resumo, o código captura imagens da webcam, processa e classifica-as usando um modelo Keras pré-treinado e, em seguida, exibe a imagem e o resultado da classificação. 
A aplicação pode ser interrompida pressionando a tecla 'esc'.

######################################################################


from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()




Créditos: DIEGO SOUZA COSTA  / 
          CLAUDENOR PANTOJA 
