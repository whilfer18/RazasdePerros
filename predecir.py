import numpy as np
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.models import load_model

longitud, altura = 150, 150
modelo = r"C:\Users\Wilmer Fernandez\Desktop\IA\RazasdePerros\modelo\modelo.h5"
pesos_modelo = r"C:\Users\Wilmer Fernandez\Desktop\IA\RazasdePerros\modelo\pesos.h5"
cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)

def predict(file):
  x = load_img(file, target_size=(longitud, altura))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = cnn.predict(x)
  result = array[0]
  answer = np.argmax(result)
  if answer == 0:
    print("pred: Cocker")
  elif answer == 1:
    print("pred: Golden")
  elif answer == 2:
    print("pred: Pitbull")

  return answer
predict(r"C:\Users\Wilmer Fernandez\Desktop\IA\RazasdePerros\9.jpg")