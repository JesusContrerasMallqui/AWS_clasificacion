# Credits:
# https://link.springer.com/article/10.1007/s10278-018-0079-6
# https://github.com/ImagingInformatics/machine-learning
# https://github.com/paras42/Hello_World_Deep_Learning

# -----------------------------------------------------------
# Cargando modelo de disco
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam

# dimensions of our images.
img_width, img_height = 50, 50

json_file = open('./model/tarea_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("./model/tarea_model.h5")
print("Cargando modelo desde el disco ...")
loaded_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
print("Modelo cargado de disco!")

# Predict
from keras.preprocessing import image

test_image_path = './dataset/samples/3.jpg'
test_image = image.load_img(test_image_path)
plt.imshow(test_image)
plt.show()

test_image = image.load_img(test_image_path,target_size = (50, 50))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = loaded_model.predict(test_image)
if result[0][0] == 1:
    print(result[0][0], ' --> Es un perro')
else:
    print(result[0][0], ' --> Es un gato ')


