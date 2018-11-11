#Import Flask
from flask import Flask, request
from keras.preprocessing import image
from cnn_executor import cargarModelo
import numpy as np

#Initialize the application service
app = Flask(__name__)
global loaded_model, graph
loaded_model, graph = cargarModelo()

#Define a route
@app.route('/')
def main_page():
	return 'Bienvenido a la URP - RNA!'

@app.route('/Clasificacion/', methods=['GET','POST'])
def rayosx():
	return 'Modelo Clasificacion!'

@app.route('/Clasificacion/default/', methods=['GET','POST'])
def default():
	print (request.args)
	# Show
	image_name = request.args.get("imagen")
	test_image_path = '../samples/'+image_name
	test_image = image.load_img(test_image_path)
	plt.imshow(test_image)
	plt.show()
	test_image = image.load_img(test_image_path,target_size = (50, 50))
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis = 0)
	
	with graph.as_default():
	result = loaded_model.predict(test_image)
		if result[0][0] == 1:
    			print(result[0][0], ' --> Es un perro')
		else:
    			print(result[0][0], ' --> Es un gato ')

# Run de application
app.run(host='0.0.0.0',port=5000)
