from flask import Flask, request, Response
from flask_uploads import UploadSet, IMAGES, configure_uploads
import jsonpickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

app = Flask(__name__)

# Load the model
model = load_model('model.h5')

# Configure the app to store uploaded files in the 'uploads' folder
app.config['UPLOADS_DEFAULT_DEST'] = 'uploads'

# Create an UploadSet for handling image uploads
images = UploadSet('images', IMAGES)

# Configure the Flask-Uploads extension
configure_uploads(app, (images,))

@app.route('/api/test', methods=['GET'])
def test():
    response = {'message': 'API hit'}
    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")

@app.route('/api/testmodel', methods=['POST'])
def testmodel():
    # Check if a file was uploaded
    if 'image' not in request.files:
        return 'No file uploaded', 400

    # Save the uploaded file
    file = request.files['image']
    file_path = images.save(file)

    # Load the image and prepare it for the model
    img = image.load_img(file_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Make a prediction
    preds = model.predict(x)
    prediction = preds[0][0]

    # Return the prediction
    if prediction > 0.6:
        response = {'prediction': 'gun'}
    else:
        response = {'prediction': 'not_gun'}

    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")

@app.route('/api/upload', methods=['POST'])
def upload():
    # Check if a file was uploaded
    if 'image' not in request.files:
        return 'No file uploaded', 400

    # Save the uploaded file
    file = request.files['image']
    file_path = images.save(file)

    response = {'message': 'File uploaded successfully', 'file_path': file_path}
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
