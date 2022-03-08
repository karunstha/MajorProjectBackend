from flask import Flask
import os
import urllib.request
from app import app
from flask import Flask, request, redirect, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from utils import save_image
from datetime import datetime
from enhance import *

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
  
@app.route('/')
def hello_world():
    return 'Enhanced Super Resolution Generative Adversarial Network'

@app.route('/enhance', methods=['POST'])
def upload_file():
	if 'file' not in request.files:
		resp = jsonify({'message' : 'No file part in the request'})
		resp.status_code = 400
		return resp
	file = request.files['file']
	if file.filename == '':
		resp = jsonify({'message' : 'No file selected for uploading'})
		resp.status_code = 400
		return resp
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		extension = filename.rsplit('.', 1)[1].lower()
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		newFileName = datetime.now().strftime("%Y%m%d%H%M%S%f")
		newFileNameWithExtension = newFileName + "." + extension
		newFilePath = os.path.join(app.config['UPLOAD_FOLDER'], newFileNameWithExtension)
		os.rename(os.path.join(app.config['UPLOAD_FOLDER'], filename), newFilePath)

		withDownsample = request.form.get('downsample')
		print(withDownsample)
		if withDownsample is not None and withDownsample == "true":
			print("Enhancing with Downsample")
			enhance_with_downsample(newFilePath)
		else:
			enhance(newFilePath)

		resp = jsonify({'enhanced' : 'http://15.206.253.19:8080/super_res/'+newFileName+".jpg"})
		resp.status_code = 201
		return resp
	else:
		resp = jsonify({'message' : 'Allowed file types are png, jpg, jpeg'})
		resp.status_code = 400
		return resp

@app.route('/super_res/<path:path>')
def send_js(path):
    return send_from_directory('super_res', path)

  
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)