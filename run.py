# Python Flask application for TensorFlow demo
# AUTHOR: Dattaraj J Rao - dattaraj.rao@ge.com

# Import Flask for building web app
from flask import Flask, request

# Other standard imports in Python
import os
import ssl
import six.moves.urllib as urllib
import cv2
import random

# Import our method for running model
from predict import run_model

# Get the port for starting web server
port = os.getenv("PORT")
if port is None:
    port = 3000

# Create the Flask app
app = Flask(__name__, static_url_path='')

# Initialize SSL context to read URLs with HTTPS
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

#### DEFINE Flask Routes ####

# Show display page by default
@app.route('/')
def default():
    return app.send_static_file('display.html')

# Handle the request when URL is passed
@app.route('/imgurl', methods=['POST', 'GET'])
def imgurl():
    # read the parameter
    myimg = request.args['imgurl']
    # open url and read image to file
    opener = urllib.request.URLopener(context=ctx)
    opener.retrieve(myimg, "static/test.jpg")
    # read the image using PIL
    image_np = cv2.imread("static/test.jpg")
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    # Call method to run Caffe model on the image
    result = run_model(image_np)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)	
    result = cv2.resize(result, (800,600))	
    cv2.imwrite("static/result.jpg", result)
    ret_str = ""	
    # prepare HTML string to write back
    ret_str = ret_str + "<br><hr><b>Analyzed Image</b><br><img src='result.jpg?%s'><br>"%str(random.random())	
    ret_str = ret_str + "<br><hr><a href='/'>BACK</a>"

    return ret_str    

# Run the application and start server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)