from flask import Flask, render_template, Response
from src.app import Inference
#from waitress import serve

app = Flask(__name__, template_folder='../template', static_folder='../images/')
inf = Inference()

# Html page
@app.route('/')
def index():
    return render_template('index.html')
    
# Video feed hook
@app.route('/video_feed')
def video_feed():

    resp = Response(inf.start_video_stream_web(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    return resp

# User terminate video hook
@app.route('/terminate', methods=['POST'])
def terminate():

    inf.video_terminate()
    return render_template('index.html')

# --------------------------------------------------------------
# TODO: need to handle video terminate if user close html window
#       instead of clicking on button
# --------------------------------------------------------------

if __name__ == "__main__":

    app.run(host="127.0.0.1", debug=True, port=8000)
    # For production mode, comment the line above and uncomment below
    #serve(app, host="0.0.0.0", port=8000)
