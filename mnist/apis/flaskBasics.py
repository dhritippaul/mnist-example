from flask import Flask , redirect , url_for , render_template
from flask.globals import request

app = Flask(__name__)


@app.route("/Detector", methods=["POST","GET"])
def detector():
    if request.method == "POST": 
        return render_template("output.html")

    else:
        return render_template("index.html")



if __name__ == "main":
    app.run()

