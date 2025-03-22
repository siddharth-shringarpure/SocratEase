from flask import Flask
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route("/api/python")
def hello_world():
    return "Hello, World!"

if __name__ == "__main__":
    port = int(os.environ.get('FLASK_RUN_PORT', 5328))
    app.run(debug=True, port=port)