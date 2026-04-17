from flask import Flask, request

app = Flask(__name__)

@app.route('/logger', methods=['GET', 'POST'])
def log():
    print("Received:", request.data.decode())
    return "OK"

app.run(host="0.0.0.0", port=8000)