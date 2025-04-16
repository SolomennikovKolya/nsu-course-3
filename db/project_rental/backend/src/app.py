from flask import Flask
from flask_cors import CORS
from auth import auth_bp

app = Flask(__name__)
CORS(app)
app.register_blueprint(auth_bp, url_prefix='/api/auth')


@app.route("/api/ping")
def ping():
    return {"message": "pong"}


if __name__ == "__main__":
    app.run(debug=True)
