from flask import Flask, jsonify
from flask_cors import CORS  # Только для разработки!

app = Flask(__name__)

# Разрешаем CORS только для разработки!
# В production это должно настраиваться на веб-сервере (Nginx)
CORS(app)


@app.route('/api/hello')
# Пример API endpoint
def hello():
    return jsonify({'message': 'Hello from Flask!'})


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
# Для обслуживания React сборки
def serve(path):
    return app.send_static_file('index.html')


if __name__ == '__main__':
    app.run(debug=True)
