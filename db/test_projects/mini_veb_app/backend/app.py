from flask import Flask, jsonify
from flask_cors import CORS  # Только для разработки!

app = Flask(__name__)

# Разрешаем CORS только для разработки!
# В production это должно настраиваться на веб-сервере (Nginx)
CORS(app)

# Пример API endpoint
@app.route('/api/hello')
def hello():
    return jsonify({'message': 'Hello from Flask!'})

# Для обслуживания React сборки
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(debug=True)
