import axios from 'axios';

// Создание экземпляра axios (предоставляет интерфейс, чтобы делать HTTP-запросы из браузера)
// baseURL — все запросы будут отправляться на http://localhost:5000/...
// withCredentials: true — включает передачу кук (важно для аутентификации, если используется HttpOnly cookie)
const instance = axios.create({
    baseURL: 'http://localhost:5000',
    withCredentials: true,
});

// Настраиваем автоматическое обновление access token, если он истёк
instance.interceptors.response.use(
    // Если запрос успешен — просто возвращаем ответ
    res => res,

    // Если ошибка — проверяем, нужно ли обновлять токен
    async err => {
        if (err.response?.status === 401 && err.response.data.msg === 'Access token expired') {
            try {
                // Отправляем запрос на обновление токена и получаем новый access token
                const refreshRes = await instance.post('/refresh');
                const newToken = refreshRes.data.access_token;

                // Повторяем оригинальный запрос с новым токеном
                err.config.headers['Authorization'] = `Bearer ${newToken}`;
                return instance(err.config);
            } catch (refreshErr) {
                // Если refresh token тоже невалиден — разлогиниваем пользователя
                return Promise.reject(refreshErr);
            }
        }
        // Если ошибка не связана с истёкшим токеном — просто прокидываем её дальше
        return Promise.reject(err);
    }
);

export default instance;