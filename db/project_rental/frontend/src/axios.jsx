import axios from 'axios';

// Создание экземпляра axios (предоставляет интерфейс, чтобы делать HTTP-запросы из браузера)
// baseURL - все запросы будут отправляться на http://localhost:5000/...
// withCredentials: true - включает передачу кук (важно для аутентификации, если используется HttpOnly cookie)
const instance = axios.create({
    baseURL: 'http://localhost:5000',
    withCredentials: true,
});

// Настраиваем автоматическую подстановку куки и токенов; обновляем токен, если он истёк
// Если запрос успешен - просто возвращаем ответ
// Если ошибка - обновляем JWT токен, если он истёк
// Если Refresh токен тоже истёк или невалиден - пробрасываем ошибку дальше (возвращаем ошибочный промис)
// Если ошибка не связана с истёкшим токеном - просто прокидываем её дальше
// interceptor - Перехватчик ответов
instance.interceptors.response.use(
    res => res,
    async err => {
        if (err.response?.status === 401 && err.response.data.msg === 'Access token expired') {
            try {
                const refreshRes = await instance.post('/refresh');
                const newToken = refreshRes.data.access_token;
                err.config.headers['Authorization'] = `Bearer ${newToken}`;
                return instance(err.config);

            } catch (refreshErr) {
                return Promise.reject(refreshErr);
            }
        }
        return Promise.reject(err);
    }
);

export default instance;
