import { slugify } from 'transliteration';

// Конвертация текста в текст, пригодный для url
export const convertToSlug = (text) => {
    return slugify(text)                // Используем transliteration для транслита
        .toLowerCase()                  // Приводим к нижнему регистру
        .replace(/\s+/g, '-')           // Заменяем пробелы на дефисы
        .replace(/[^a-zA-Z0-9\-]/g, '') // Убираем все спецсимволы
};
