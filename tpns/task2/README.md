
### Многослойный персептрон

Задачи:
1. Посмотреть библиотечную реализацию персептрона - регрессора
    - Протестировать и настроить его на очищенном датасете из 1-ого задания
    - Построить графики: предсказание vs реальность, распределение ошибок, изменение ошибки по эпохам, матрица ошибок
    - Дополнительно: попробовать oversampling и undersampling
2. Сделать ручную реализацию многослойного персептрона
    - Реализовать алгоритм обратного распространения ошибки для обучения сети
    - Сравнить с библиотечной реализацией
    - Сделать вывод ошибки по ходу обучения

### Команды
- `python main.py` - запуск с дефолтными параметрами
- `python main.py -h` - посмотреть подсказку по использованию
- `set LOKY_MAX_CPU_COUNT=4` - чтобы не вылезало предупреждение (опционально)
