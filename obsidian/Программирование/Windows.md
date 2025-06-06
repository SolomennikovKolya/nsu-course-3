
#### Основные режимы питания
1. **Рабочий режим (Normal operation)**
	- Система работает в полную мощность, все компоненты активны
	- Используется при обычной работе за ПК/ноутбуком
2. **Режим сна (Sleep / Suspend-to-RAM, S3)**
	- Состояние: ПК потребляет минимум энергии, но остается включенным
	- Что сохраняется: Все данные в оперативной памяти (RAM)
	- Выход: Быстрое пробуждение (1-2 сек.) нажатием кнопки или движением мыши
	- Когда используется: При кратковременных перерывах в работе
3. **Гибернация (Hibernation / Suspend-to-Disk, S4)**
	- Состояние: Полное выключение, но с сохранением состояния системы на диск (`hiberfil.sys`)
	- Что сохраняется: Все открытые программы и данные
	- Выход: Медленнее, чем сон (зависит от скорости SSD/HDD)
	- Когда используется: Для ноутбуков при долгом бездействии (чтобы не разряжать батарею)
3. **Гибридный спящий режим (Hybrid Sleep)**
	- Комбинация сна + гибернации
	- Данные сохраняются и в RAM, и на диск
	- Если отключится питание — система восстановится из `hiberfil.sys`
	- По умолчанию включен для настольных ПК
5. **Быстрый запуск (Fast Startup)**
	- Ускоренная загрузка после выключения
	- Ядро системы и драйверы сохраняются в `hiberfil.sys`
	- Неполная гибернация (пользовательские приложения не сохраняются)
