
#### Подключение через SSH
1. Проверьте наличие SSH-ключа 
	- `ls -al ~/.ssh`
	- Вы должны увидеть файлы с именами `id_rsa` и `id_rsa.pub`
	- Если их нет, вам нужно создать SSH-ключ.
2. Создайте SSH-ключ
	- `ssh-keygen -t ed25519 -C "your_email@example.com"`
	- Укажите название `id_rsa`
	- После создания ключа появятся файлы `id_rsa` и `id_rsa.pub`
	- Можно переместить ключи в `~/.ssh` чтобы не потерять их
		- `mv id_rsa ~/.ssh/id_rsa`
		- `mv id_rsa.pub ~/.ssh/id_rsa.pub`
3. Добавьте SSH-ключ в агент SSH
	- Чтобы активировать ключ, добавьте его в SSH-агент:
	- `eval "$(ssh-agent -s)" ssh-add ~/.ssh/id_rsa
	- `ssh-add ~/.ssh/id_rsa`
4. Скопируйте публичный SSH-ключ и добавьте его в GitHub
	- `cat ~/.ssh/id_rsa.pub`
	- Перейдите в настройки GitHub SSH and GPG keys и добавьте ключ
5. Проверьте подключение
	- `ssh -T git@github.com`

#### Клонирование репозитория
1. Настройка git:
	- `git config --global user.name "Ваше Имя"` - сохранение вышего имени в переменных гита
	- `git config --global user.email "ваш.email@пример.com"` - сохранение почты
	- `git config --list` - проверка настроек