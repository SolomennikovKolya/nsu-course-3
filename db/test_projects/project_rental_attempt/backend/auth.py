def authenticate_user(mysql, username, password):
    cursor = mysql.connection.cursor()
    
    # Используем встроенные средства MySQL для проверки пароля
    query = "SELECT id, username, role FROM users WHERE username = %s AND password = SHA2(%s, 256)"
    cursor.execute(query, (username, password))
    
    user = cursor.fetchone()
    cursor.close()
    
    return user
