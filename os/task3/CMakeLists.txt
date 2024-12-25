cmake_minimum_required(VERSION 3.10)

project(proxy LANGUAGES C)                     # Имя проекта
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -Wall") # Установка опций компиляции
include_directories("${CMAKE_SOURCE_DIR}")     # Добавление директории с заголовочными файлами
set(SRCS proxy.c)                              # Установка исходных файлов
set(LIBS pthread curl)                         # Библиотеки
add_executable(proxy.out ${SRCS})              # Добавление исполняемого файла
target_link_libraries(proxy.out ${LIBS})       # Связывание библиотек

# Папка для сборки:        mkdir build && cd build
# Генерация файлов сборки: cmake -DCMAKE_BUILD_TYPE=Debug ..
# Сборка:                  cmake --build .
# Запуск:                  ./proxy.out
