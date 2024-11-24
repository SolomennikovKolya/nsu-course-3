# Устанавливаем минимальную версию CMake
cmake_minimum_required(VERSION 3.10)

# Указываем имя проекта
project(UthreadProject)

# Включаем поддержку динамических библиотек
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Указываем каталог с заголовочными файлами
include_directories(uthread)

# Добавляем динамическую библиотеку uthread
add_library(uthread SHARED uthread/uthread.c)

# Добавляем исполняемый файл main
add_executable(main main.c)

# Связываем библиотеку uthread с исполняемым файлом
target_link_libraries(main uthread)
