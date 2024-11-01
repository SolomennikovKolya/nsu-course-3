CC = gcc
CFLAGS = -Wall
LDFLAGS = -L./mythread -lmythread
SOURCES = main.c
TARGET = main.out
OBJECTS = $(SOURCES:.c=.o)

# Объявляем зависимость от библиотеки
LIBRARY = mythread/libmythread.so
LIB_SOURCES = mythread/mythread.c

all: run

run: build
	./$(TARGET)

build: $(OBJECTS) $(LIBRARY)
	$(CC) -o $(TARGET) $^ $(LDFLAGS)

# Правило для компиляции библиотеки
$(LIBRARY): $(LIB_SOURCES)
	$(MAKE) -C mythread

clean:
	rm -f $(OBJECTS) $(TARGET)
	$(MAKE) -C mythread clean
