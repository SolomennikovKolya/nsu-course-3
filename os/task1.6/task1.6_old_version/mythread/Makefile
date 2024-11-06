CC = gcc
CFLAGS = -fPIC -pthread -D_GNU_SOURCE
LDFLAGS = -shared
LIBNAME = libmythread.so
SOURCES = mythread.c mythreadmap.c
OBJECTS = $(SOURCES:.c=.o)

all: $(LIBNAME)

$(LIBNAME): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

# Транслирует все исходники в объектники
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(LIBNAME) $(OBJECTS)
