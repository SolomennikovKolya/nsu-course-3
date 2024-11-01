CC = gcc
CFLAGS = -fPIC -Wall -pthread
LDFLAGS = -shared
LIBNAME = libmythread.so
SOURCES = mythread.c
OBJECTS = $(SOURCES:.c=.o)

all: $(LIBNAME)

$(LIBNAME): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(LIBNAME) $(OBJECTS)
