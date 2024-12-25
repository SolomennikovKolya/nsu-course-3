TARGET = proxy.out
SRCS = proxy.c

CC=gcc
RM=rm
CFLAGS= -g -Wall
LIBS=-lpthread -lcurl
INCLUDE_DIR="."

all: run

run: ${TARGET}
	./${TARGET}

${TARGET}: ${SRCS}
	${CC} ${CFLAGS} -I${INCLUDE_DIR} ${SRCS} ${LIBS} -o ${TARGET}

clean:
	${RM} -f *.o ${TARGET}
