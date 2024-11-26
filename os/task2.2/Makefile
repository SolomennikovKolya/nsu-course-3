TARGET_1 = queue-async.out
SRCS_1 = main.c queue-async.c

TARGET_2 = queue-spinlock.out
SRCS_2 = main.c queue-spinlock.c

TARGET_3 = queue-mutex.out
SRCS_3 = main.c queue-mutex.c

TARGET_4 = queue-condition.out
SRCS_4 = main.c queue-condition.c

TARGET_5 = queue-semaphore.out
SRCS_5 = main.c queue-semaphore.c

CC=gcc
RM=rm
CFLAGS= -g -Wall
LIBS=-lpthread
INCLUDE_DIR="."

# Параметры программы
MAX_COUNT=1000000
READER_CORE=1
WRITER_CORE=2

all: ${TARGET_1} ${TARGET_2} ${TARGET_3} ${TARGET_4} ${TARGET_5}

run-async: ${TARGET_1}
	./${TARGET_1} ${MAX_COUNT} ${READER_CORE} ${WRITER_CORE}
run-spinlock: ${TARGET_2}
	./${TARGET_2} ${MAX_COUNT} ${READER_CORE} ${WRITER_CORE}
run-mutex: ${TARGET_3}
	./${TARGET_3} ${MAX_COUNT} ${READER_CORE} ${WRITER_CORE}
run-condition: ${TARGET_4}
	./${TARGET_4} ${MAX_COUNT} ${READER_CORE} ${WRITER_CORE}
run-semaphore: ${TARGET_5}
	./${TARGET_5} ${MAX_COUNT} ${READER_CORE} ${WRITER_CORE}

${TARGET_1}: queue.h ${SRCS_1}
	${CC} ${CFLAGS} -I${INCLUDE_DIR} ${SRCS_1} ${LIBS} -o ${TARGET_1}
${TARGET_2}: queue.h ${SRCS_2}
	${CC} ${CFLAGS} -I${INCLUDE_DIR} ${SRCS_2} ${LIBS} -o ${TARGET_2}
${TARGET_3}: queue.h ${SRCS_3}
	${CC} ${CFLAGS} -I${INCLUDE_DIR} ${SRCS_3} ${LIBS} -o ${TARGET_3}
${TARGET_4}: queue.h ${SRCS_4}
	${CC} ${CFLAGS} -I${INCLUDE_DIR} ${SRCS_4} ${LIBS} -o ${TARGET_4}
${TARGET_5}: queue.h ${SRCS_5}
	${CC} ${CFLAGS} -I${INCLUDE_DIR} ${SRCS_5} ${LIBS} -o ${TARGET_5}

clean:
	${RM} -f *.o ${TARGET_1} ${TARGET_2} ${TARGET_3} ${TARGET_4} ${TARGET_5}
