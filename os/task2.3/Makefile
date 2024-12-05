TARGET_1 = list-spinlock.out
SRCS_1 = list-spinlock.c
TARGET_2 = list-mutex.out
SRCS_2 = list-mutex.c
TARGET_3 = list-rwlock.out
SRCS_3 = list-rwlock.c

CC=gcc
RM=rm
CFLAGS= -g -Wall
LIBS=-lpthread
INCLUDE_DIR="."

# Параметры программы
STORAGE_SIZE=1000
SWAP_CHANCE=1

all: ${TARGET_1} ${TARGET_2} ${TARGET_3}

run-spinlock: ${TARGET_1}
	./${TARGET_1} ${STORAGE_SIZE} ${SWAP_CHANCE}
run-mutex: ${TARGET_2}
	./${TARGET_2} ${STORAGE_SIZE} ${SWAP_CHANCE}
run-rwlock: ${TARGET_3}
	./${TARGET_3} ${STORAGE_SIZE} ${SWAP_CHANCE}

${TARGET_1}: ${SRCS_1}
	${CC} ${CFLAGS} -I${INCLUDE_DIR} ${SRCS_1} ${LIBS} -o ${TARGET_1}
${TARGET_2}: ${SRCS_2}
	${CC} ${CFLAGS} -I${INCLUDE_DIR} ${SRCS_2} ${LIBS} -o ${TARGET_2}
${TARGET_3}: ${SRCS_3}
	${CC} ${CFLAGS} -I${INCLUDE_DIR} ${SRCS_3} ${LIBS} -o ${TARGET_3}

clean:
	${RM} -f *.o ${TARGET_1} ${TARGET_2} ${TARGET_3}
