#mac
#CC = gcc-9 -Os -fopenmp 
#linux
CC = gcc -Os -fopenmp
CFLAGS = -L/home/users/yuyifan/opt/sac/lib -lsac -lsacio -lm -mcmodel=large -lgsl -lgslcblas
BIN = ../bin

FDTCC: FDTCC.o sacio.o 
	${CC} -o ${BIN}/$@ $^ ${CFLAGS}
clean:
	rm -f *.o
