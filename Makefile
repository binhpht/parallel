CFLAGS=-std=c99
LDFLAGS=-lglut -lGL -lm

OPTIMAL_CFLAGS=${CFLAGS} -Ofast -ftree-vectorize -mavx2 -mfma -march=native

parallel_1: parallel.c
	${CC} ${CFLAGS} $? -o $@ ${LDFLAGS}

parallel_2: parallel.c
	${CC} ${CFLAGS} -O2 $? -o $@ ${LDFLAGS}

parallel_3: parallel.c
	${CC} ${CFLAGS} -O2 $? -o $@ ${LDFLAGS} -ftree-vectorize -fopt-info-vec

parallel_4_3: parallel.c
	${CC} ${OPTIMAL_CFLAGS} $? -o $@ ${LDFLAGS} -fopt-info-vec

parallel_6_2_1: parallel_6_2_1.c
	${CC} ${OPTIMAL_CFLAGS} $? -o $@ ${LDFLAGS} -fopt-info-vec

parallel_6_2_1_bad: parallel_6_2_1_bad.c
	${CC} ${OPTIMAL_CFLAGS} $? -o $@ ${LDFLAGS} -fopt-info-vec

parallel_6_2_1_bad_good: parallel_6_2_1_bad.c
		${CC} ${CFLAGS} $? -o $@ ${LDFLAGS}

parallel_6_2_2: parallel_6_2_2.c
	${CC} ${OPTIMAL_CFLAGS} $? -o $@ ${LDFLAGS} -fopt-info-vec

parallel_6_2_2_bad: parallel_6_2_2.c
	${CC} ${CFLAGS} $? -o $@ ${LDFLAGS}

parallel_6_2_3: parallel_6_2_3.c
	${CC} ${OPTIMAL_CFLAGS} $? -o $@ ${LDFLAGS} -fopt-info-vec

parallel_6_2_4: parallel_6_2_4.c
	${CC} ${OPTIMAL_CFLAGS} $? -o $@ ${LDFLAGS} -fopt-info-vec

parallel_6_3_6: parallel_6_3_6.c
	${CC} ${OPTIMAL_CFLAGS} $? -o $@ ${LDFLAGS} -fopt-info-vec

parallel_6_4: parallel_6_4.c
	${CC} ${OPTIMAL_CFLAGS} $? -o $@ ${LDFLAGS} -fopenmp

parallel_6_4_3: parallel_6_4_3.c
	${CC} ${OPTIMAL_CFLAGS} $? -o $@ ${LDFLAGS} -fopt-info-vec -fopenmp

parallel_6_4_3_physicsfunction: parallel_6_4_3_physicsfunction.c
	${CC} ${OPTIMAL_CFLAGS} $? -o $@ ${LDFLAGS} -fopt-info-vec -fopenmp

parallel_6_4_4: parallel_6_4_4.c
	${CC} ${OPTIMAL_CFLAGS} $? -o $@ ${LDFLAGS} -fopt-info-vec -fopenmp

parallel_6_4_4_physicsfunction: parallel_6_4_4_physicsfunction.c
	${CC} ${OPTIMAL_CFLAGS} $? -o $@ ${LDFLAGS} -fopt-info-vec -fopenmp
part2: part2.c
	${CC} ${OPTIMAL_CFLAGS} $? -I/usr/local/cuda-10.1/targets/x86_64-linux/include/ -o $@ ${LDFLAGS} -fopt-info-vec -fopenmp -lOpenCL

part2_original: parallel_part2.c
	${CC} ${CFLAGS} -O3 $? -o $@ ${LDFLAGS} -ftree-vectorize -fopt-info-vec
