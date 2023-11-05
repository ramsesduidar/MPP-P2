CC = mpicc
CFLAGS = -O3 -Wall -std=c99 -g

N = 
N_GEN = 
T_POB = 
M_RATE =
NGM =
NEM =

EXEC = sec
OUTFILE = ../output/out.txt
C_FILES = main.c io.c ga.c crear.c

sec: $(C_FILES)
	$(CC) $(CFLAGS) $(C_FILES) -o $(EXEC) -DTIME -DDEBUG

test_sec:
	mpirun -np 5 ./$(EXEC) $(N) $(N_GEN) $(T_POB) $(M_RATE) $(NGM) $(NEM) >> $(OUTFILE)

clean:
	rm -f $(EXEC)
