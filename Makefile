CC=gcc
CFLAGS=-ansi -Wall -Wextra -Werror -pedantic-errors
LDFLAGS=-lm

all: symnmf

symnmf: symnmf.o
	$(CC) $(CFLAGS) -o symnmf symnmf.o $(LDFLAGS)

symnmf.o: symnmf.c symnmf.h
	$(CC) $(CFLAGS) -c symnmf.c

clean:
	rm -f *.o symnmf

.PHONY: all clean
