CC = gcc
CFLAGS = -std=c99 -Wall -Wextra -Wshadow -Wconversion -Iinclude
DEBUGFLAGS = -g -O0 -DDEBUG
OPTIMIZED = -O3 -DNDEBUG
SANITIZE = -fsanitize=address,undefined

SRC = \
	src/main.c \
	src/cli/cli.c \
	src/cli/ui.c \
	src/util/bitUtils.c \
	src/util/printUtils.c \
	src/util/zobrist.c

OUT = shittychess.exe

all: debug

debug:
	$(CC) $(CFLAGS) $(DEBUGFLAGS) $(SRC) -o $(OUT)

optimized:
	$(CC) $(CFLAGS) $(OPTIMIZED) $(SANITIZE) $(SRC) -o $(OUT)

clean:
	del $(OUT)