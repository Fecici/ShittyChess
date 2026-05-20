CC = gcc
CFLAGS = -std=c99 -Wall -Wextra -Wshadow -Wconversion -Iinclude
DEBUGFLAGS = -g -O0 -DDEBUG
OPTIMIZED = -O3 -DNDEBUG
SANITIZE = -fsanitize=address,undefined

SRC = \
	src/core/definitions.c \
	src/main.c \
	src/cli/cli.c \
	src/cli/ui.c \
	src/core/fen.c \
	src/core/movegen.c \
	src/core/move.c \
	src/core/history.c \
	src/core/legal.c \
	src/core/parse.c \
	src/util/bitUtils.c \
	src/util/printUtils.c \
	src/util/zobrist.c \
	src/core/command.c \
	src/engine/search.c \
	src/engine/eval.c \
	src/engine/engine.c

OUT = shittychess.exe
DOUT = DEBUG_shittychess.exe

all: debug

debug:
	$(CC) $(CFLAGS) $(DEBUGFLAGS) $(SRC) -o $(OUT)

optimized:
	$(CC) $(CFLAGS) $(OPTIMIZED) $(SANITIZE) $(SRC) -o $(OUT)

clean:
	del $(OUT)
