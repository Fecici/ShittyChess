#ifndef COMMAND_HEADER
#define COMMAND_HEADER

#define MAX_ARG 5  // how am i going to get more than 5 arguments for a single thing??? im wayy too lazy
#define MAX_STDIN 256
#define MAX_CMD_NAME 32

#include "definitions.h"
#include "move.h"
#include "history.h"
#include "fen.h"
#include "parse.h"
#include "printUtils.h"

typedef struct Game Game;  // incomplete type, because we only pass around pointers in command.c

typedef int (*Cmd)(int argc, char** argv);

typedef struct {
    char* name;
    Cmd cmd;
} CommandAbstract;

void setCommandGame(Game* g);
int getCommandCount(void);
CommandAbstract* getCommand(char input[], int nCmds);

// these handle the formatting and arg processing before calling the functions they map to
int cmd_undo(int argc, char** argv);
int cmd_move(int argc, char** argv);
int cmd_perft(int argc, char** argv);
int cmd_children(int argc, char** argv);
int cmd_quit(int argc, char** argv);
int cmd_resign(int argc, char** argv);
int cmd_help(int argc, char** argv);
int cmd_fen(int argc, char** argv);
int cmd_moves(int argc, char** argv);
int cmd_hist(int argc, char** argv);
int cmd_eval(int argc, char** argv);
int cmd_hash(int argc, char** argv);
int cmd_att(int argc, char** argv);
int cmd_pins(int argc, char** argv);
int cmd_checkers(int argc, char** argv);
int cmd_board(int argc, char** argv);

#endif
