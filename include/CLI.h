#ifndef CLI_HEADER
#define CLI_HEADER

#define MAX_ARG 5  // how am i going to get more than 5 arguments for a single thing??? im wayy too lazy
#define MAX_STDIN 256
#define MAX_CMD_NAME 32

#include "definitions.h"
#include "printUtils.h"
#include "ui.h"
#include "engine.h"
#include "zobrist.h"

typedef enum {HUMAN, ENGINE} PlayerType;
typedef int (*Cmd)(int argc, char** argv);

typedef struct {
    char* name;
    Cmd cmd;
} CommandAbstract;

// Command stuff
void printHelp();
void printLegalMoves(Board* b);
void printHistory(History* h);
void printEval(Board* b);  // eval will be written somewhere else, this is a printing wrapper
void printAttacksFromSquare(Board* b, Square sq);
void printPinsBitboards(Board* b);
void printCheckersBitboards(Board* b);

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

typedef struct {

    PlayerType playerType;
    Colour colour;  // 0 or 1
    Engine* engine;  // null if playertype is human

} Player;

typedef struct {

    uint64_t hashHistory[MAX_PLY];
    Move moveHistory[MAX_PLY];
    Undo undoHistory[MAX_PLY];

    
} History;

typedef struct {

    UI ui;
    Board* board;
    Player white, black;
    unsigned int moves;  // 2ply = 1 move
    unsigned int ply;
    /// TODO: time control eventually
    unsigned int whiteTime, blackTime;
    uint8_t gameResult;  // format to be defined, but basically its a flag that describes how the game ended
    History history;
} Game;

void initGame(Game* game, const char* fen, Player white, Player black, GameType gt);  // init all, setup history, ui, etc.
static CommandAbstract* getCommand(char input[], int nCmds);
static int tokenize(char* line, char** argv);
static void getInput(char input[]);



bool checkTermination(Board* b);
void handleStalemate(Board* b);
void handleCheckmate(Board* b);

bool loadFromFen(Board* b, const char* fen);
bool isCharInt(const char c);
unsigned int getPieceFromChar(const char c);
char* convertToFen(Board* b);

void cliMainLoop(Game* game, void (*performCommand)(Board* b));
Move getmove(Board* b, Player player);
bool isValidMove(Board* b, Move move); 
void handleIllegal();

void performUndo(Board* b, Undo undo);
void performMove(Board* b, Move move);

// one of these is chosen for the performCommand pointer
void __DEBUG_performCommand(Board* b);
void noDebugGetMove(Board* b);

// commands:
void handleUndo(Board* b, Undo undo);
void handleMakeMove(Board* b, Move move);
void handlePerft(Board* b);
void handleChildren(Board* b);
void handleQuit();
void handleResign(Board* b);

// print stuff is found in print utils


/*
loop:   
    check termination conditions
        handle stalemate
        handle checkmate
    
    get colour to move

    IF DEBUG: CHANGE THIS LINE OF CODE - TEMPLATE METHOD 
    {
        get player/engine move
        get player CMD

        validation happens in this step
        if illegal:
            handle illegal
    }

    render board
    goto loop
*/


#endif