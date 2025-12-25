#ifndef CLI_HEADER
#define CLI_HEADER

#include "definitions.h"
#include "printUtils.h"
#include "ui.h"
#include "engine.h"

typedef enum {human, engine} PlayerType;
enum Command {

    MOVE,
    UNDO,
    PERFT,
    CHILDREN,
    QUIT,
    RESIGN,
    PRINT_BOARD,
    PRINT_ZOBRIST,
    PRINT_BITBOARD
};

typedef struct {

    PlayerType playerType;
    uint8_t colour;  // 0 or 1
    Engine* engine;  // null if playertype is human

} Player;

typedef struct {

    UI ui;
    Board* board;
    Player white, black;
    unsigned int moves;  // 2ply = 1 move
    unsigned int ply;
    uint8_t gameResult;  // format to be defined, but basically its a flag that describes how the game ended

} Game;

void initPlayer(Player* p, PlayerType type, uint8_t colour, Engine* engine);
void initGame(Game* game, UI ui);
void getCommand();
void handleCommand();
void setUI(Game* game, UI ui);
void checkTermination(Board* b);
void handleStalemate(Board* b);
void handleCheckmate(Board* b);

void cliMainLoop(Game* game, void (*performCommand)(Board* b));
Move getmove(Board* b);
bool isValid(Board* b, Move move); 
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