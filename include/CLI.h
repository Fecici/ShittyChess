#ifndef CLI_HEADER
#define CLI_HEADER

#include "definitions.h"
#include "ui.h"
#include "engine.h"
#include "history.h"
#include "command.h"
#include "fen.h"
#include "parse.h"
#include "printUtils.h"

typedef enum {HUMAN, ENGINE} PlayerType;

typedef struct {

    PlayerType playerType;
    Colour colour;  // 0 or 1
    Engine* engine;  // null if playertype is human

} Player;

typedef struct Game {

    UI ui;
    Board* board;
    Player white, black;
    unsigned int moves;  // 2ply = 1 move
    /// TODO: time control eventually
    int whiteTime, blackTime;
    uint8_t gameResult;  // format to be defined, but basically its a flag that describes how the game ended
} Game;


Game* initGame(char* fen, Player white, Player black, GameType gt);  // init all, setup history, ui, etc.

bool checkTermination(Board* b);
void handleStalemate(Board* b);
void handleCheckmate(Board* b);

void cliMainLoop(Game* game, void (*performCommand)(Board* b));
Move getMove(Board* b, Player player);  // ignore this, use the cli cmd instead get_move
void handleIllegal();

// one of these is chosen for the performCommand pointer
void __DEBUG_performCommand(Board* b);
void noDebugGetMove(Board* b);

void handleQuit();


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
