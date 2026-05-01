#include "cli.h"

// hold data for commands to be checked against by the tokenizer and the getCommand (we check against name and return the cmd)
CommandAbstract cmds[] = {
    {.name = "help", .cmd = cmd_help},
    {.name = "undo", .cmd = cmd_undo},
    {.name = "move", .cmd = cmd_move},
    {.name = "perft", .cmd = cmd_perft},
    {.name = "children", .cmd = cmd_children},
    {.name = "quit", .cmd = cmd_quit},
    {.name = "resign", .cmd = cmd_resign},
    {.name = "fen", .cmd = cmd_fen},
    {.name = "legal-moves", .cmd = cmd_moves},
    {.name = "history", .cmd = cmd_hist},
    {.name = "eval", .cmd = cmd_eval},
    {.name = "hash", .cmd = cmd_hash},
    {.name = "atk", .cmd = cmd_att},
    {.name = "pins", .cmd = cmd_pins},
    {.name = "checkers", .cmd = cmd_checkers},
    {.name = "board", .cmd = cmd_board}

};

static Game* game;  // this will hold the globals we need

/*
CMDS: 
help
fen <string> / startpos
d (display board)
moves (print legal moves in UCI)
perft <depth>
go <depth> or go movetime <ms>
undo / redo
history
eval - likely a 0 for now, or greedy. doesnt matter atm
hash (print zobrist)
att <sq> (print attacks to/from a square)
pins (print pinned pieces mask)
checkers (print checkers mask)
quit

command abstract wrapper mappings
cmd_undo      ---> void handleUndo(Board* b, Undo undo);
cmd_move      ---> void handleMakeMove(Board* b, Move move);
cmd_perft     ---> void handlePerft(Board* b);
cmd_children  ---> void handleChildren(Board* b);
cmd_quit      ---> void handleQuit();
cmd_resign    ---> void handleResign(Board* b);
cmd_help      ---> void prinHelp();
cmd_fen       ---> bool loadFromFen(Board* b, const char* fen);
cmd_moves     ---> void printLegalMoves(Board* b);
cmd_hist      ---> void printHistory(History* h);
cmd_eval      ---> void printEval(Board* b);
cmd_hash      ---> void printZobrist(Board* b);  // written in printUtils
cmd_att       ---> void printAttacksFromSquare(Board* b, Square sq);
cmd_pins      ---> void printPinsBitboards(Board* b);
cmd_checkers  ---> void printCheckersBitboards(Board* b);
cmd_board     ---> void printBoard(Board* b);  // written in printUtils

*/
void setCommandGame(Game* g) {
    game = g;
}

int getCommandCount(void) {
    return (int)(sizeof(cmds) / sizeof(CommandAbstract));
}

CommandAbstract* getCommand(char input[], int nCmds) {

    for (int i = 0; i < nCmds; i++) {
        if (strncmp(input, cmds[i].name, MAX_CMD_NAME) == 0) return &cmds[i];
    }

    return NULL;

}

// these handle the formatting and arg processing before calling the functions they map to
int cmd_undo(int argc, char** argv) {

    // uhhh so i guess the -f literally does nothing lmfao

    // only one flag
    bool force = false;

    Board* b = game->board;
    Gamestack* stack = b->gamestack;
    Undo* undo = &(stack->undoStack[stack->ply]);

    if (argc <= 1) {
        ///TODO: 
        //check that we can undo - would entail checking stack bounds pretty much
        performUndo(b, undo);

        return 0;
    }

    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "-f", 2) != 0) {
            continue;
        }

        else {
            force = true;
        }
    }

    if (!force) {
        ///TODO:
        // check if we can undo
        performUndo(b, undo);

        return 0;
    }

    else {
        // purely just to see what the effect of this undo struct would have on this board struct
        performUndo(b, undo);
    }

    return 0;

}

int cmd_move(int argc, char** argv) {

    Board* b = game->board;

    bool force = false;
    bool visual = false;
    Move mv = 0;

    if (argc <= 1) {
        return 1;
    }

    for (int i = 2; i < argc; i++) {
        if (strncmp(argv[i], "-f", 2) == 0) {
            force = true;
        }

        else if (strncmp(argv[i], "-v", 2) == 0) {
            visual = true;
        }

        else if (strncmp(argv[i], "-m", 2) == 0) {
            char* mvHex = argv[++i];
            mv = (Move) strtol(mvHex, NULL, 0);
        }
    }

    if (mv != 0) {
        if (visual) {
            Undo* undo = getUndoFromMove(b, mv);
            performMove(b, mv);
            printBoard(b);
            performUndo(b, undo);
            return 0;
        }

        if (!force) { 
            // idk lol
            //check legal
            performMove(b, mv);
            return 0;
        }
    }

    // "move e2e4 -f -v -etc"
    char* strMove = argv[1];
    if (!validMoveNotation(strMove)) return 1;

    mv = getMoveFromNotation(b, strMove);
    if (!force) {
        // check legal
        performMove(b, mv);
        return 0;
    }

    if (visual) {
        Undo* undo = getUndoFromMove(b, mv);
        performMove(b, mv);
        printBoard(b);
        performUndo(b, undo);
        return 0;
    }

    else {
        performMove(b, mv);
    }

    return 0;
}

int cmd_perft(int argc, char** argv) {

    (void) argc;
    (void) argv;

    return 0;
}

int cmd_children(int argc, char** argv) {
    (void) argc;
    (void) argv;
    return 0;
}


int cmd_quit(int argc, char** argv) { 
    ///TODO: free heap
    (void) argc; 
    (void) argv; 
    handleQuit(); 
    return 0;
}

int cmd_resign(int argc, char** argv) {
    (void) argc;
    (void) argv;
    return 0;
}
int cmd_help(int argc, char** argv) {
    (void) argc;
    (void) argv;
    return 0;
}
int cmd_fen(int argc, char** argv) {
    (void) argc;
    (void) argv;
    return 0;
}
int cmd_moves(int argc, char** argv) {
    (void) argc;
    (void) argv;
    return 0;
}
int cmd_hist(int argc, char** argv) {
    (void) argc;
    (void) argv;
    return 0;
}
int cmd_eval(int argc, char** argv) {
    (void) argc;
    (void) argv;
    return 0;
}

int cmd_hash(int argc, char** argv) {
    (void) argc;
    (void) argv;
    return 0;
}

int cmd_att(int argc, char** argv) {
    (void) argc;
    (void) argv;
    return 0;
}
int cmd_pins(int argc, char** argv) {
    (void) argc;
    (void) argv;
    return 0;
}
int cmd_checkers(int argc, char** argv) {
    (void) argc;
    (void) argv;
    return 0;
}


int cmd_board(int argc, char** argv) {
    // for now
    (void) argc;
    (void) argv;
    printBoard(game->board);

    return 0;
}
