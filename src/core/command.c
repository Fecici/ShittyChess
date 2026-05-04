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
            // 
            mv = (Move) strtol(mvHex, NULL, 0);
        }
    }

    if (mv != 0) {
        if (visual) {

            // perform visual cmd in move.c

            Undo* undo = getUndoFromMove(b, mv);
            performMove(b, mv);
            printBoard(b);
            performUndo(b, undo);
            return 0;
        }

        if (!force) { 
            // idk lol
            // maybe another move.c wrapper here
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
    // --default, prints board fen, -l to load fen to board.

    if (argc <= 1) {
        char* fen = convertToFen(game->board);
        printf("%s\n", fen);
        free(fen);
        return 0;
    }

    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--default", 9) == 0) {
            char* fen = convertToFen(game->board);
            printf("%s\n", fen);
            free(fen);
            return 0;
        }
        if (strncmp(argv[i], "-l", 2) == 0) {
            if (i + 1 < argc) {
                char* fen = argv[++i];
                if (!validFen(fen)) {
                    fprintf(stderr, "Error: Invalid FEN string\n");
                    return 1;
                }
                loadFromFen(game->board, fen);
                return 0;
            } else {
                fprintf(stderr, "Error: -l option requires a FEN string argument\n");
                return 1;
            }
        }
    }
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
    ///TODO: there are some more specs that i want implemented later, found in the txt
    // -b for bitboard, optional -p to specify piece, capitalizd for white under normal fen notation, 
    // -bx for bitboard in hex -g gamestate, -gx gamestate hex. non x is printed in binary by default
    // the bitboards with unspecified pieces otherwise are labeled and all are printed
    // if -s passed after -b in args, print square bitboard
    // -z is zobrist (already hex). print immediately and exit

    if (argc <= 1) {
        printBoard(game->board);
        return 0;
    }

    Piece piece = EMPTY;  // default
    uint8_t pieceIndex = 0;
    uint64_t bitboard = 0;

    bool flag_b = false;
    bool flag_x = false;
    bool specifiedPiece = false;
    bool make_square = false;

    for (int i = 0; i < argc; i++) {

        if (strncmp(argv[i], "--default", 9) == 0) {
            printBoard(game->board);
            return 0;
        }

        if (strncmp(argv[i], "-z", 2) == 0) {
            printZobrist(game->board);
            return 0;
        }

        if (strncmp(argv[i], "-b", 2) == 0) {

            flag_b = true;
            continue;
        }

        // ply --ply
        if (strncmp(argv[i], "--ply", 5) == 0) {
            printf("Ply: %d\n", game->board->ply);
            return 0;
        }

        // print bitboard
        // check for piece specifier
        // check for -p flag
        if (strncmp(argv[i], "-bx", 3) == 0) {
            flag_b = true;
            flag_x = true;
            continue;
        }

        if (strncmp(argv[i], "-g", 2) == 0) {
            printGameState(game->board, make_square);
            return 0;
        }
        
        if (flag_b) {
            if (strncmp(argv[i], "-p", 2) == 0) {
                if (i + 1 < argc) {
                    char* pieceStr = argv[++i];
                    if (!isValidPiece(pieceStr[0])) {
                        fprintf(stderr, "Error: Invalid piece specifier\n");
                        return 1;
                    }
                    piece = getPieceFromChar(pieceStr[0]);
                    pieceIndex = getBitboardIndex(piece);
                    bitboard = game->board->bitboards[pieceIndex];
                    specifiedPiece = true;
                } else {
                    fprintf(stderr, "Error: -p option requires a piece specifier argument\n");
                    return 1;
                }
            } else if (!flag_x && strncmp(argv[i], "-s", 2) == 0) {
                make_square = true;
                continue;
            }
        }
            
        
    }
    // if the user is retarded and puts in multiple flags, we let fate decide
    if (flag_b && !flag_x) {
        if (specifiedPiece) {
            printBitBoard(bitboard, getPieceNameFromIndex(pieceIndex), make_square);  // i fucking forgot to fix the name
            return 0;
        }
        
        printBitboards(game->board, make_square);
        return 0;

    } else if (flag_b && flag_x) {

        if (specifiedPiece) {
            printBitboardHex(bitboard, getPieceNameFromIndex(pieceIndex));
            return 0;
        }
        
        printBitboardHexAll(game->board);
        return 0;

    } else {
        fprintf(stderr, "Error: Invalid flag combination\n");
        return 1;
    }

    return 0;
}
