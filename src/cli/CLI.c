#include "cli.h"

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
///TODO: these need to be added to the header file as signatures
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
int cmd_quit(int argc, char** argv) { handleQuit(); }
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


int cmd_board(int argc, char** argv) {
    // for now
    printBoard(game->board);
}

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

Game* game;  // this will hold the globals we need

 // init all, setup history, ui, etc.
void initGame(Game* game, const char* fen, Player white, Player black, GameType gt) {

    /*
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
} Game;*/

    Board* b = (Board*) malloc(sizeof(Board));

    if (b == NULL) {fprintf(stderr, "Failed to allocate memory for board.\n"); exit(1); }

    game->board = b;

    game->white = white;
    game->black = black;

    if (!loadFromFen(b, fen)) {
        fprintf(stderr, "Failed to parse fen string: %s\n", fen);
        exit(1);
    }

    unsigned int ply = b->ply;
    game->ply = ply;
    game->moves = (ply >> 1) + 1;

    game->whiteTime = -1;
    game->blackTime = -1;

    game->gameResult = 0;

    UI ui;
    initUI(&ui, "CLI", gt, ascii_render, stdout_messager);

    game->ui = ui;

    return;

}

// this needs to turn all whitespace into a '\0' and count the args. this also mutates argv
static int tokenize(char* line, char** argv) {

    int argc = 0;
    char* split = line;
    while (*split) {
        // skip until whitespace
        while (!isspace((unsigned char) *split)) split++;
        
        if (!*split) break;
        if (argc >= MAX_ARG) return argc;

        argv[argc] = split;
        argc++;

        // fill whitespace with \0
        while (*split && !isspace((unsigned char) *split)) split++;
        if (*split) *split = '\0';
        split++;
    }

    return argc;

}

static void getInput(char input[]) {

    printf(">>> ");
    if (!fgets(input, sizeof(input), stdin)) {fprintf("Error reading command, try again...\n", stderr); return getInput(input);}

    // strip
    input[strcspn(input, "\r\n")] = '\0';
}

static inline CommandAbstract* getCommand(char input[], int nCmds) {

    for (int i = 0; i < nCmds; i++) {
        if (strncmp(input, cmds[i].name, MAX_CMD_NAME)) return &cmds[i];
    }

    return NULL;

}

// terminal functions
void checkTermination(Board* b);
void handleStalemate(Board* b) {
    printf("Stalemate: 0.5 -- 0.5.\n");
    exit(0);  // new game maybe another time (make this function return a bool i guess)
}
void handleCheckmate(Board* b);

static inline bool isCharInt(const char c) {
    return '0' <= c && c <= '9';
}


Piece getPieceFromChar(const char c) {

    switch (c) {
        case 'P': return WP;
        case 'N': return WN;
        case 'B': return WB;
        case 'R': return WR;
        case 'K': return WK;
        case 'Q': return WQ;

        case 'p': return BP;
        case 'n': return BN;
        case 'b': return BB;
        case 'r': return BR;
        case 'k': return BK;
        case 'q': return BQ;
        default: return EMPTY;  // not valid piece
    }
}

static inline bool isValidPiece(const char c) {
    // basically one of these will kill c if its valid
    return !(
        (c ^ 'r') & (c ^ 'n') & (c ^ 'b') & (c ^ 'q') & (c ^ 'k') & (c ^ 'p') &
        (c ^ 'R') & (c ^ 'N') & (c ^ 'B') & (c ^ 'Q') & (c ^ 'K') & (c ^ 'P')
    );
}

static inline unsigned int getSquareIndex(const int i, const int j) {

    // i gives the chunk, j gives the index.
    // eg, 00001000 00000000 ...
    // is the 0th i and 3rd j, and the square is 59. so we need the conversion 64 - i*8 + j - 8 = 56 - i * 8 + j
    return 56 - i * 8 + j;

}

// return the uint64_t with a 1 in the position of rank 8 - i and file j
static inline uint64_t getPieceBitboardSetter(const int i, const int j) {

    uint64_t k = 1;

    return k << getSquareIndex(i, j);
}

static inline uint8_t getValidCastlingFen(const char c) {
    switch (c) {
        case 'K': return whiteShortCastleMask;
        case 'Q': return whiteLongCastleMask;
        case 'k': return blackShortCastleMask;
        case 'q': return blackLongCastleMask;
        default:  return 0x0;
    }
}

static inline uint8_t convertSquareNotationToEP(const char file, const char rank) {

    if (rank != '3' || rank != '6' || rank < '1' || rank > '8') return 0;

    uint8_t k = 16;
    k += (file - '0' - 1);
    if (rank == '6') k += 24;
    return k;

}

static inline unsigned int convertFullmoveStringToPly(const char* fullmoves, uint64_t blackToMove) {


    return ((((unsigned int) (fullmoves - '0')) - 1) << 1) + blackToMove;

}

bool loadFromFen(Board* b, const char* fen) {

    for (int i = 0; i < 8; i++) { 
        for (int j = 0; j < 8; j++) {

            fen++;
            char c = *fen;
            
            if (isCharInt(c)) {
                fen += (c - 0x30) - 1;  // add int to fen ptr
                continue;
            }

            if (!isValidPiece(c)) return false;
            Piece piece = getPieceFromChar(c);
            if (piece == EMPTY) return false;
            uint8_t pieceIndex = getBitboardIndex(piece);
            unsigned int squareIndex = getSquareIndex(i, j);

            b->pieces[squareIndex] = piece;
            b->bitboards[pieceIndex] |= getPieceBitboardSetter(i, j);
        }

    }

    if (*fen != ' ') return false;
    fen++;

    uint8_t colourToMove = 0;
    if (*fen == 'b') colourToMove = 1;
    else if (*fen != 'w') return false;
    setColourToMove(&(b->gameState), colourToMove);
    fen += 2;
    if (*fen != '-') {
        
        while (*fen != ' ') {
            uint8_t castleState = getValidCastlingFen(*fen);
            if (!castleState) return false;

            orCastlingRights(&(b->gameState), castleState);
            fen++;
        }
    }
    else {
        fen++;
    }

    if (*fen != ' ') return false;
    fen++;
    if (*fen != '-') {

        char file = *fen;
        char rank = *(fen + 1);

        uint8_t epSquare = convertSquareNotationToEP(file, rank);
        if (!epSquare) return false;
        setEnPassantSquare(&(b->gameState), epSquare);
        fen += 2;
    }
    else fen++;
    if (*fen != ' ') return false;
    fen++; 
    
    char digit1 = *fen;
    char digit2 = *(fen + 1);
    fen += 2;

    uint8_t halfmove = 0;

    if (!isCharInt(digit1)) return false;

    if (digit1 == '0') {
        if (digit2 != ' ') return false;
    }
    else {

        if (digit2 == ' ') {
            halfmove = digit1 - 0x30;
        }

        else {
            if (!isCharInt(digit2)) return false;
            halfmove = (digit1 - 0x30) * 10 + (digit2 - 0x30);
            if (*fen != ' ') return false;
            fen++;
        }
    }
    setHalfmoveClock(&(b->gameState), halfmove);

    char* fullmoves = fen;  // from here until \0
    unsigned int fenPly = convertFullmoveStringToPly(fullmoves, colourToMove);

    if (!fenPly) return false;
    fenPly += colourToMove;
    b->ply = fenPly;

    b->zobrist = generateZobristHash(b);

    return true;
}

// convert position to fen (lets call this with a flag in the fen cmd)
char* convertToFen(Board* b);

void cliMainLoop(Game* g) {

    game = g;  // set our global game ptr to the one passed in

    int nCmds = (int)(sizeof(cmds) / sizeof(CommandAbstract));

    
    char input[MAX_STDIN];
    int argc = 0;
    char* argv[MAX_ARG];

    bool checkTermination = false;

    while (true) {

        if (checkTermination) (game->board);

        getInput(input);

        argc = tokenize(input, argv);
        if (!argc) continue;

        CommandAbstract* cmd = getCommand(input, nCmds);

        if (!cmd) {fprintf("Command not found: \"%s\"\n", input, stderr); continue;}  // since we tokenized input, this only prints the name

        // perform cmd
        if (cmd->cmd(argc, argv) < 0) fprintf("Something went wrong...\n", stderr);

        // Player player = (isBlackToMove(board->gameState)) ? black : white;

        // Move move = getMove(board, player);

        // performCommand(board);  // i am not sure how to implement this still

        // performMove(board, move);

    }


}



Move getmove(Board* b, Player player);
bool isValidMove(Board* b, Move move); 
void handleIllegal();

void performUndo(Board* b, Undo undo);
void performMove(Board* b, Move move);

// one of these is chosen for the performCommand pointer 
///TODO: not anymore, we're doign something different. could be a bool
void __DEBUG_performCommand(Board* b);
void noDebugGetMove(Board* b);

// commands:
void handleUndo(Board* b, Undo undo);
void handleMakeMove(Board* b, Move move);
void handlePerft(Board* b);
void handleChildren(Board* b);
void handleQuit() {exit(0);}
void handleResign(Board* b);