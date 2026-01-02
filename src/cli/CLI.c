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



SPEC:

History should store:
Move
Undo
zobrist after move (or before, either is fine if consistent)

make_move should:
update board state, bitboards, piece[64], castling/ep/halfmove, zobrist
return false if move leaves own king in check
undo_last_move pops history and calls unmake_move

*/



/*
PSEUDO:

function run_game(game):
  ui.render(game.board)

  while true:
    // check terminal conditions BEFORE move
    if is_draw(game):  // repetition, 50-move, insufficient material, stalemate
        game.result = DRAW
        break
    if is_checkmate(game.board):
        game.result = WIN(opposite(side_to_move(game.board)))
        break

    // determine whose turn it is
    stm = side_to_move(game.board)
    player = (stm == WHITE) ? game.white : game.black

    // get a move from that player
    if player.type == HUMAN:
        move = get_human_move(game)
        // includes: parse input, validate legal, allow commands (undo, resign, etc.)
    else:
        move = get_engine_move(game, player.engine)

    // if command was issued, handle it
    if move == CMD_UNDO:
        if game.history.size >= 1:
           undo_last_move(game)
           ui.render(game.board)
        continue

    if move == CMD_RESIGN:
        game.result = WIN(opposite(stm))
        break

    if move == CMD_QUIT:
        game.result = ABORTED
        break

    // Apply the move (must be legal)
    ok = make_move(game.board, move, &undo)
    if not ok:
        // if human: tell them invalid and retry
        // if engine: it's a bug (fuck)
        ui.message("Illegal move.")
        continue

    // Record history (for undo, repetition, etc.)
    history_push(game.history, move, undo, game.board.zobrist)

    // render / notify
    ui.render(game.board)

*/

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

    if (b == NULL) {fprintf(stderr, "Failed to allocate memory for board."); exit(1); }

    game->board = b;

    game->white = white;
    game->black = black;

    if (!loadFromFen(b, fen)) {
        fprintf(stderr, "Failed to parse fen string: %s", fen);
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


void getCommand();
void handleCommand();
void checkTermination(Board* b);
void handleStalemate(Board* b);
void handleCheckmate(Board* b);

static inline bool isCharInt(const char c) {
    return '0' <= c && c <= '9';
}


Piece getPieceFromChar(const char c);
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


char* convertToFen(Board* b);

void cliMainLoop(Game* game, void (*performCommand)(Board* b)) {

    Board* board = game->board;
    Player white = game->white;
    Player black = game->black;

    while (true) {

        Player player = (isBlackToMove(board->gameState)) ? black : white;

        Move move = getMove(board, player);

        performCommand(board);  // i am not sure how to implement this still

        performMove(board, move);

    }


}



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