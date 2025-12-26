#include "CLI.h"

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


void initPlayer(Player* p, PlayerType type, uint8_t colour, Engine* engine);
Game initGame();  // init all, setup history, ui, etc.
void getCommand();
void handleCommand();
void setUI(Game* game, UI ui);
void checkTermination(Board* b);
void handleStalemate(Board* b);
void handleCheckmate(Board* b);

bool isCharInt(const char c);
Piece getPieceFromChar(const char c);
static inline bool isValidPiece(const char c) {
    // basically one of these will kill c if its valid
    return !(
        (c ^ 'r') & (c ^ 'n') & (c ^ 'b') & (c ^ 'q') & (c ^ 'k') & (c ^ 'p') &
        (c ^ 'R') & (c ^ 'N') & (c ^ 'B') & (c ^ 'Q') & (c ^ 'K') & (c ^ 'P')
    );
}

static inline unsigned int getSquareIndex(const int i, const int j) {

}

static inline uint64_t getPieceBitboardSetter(const int i, const int j) {

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

static inline bool digitIsValid(const char c) {

}

static inline unsigned int convertFullmoveStringToPly(const char* fullmoves) {

}

bool loadFromFen(Board* b, const char* fen) {

    int k;  // index to string

    for (int i = 0; i < 8; i++) { 
        for (int j = 0; j < 9; j++) {  // do +9 because of the '/'
            /// TODO: this is incorrect because of empty spaces
            k = i * 9 + j;
            if (j == 8) {
                if (fen[k] != '/') return false;
                continue;
            }
            char c = fen[k];
            if (isCharInt(c)) {
                j += (c - 0x30) - 1;  // add int to j counter
                if (j >= 8) return false;  // invalid
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

    k++;
    if (fen[k] != ' ') return false;
    k++;

    uint8_t colourToMove = 0;
    if (fen[k] == 'b') colourToMove = 1;
    else if (fen[k] != 'w') return false;
    setColourToMove(&(b->gameState), colourToMove);

    k++;
    if (fen[k] != '-') {
        
        for (int i = k; i < k + 4; i++) {
            if (i >= k + 1 && fen[i] == ' ') break;
            uint8_t castleState = getValidCastlingFen(fen[i]);
            if (!castleState) return false;

            orCastlingRights(&(b->gameState), castleState);
            k = i;  // at i = k + 4, k has the value k = k_old + 4
        }
    }
    else {
        k++;
    }

    if (fen[k] != ' ') return false;
    k++;
    if (fen[k] != '-') {

        char file = fen[k];
        char rank = fen[k+1];

        uint8_t epSquare = convertSquareNotationToEP(file, rank);
        if (!epSquare) return false;
        setEnPassantSquare(&(b->gameState), epSquare);
        k += 2;
    }
    else k++;
    if (fen[k] != ' ') return false;
    k++; 
    
    char digit1 = fen[k];
    char digit2 = fen[k+1];
    k += 2;

    uint8_t halfmove = 0;

    if (!digitIsValid(digit1)) return false;

    if (digit1 == '0') {
        if (digit2 != ' ') return false;
    }
    else {

        if (digit2 == ' ') {
            halfmove = digit1 - 0x30;
        }

        else {
            if (!digitIsValid(digit2)) return false;
            halfmove = (digit1 - 0x30) * 10 + (digit2 - 0x30);
            if (fen[k] != ' ') return false;
            k++;
        }
    }
    setHalfmoveClock(&(b->gameState), halfmove);

    char* fullmoves = fen + k;  // from here until \0
    unsigned int fenPly = convertFullmoveStringToPly(fullmoves);

    if (!fenPly) return false;
    fenPly += colourToMove;
    b->ply = fenPly;

    b->zobrist = generateZobristHash(b);

    return true;
}


char* convertToFen(Board* b);

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