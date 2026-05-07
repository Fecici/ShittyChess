#include "history.h"

Undo* getUndoFromMove(Board* b, Move move) {

    Undo* undo = calloc(1, sizeof(Undo));
    undo->zobrist = b->zobrist;
    undo->captured = getPieceOnSquare(b, getDst(move));
    undo->enpassant = getEnPassant(move);
    undo->castling_rights = getCastlingRights(b->gamestate);
    undo->halftime = (b->gamestate & GS_halfmoveClockMask) >> 10;

    return undo;
}


void performUndo(Board* b, Undo* undo) {
    
    // null check in wrapper; assumed not null
    b->zobrist = undo->zobrist;
    setEnPassantSquare(&b->gamestate, undo->enpassant);
    setCastlingRights(&b->gamestate, undo->castling_rights);
    setHalfmoveClock(&b->gamestate, undo->halftime);

    // using history struct, reconscruct based on move, hash, gamestate, undo
    // bitboards
    Move move = b->history->moveHistory[b->history->ply];
    Square src = getSrc(move);
    Square dst = getDst(move);
    Piece capturedPiece = undo->captured;
    Piece srcPiece = getPieceOnSquare(b, dst);  // since we are undoing, the piece on the destination square is the piece that moved
    uint64_t srcMask = 1ULL << src;
    uint64_t dstMask = 1ULL << dst;

    // toggle bits xor
    b->bitboards[getBitboardIndex(srcPiece)] ^= srcMask;  // add
    b->bitboards[getBitboardIndex(srcPiece)] ^= dstMask;  // remove
    if (capturedPiece != EMPTY) {
        b->bitboards[getBitboardIndex(capturedPiece)] ^= dstMask;  // add captured piece back to destination square
    }

    // handle castling undo
    uint64_t t = 1;
    if (isCastled(move)) {
        if (dst == g1) {  // white kingside
            b->bitboards[getBitboardIndex(WR)] ^= (t << h1) | (t << f1);
            setCastlingRights(&b->gamestate, getCastlingRights(b->gamestate) | whiteLongCastleMask);  // restore white kingside castling right
            b->zobrist ^= getZobristHash(WR, h1) ^ getZobristHash(WR, f1);  // move rook from h1 to f1
            b->zobrist ^= getZobristHash(WK, e1) ^ getZobristHash(WK, g1);  // move king from e1 to g1  
        } else if (dst == c1) {  // white queenside
            b->bitboards[getBitboardIndex(WR)] ^= (t << a1) | (t << d1);
            setCastlingRights(&b->gamestate, getCastlingRights(b->gamestate) | whiteShortCastleMask);  // restore white queenside castling right
            b->zobrist ^= getZobristHash(WR, a1) ^ getZobristHash(WR, d1);  // move rook from a1 to d1
            b->zobrist ^= getZobristHash(WK, e1) ^ getZobristHash(WK, c1);  // move king from e1 to c1

        } else if (dst == g8) {  // black kingside
            b->bitboards[getBitboardIndex(BR)] ^= (t << h8) | (t << f8);
            setCastlingRights(&b->gamestate, getCastlingRights(b->gamestate) | blackLongCastleMask);  // restore black kingside castling right
            b->zobrist ^= getZobristHash(BR, h8) ^ getZobristHash(BR, f8);  // move rook from h8 to f8
            b->zobrist ^= getZobristHash(BK, e8) ^ getZobristHash(BK, g8);  // move king from e8 to g8

        } else if (dst == c8) {  // black queenside
            b->bitboards[getBitboardIndex(BR)] ^= (t << a8) | (t << d8);
            setCastlingRights(&b->gamestate, getCastlingRights(b->gamestate) | blackShortCastleMask);  // restore black queenside castling right
            b->zobrist ^= getZobristHash(BR, a8) ^ getZobristHash(BR, d8);  // move rook from a8 to d8
            b->zobrist ^= getZobristHash(BK, e8) ^ getZobristHash(BK, c8);  // move king from e8 to c8
        }
    }

}

// commands:
int handleUndo(Board* b, Undo* undo) {

    // if undo is null, use stack undo
    if (undo == NULL) {
        if (b->history->ply == 0) {
            fprintf(stderr, "Error: No moves to undo\n");
            return 1;
        }
        else {
            // pop
            undo = &b->history->undoHistory[--b->history->ply];
        }
    }

    performUndo(b, undo);
    return 0;
}


void handlePerft(Board* b);
void handleChildren(Board* b);

void handleResign(Board* b) {

    // placeholder. optionally, show eval o resignation once that is implemented
    bool colourToMove = isBlackToMove(b->gamestate);
    if (colourToMove) {
        fprintf(stderr, "Black resigns. 1-0.\n");
    } else {
        fprintf(stderr, "White resigns. 0-1.\n");
    }
    exit(0);
}

bool pushUndoToStack(Board* b, Undo* undo) {
    if (b->history->ply >= MAX_PLY) {
        fprintf(stderr, "Error: Maximum undo stack size reached\n");
        return false;
    }

    b->history->undoHistory[b->history->ply] = *undo;
    b->history->ply++;
    return true;
}
