#ifndef FEN_HEADER
#define FEN_HEADER

#include "definitions.h"
#include "bitUtils.h"
#include "zobrist.h"

bool loadFromFen(Board* b, char* fen);
bool validFen(const char* fen);

static inline Piece getPieceFromChar(const char c) {

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

static inline char getCharFromPiece(Piece piece) {

    switch (piece) {
        case WP: return 'P';
        case WN: return 'N';
        case WB: return 'B';
        case WR: return 'R';
        case WK: return 'K';
        case WQ: return 'Q';

        case BP: return 'p';
        case BN: return 'n';
        case BB: return 'b';
        case BR: return 'r';
        case BK: return 'k';
        case BQ: return 'q';
        default: return ' ';  // not valid piece
    }
}

static inline bool isValidPiece(const char c) {
    // basically one of these will kill c if its valid
    return !(
        (c ^ 'r') & (c ^ 'n') & (c ^ 'b') & (c ^ 'q') & (c ^ 'k') & (c ^ 'p') &
        (c ^ 'R') & (c ^ 'N') & (c ^ 'B') & (c ^ 'Q') & (c ^ 'K') & (c ^ 'P')
    );
}

static inline bool isCharInt(const char c) {
    return '0' <= c && c <= '9';
}

// convert position to fen (lets call this with a flag in the fen cmd)
char* convertToFen(Board* b);

#endif
