#ifndef FEN_HEADER
#define FEN_HEADER

#include "definitions.h"
#include "bitUtils.h"
#include "zobrist.h"

bool loadFromFen(Board* b, char* fen);
bool validFen(const char* fen);

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
