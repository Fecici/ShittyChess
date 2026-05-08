#ifndef FEN_HEADER
#define FEN_HEADER

#include "definitions.h"
#include "bitUtils.h"
#include "zobrist.h"

bool loadFromFen(Board* b, char* fen);
bool validFen(const char* fen);


static inline unsigned int getSquareIndex(const int i, const int j) {

    // i gives the chunk, j gives the index.
    // eg, 00001000 00000000 ...
    // is the 0th i and 3rd j, and the square is 59. so we need the conversion 64 - i*8 + j - 8 = 56 - i * 8 + j
    return (unsigned int) (56 - i * 8 + j);

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

    if ((rank != '3' && rank != '6') || (file < 'a' || file > 'h')) return 0;

    uint8_t k = 16;
    k += (uint8_t) (file - 'a');
    if (rank == '6') k += 24;
    return k;

}

static inline unsigned int convertFullmoveStringToPly(const char* fullmoves, uint64_t blackToMove) {


    return ((((unsigned int) strtol(fullmoves, NULL, 0)) - 1) << 1) + ((unsigned int) blackToMove);

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
