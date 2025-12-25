#ifndef DEFINITIONS_HEADER
#define DEFINITIONS_HEADER

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

// MACROS AND DEFS
#define MAX_PLY 256

// promo needs 3 bits instead of 2, since the third will represent whether or not a promo actually happened
//                                        v----- double pawn push (need more bits? no we have what we need)
// Move specification:      R R R R | R K D C | C C C E | E E E E | E P P P | T T T T | T T S S | S S S S
//                         reserved castled capturedPiece enpassant promo     target        source
typedef uint32_t Move;

// Move masks
static const uint32_t castleMask        = 0x04000000;
static const uint32_t capturedPieceMask = 0x01E00000;
static const uint32_t enPassantMask     = 0x001F8000;
static const uint32_t promoMask         = 0x00007000;
static const uint32_t targetMask        = 0x00000FC0;
static const uint32_t sourceMask        = 0x0000003F;

// stuff here stores data to make undoing trivial
typedef struct {
    uint64_t zobrist;
    uint8_t captured;

    uint8_t enpassant;
    uint8_t castling_rights;
    uint8_t halftime;  // 50 move thing

} Undo;


// hold bit boards for the game. will also store game meta data like castling rights, enpassant, etc.
typedef struct {

    uint64_t bitboards[12];
    uint64_t boardUnions[3];  // eg all white, all black, all pieces - "blockers"
    Gamestack* gamestack;   // idk if i need this
    uint64_t gameState;

} Board;

// these update in parallel. undo holds metadata for easy undo of boards
typedef struct {

    Move moveStack[MAX_PLY];
    Undo undoStack[MAX_PLY];
    uint8_t ply;  // init to 0

} Gamestack;

enum Piece {
    EMPTY,      // no piece
    WP = 1,     // 0001
    WN,
    WB,
    WR,
    WQ,
    WK,
    BP = 9,     // 1001
    BN,
    BB,
    BR,
    BQ,
    BK,
};

// index into bitboards
enum PieceIndex {
    iWP,     // 0
    iWN,
    iWB,
    iWR,
    iWQ,
    iWK,
    iBP,     // 6
    iBN,
    iBB,
    iBR,
    iBQ,
    iBK
};

const int victim_value[7] = {0, 100, 320, 330, 500, 900, 2000000000}; // 0, P, N, B, R, Q, K

#endif