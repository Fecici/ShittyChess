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

    uint64_t bitboards[12];   // stores all bitboards, indexed by iCT for Colour, Type = CT
    uint64_t boardUnions[3];  // eg all white, all black, all pieces - "blockers"
    Piece    pieces[64];    // board-centrix view of pieces because this is cheap and convenient
    Gamestack* gamestack;   // idk if i need this
    // gameState format:
    // _ _ _ _ _ _ _ _ | _ _ _ _ _ _ _ _ | _ _ _ _ _ _ _ _ | _ _ _ _ _ _ _ _ | _ _ _ _ _ _ _ _ | _ _ _ _ _ _ _ _ | _ _ _ _ _ _ _ _ | _ _ _ _ _ _ _ _ |
    //
    uint64_t gameState;
    uint64_t zobrist;  // updated incrementally each move or undo via xor
} Board;

// these update in parallel. undo holds metadata for easy undo of boards
typedef struct {

    Move moveStack[MAX_PLY];
    Undo undoStack[MAX_PLY];
    uint8_t ply;  // init to 0

} Gamestack;

typedef enum {
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
} Piece;

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

typedef enum {

    a1 = 0, b1, c1, d1, e1, f1, g1, h1,
    a2, b2, c2, d2, e2, f2, g2, h2,
    a3, b3, c3, d3, e3, f3, g3, h3,
    a4, b4, c4, d4, e4, f4, g4, h4,
    a5, b5, c5, d5, e5, f5, g5, h5,
    a6, b6, c6, d6, e6, f6, g6, h6,
    a7, b7, c7, d7, e7, f7, g7, h7,
    a8, b8, c8, d8, e8, f8, g8, h8


}  Square;  // a1 = 0, h8 = 63


const int victim_value[7] = {0, 100, 320, 330, 500, 900, 2000000000}; // 0, P, N, B, R, Q, K

#endif