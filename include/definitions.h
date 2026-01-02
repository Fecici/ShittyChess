#ifndef DEFINITIONS_HEADER
#define DEFINITIONS_HEADER

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

// MACROS AND DEFS
#define MAX_PLY 0x7FFF
#define MAX_DEPTH 256

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

static const uint32_t GS_castlingRightsMask  = 0x0000000F;
static const uint32_t GS_enpassantSquareMask = 0x000003F0;
static const uint32_t GS_halfmoveClockMask   = 0x0001FC00;
static const uint32_t GS_colourtoMoveMask    = 0x00020000;

static const uint8_t whiteLongCastleMask  = 0x1;
static const uint8_t whiteShortCastleMask = 0x2;
static const uint8_t blackLongCastleMask  = 0x4;
static const uint8_t blackShortCastleMask = 0x8;

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

    Piece    pieces[64];    // board-centrix view of pieces because this is cheap and convenient
    uint64_t bitboards[12];   // stores all bitboards, indexed by iCT for Colour, Type = CT
    uint64_t boardUnions[3];  // eg all white, all black, all pieces - "blockers"
    uint64_t zobrist;  // updated incrementally each move or undo via xor
    Gamestack* gamestack;   // idk if i need this

    // gameState format:
    // _ _ _ _ _ _ _ _ | _ _ _ _ _ _ T H | H H H H H H E E | E E E E C C C C 
    // colour to move | halfmove clock (50 move counter) | ep square (0 for none because 0 = a1 is never ep) | castling rights
    // The halfmove clock specifies a decimal number of half moves with respect to the 50 move draw rule. 
    // It is reset to zero after a capture or a pawn move and incremented otherwise.

    // castling rights: _ _ _ _ | black short, black long, white short, white long

    uint32_t gameState;
    unsigned int ply;  // 0 initially. >> 1 to get full move clock. 
} Board;

// these update in parallel. undo holds metadata for easy undo of boards
typedef struct {

    Move moveStack[MAX_DEPTH];
    Undo undoStack[MAX_DEPTH];
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

typedef enum {WHITE, BLACK} Colour;

typedef enum {

    a1, b1, c1, d1, e1, f1, g1, h1,
    a2, b2, c2, d2, e2, f2, g2, h2,
    a3, b3, c3, d3, e3, f3, g3, h3,
    a4, b4, c4, d4, e4, f4, g4, h4,
    a5, b5, c5, d5, e5, f5, g5, h5,
    a6, b6, c6, d6, e6, f6, g6, h6,
    a7, b7, c7, d7, e7, f7, g7, h7,
    a8, b8, c8, d8, e8, f8, g8, h8


}  Square;  // a1 = 0, h8 = 63. for rank, file: square(rank, file) = 64 - (8 - (rank - 1)) * 8 - (8 - (file - 1)). see "getSqaureIndex" function in cli.c


const int victim_value[7] = {0, 100, 320, 330, 500, 900, 2000000000}; // 0, P, N, B, R, Q, K

#endif