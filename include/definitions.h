#ifndef DEFINITIONS_HEADER
#define DEFINITIONS_HEADER

// this is a root header. it must not contain any other custom header.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <ctype.h>

// MACROS AND DEFS
#define MAX_PLY 0x7FFF
#define MAX_DEPTH 256

// promo needs 3 bits instead of 2, since the third will represent whether or not a promo actually happened
//                                        v----- double pawn push (need more bits? no we have what we need)
// Move specification:      R R R R | R K D C | C C C E | E E E E | E P P P | T T T T | T T S S | S S S S
//                                      ^
//                         reserved castled capturedPiece enpassant promo     target        source
typedef uint32_t Move;
typedef uint32_t Gamestate;
typedef uint64_t Undo64;

static const uint64_t UNDO_moveMask = 0x00000000FFFFFFFF;  // the last-played move
static const uint64_t UNDO_gsMask   = 0xFFFFFFFF00000000;  // the gamestate before the move is played

static const Move NULL_MOVE = 0;
static const Gamestate NULL_GAMESTATE = 0;
static const Undo64 NULL_UNDO = 0;

// who knows if ill use all of these, but some of them are useful
static const uint64_t fileH = 0x8080808080808080;
static const uint64_t fileA = 0x0101010101010101;
static const uint64_t rank1 = 0x00000000000000FF;
static const uint64_t rank2 = 0x000000000000FF00;
static const uint64_t rank3 = 0x0000000000FF0000;
static const uint64_t rank4 = 0x00000000FF000000;
static const uint64_t rank5 = 0x000000FF00000000;
static const uint64_t rank6 = 0x0000FF0000000000;
static const uint64_t rank7 = 0x00FF000000000000;
static const uint64_t rank8 = 0xFF00000000000000;

// squares that need to be empty for castling
static const uint64_t bcd1  = 0x000000000000000E;
static const uint64_t fg1   = 0x0000000000000060;  // for white

static const uint64_t bcd8  = 0x0E00000000000000;  // for black
static const uint64_t fg8   = 0x6000000000000000;

// Move masks
static const uint32_t castleMask        = 0x04000000;
static const uint32_t doublePushMask    = 0x02000000;
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

// 01xx, set the on bit
static const uint8_t promoKnight = 0x4;
static const uint8_t promoBishop = 0x5;
static const uint8_t promoRook   = 0x6;
static const uint8_t promoQueen  = 0x7;

// stuff here stores data to make undoing trivial
// REPLACE WITH UINT64_T FOR SPEED
typedef struct {
    uint64_t zobrist;
    uint8_t captured;
    uint8_t enpassant;
    uint8_t castling_rights;
    uint8_t halftime;  // 50 move thing

} Undo;

// these update in parallel. undo holds metadata for easy undo of boards
typedef struct {

    uint64_t hashHistory[MAX_PLY];  // needed? undo hash is just xor again, g^2 = 0
    Move moveHistory[MAX_PLY];
    Undo undoHistory[MAX_PLY];
    Gamestate gamestateHistory[MAX_PLY];
    uint8_t ply;
    
} History;

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

typedef enum {
    EMPTY_TYPE,
    PAWN,
    KNIGHT,
    BISHOP,
    ROOK,
    QUEEN,
    KING
} PieceType;


// hold bit boards for the game. will also store game meta data like castling rights, enpassant, etc.
typedef struct {

    Piece    pieces[64];    // board-centrix view of pieces because this is cheap and convenient
    uint64_t bitboards[12];   // stores all bitboards, indexed by iCT for Colour, Type = CT
    uint64_t boardUnions[3];  // eg all white, all black, all pieces - "blockers"
    uint64_t zobrist;  // updated incrementally each move or undo via xor
    Undo64   undoStack[MAX_PLY];  // indexed by ply

    // gameState format:
    // _ _ _ _ _ _ _ _ | _ _ _ _ _ _ T H | H H H H H H E E | E E E E C C C C 
    // colour to move | halfmove clock (50 move counter) | ep square (0 for none because 0 = a1 is never ep) | castling rights
    // The halfmove clock specifies a decimal number of half moves with respect to the 50 move draw rule. 
    // It is reset to zero after a capture or a pawn move and incremented otherwise.

    // castling rights: _ _ _ _ | black short, black long, white short, white long

    Gamestate gamestate;
    unsigned int ply;  // 0 initially. >> 1 to get full move clock. 
} Board;

static inline void updateBoardUnions(Board* b) {

    // update the union bitboards after loading from fen, for testing purposes

    uint64_t* bitboards = b->bitboards;
    uint64_t* boardUnions = b->boardUnions;
    uint64_t whitePieces = 0;
    uint64_t blackPieces = 0;

    for (int i = 0; i < 6; i++) {
        whitePieces |= bitboards[i];
        blackPieces |= bitboards[i + 6];
    }

    boardUnions[0] = whitePieces;
    boardUnions[1] = blackPieces;
    boardUnions[2] = whitePieces | blackPieces;

}

// index into bitboards
typedef enum {
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
} PieceIndex;

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


}  Square;  // a1 = 0, h8 = 63. for rank, file: square(rank, file) = 64 - (8 - (rank - 1)) * 8 - (8 - (file - 1)). see "getSquareIndex" function in fen.c


// "why" because its O(1) lookup
static const uint64_t squareBitboards[64] = {

    0b0000000000000000000000000000000000000000000000000000000000000001,
    0b0000000000000000000000000000000000000000000000000000000000000010,
    0b0000000000000000000000000000000000000000000000000000000000000100,
    0b0000000000000000000000000000000000000000000000000000000000001000,
    0b0000000000000000000000000000000000000000000000000000000000010000,
    0b0000000000000000000000000000000000000000000000000000000000100000,
    0b0000000000000000000000000000000000000000000000000000000001000000,
    0b0000000000000000000000000000000000000000000000000000000010000000,
    0b0000000000000000000000000000000000000000000000000000000100000000,
    0b0000000000000000000000000000000000000000000000000000001000000000,
    0b0000000000000000000000000000000000000000000000000000010000000000,
    0b0000000000000000000000000000000000000000000000000000100000000000,
    0b0000000000000000000000000000000000000000000000000001000000000000,
    0b0000000000000000000000000000000000000000000000000010000000000000,
    0b0000000000000000000000000000000000000000000000000100000000000000,
    0b0000000000000000000000000000000000000000000000001000000000000000,
    0b0000000000000000000000000000000000000000000000010000000000000000,
    0b0000000000000000000000000000000000000000000000100000000000000000,
    0b0000000000000000000000000000000000000000000001000000000000000000,
    0b0000000000000000000000000000000000000000000010000000000000000000,
    0b0000000000000000000000000000000000000000000100000000000000000000,
    0b0000000000000000000000000000000000000000001000000000000000000000,
    0b0000000000000000000000000000000000000000010000000000000000000000,
    0b0000000000000000000000000000000000000000100000000000000000000000,
    0b0000000000000000000000000000000000000001000000000000000000000000,
    0b0000000000000000000000000000000000000010000000000000000000000000,
    0b0000000000000000000000000000000000000100000000000000000000000000,
    0b0000000000000000000000000000000000001000000000000000000000000000,
    0b0000000000000000000000000000000000010000000000000000000000000000,
    0b0000000000000000000000000000000000100000000000000000000000000000,
    0b0000000000000000000000000000000001000000000000000000000000000000,
    0b0000000000000000000000000000000010000000000000000000000000000000,
    0b0000000000000000000000000000000100000000000000000000000000000000,
    0b0000000000000000000000000000001000000000000000000000000000000000,
    0b0000000000000000000000000000010000000000000000000000000000000000,
    0b0000000000000000000000000000100000000000000000000000000000000000,
    0b0000000000000000000000000001000000000000000000000000000000000000,
    0b0000000000000000000000000010000000000000000000000000000000000000,
    0b0000000000000000000000000100000000000000000000000000000000000000,
    0b0000000000000000000000001000000000000000000000000000000000000000,
    0b0000000000000000000000010000000000000000000000000000000000000000,
    0b0000000000000000000000100000000000000000000000000000000000000000,
    0b0000000000000000000001000000000000000000000000000000000000000000,
    0b0000000000000000000010000000000000000000000000000000000000000000,
    0b0000000000000000000100000000000000000000000000000000000000000000,
    0b0000000000000000001000000000000000000000000000000000000000000000,
    0b0000000000000000010000000000000000000000000000000000000000000000,
    0b0000000000000000100000000000000000000000000000000000000000000000,
    0b0000000000000001000000000000000000000000000000000000000000000000,
    0b0000000000000010000000000000000000000000000000000000000000000000,
    0b0000000000000100000000000000000000000000000000000000000000000000,
    0b0000000000001000000000000000000000000000000000000000000000000000,
    0b0000000000010000000000000000000000000000000000000000000000000000,
    0b0000000000100000000000000000000000000000000000000000000000000000,
    0b0000000001000000000000000000000000000000000000000000000000000000,
    0b0000000010000000000000000000000000000000000000000000000000000000,
    0b0000000100000000000000000000000000000000000000000000000000000000,
    0b0000001000000000000000000000000000000000000000000000000000000000,
    0b0000010000000000000000000000000000000000000000000000000000000000,
    0b0000100000000000000000000000000000000000000000000000000000000000,
    0b0001000000000000000000000000000000000000000000000000000000000000,
    0b0010000000000000000000000000000000000000000000000000000000000000,
    0b0100000000000000000000000000000000000000000000000000000000000000,
    0b1000000000000000000000000000000000000000000000000000000000000000
};


static const int victim_value[6] = {100, 320, 330, 500, 900, 2000000000}; // 0, P, N, B, R, Q, K

// debug
extern const char* testFens[];
extern int sizeOfTestFens;

#endif