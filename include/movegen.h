#ifndef MOVEGEN_HEADER
#define MOVEGEN_HEADER


#include "definitions.h"
#include "bitUtils.h"


// strictly movegen stuff for each piece. 
uint64_t generateWhitePawnMoves(Board* b, Square src);
uint64_t generateWhiteKnightMoves(Board* b, Square src);
uint64_t generateWhiteBishopMoves(Board* b, Square src);
uint64_t generateWhiteRookMoves(Board* b, Square src);
uint64_t generateWhiteQueenMoves(Board* b, Square src);
uint64_t generateWhiteKingMoves(Board* b, Square src);

uint64_t generateBlackPawnMoves(Board* b, Square src);
uint64_t generateBlackKnightMoves(Board* b, Square src);
uint64_t generateBlackBishopMoves(Board* b, Square src);
uint64_t generateBlackRookMoves(Board* b, Square src);
uint64_t generateBlackQueenMoves(Board* b, Square src);
uint64_t generateBlackKingMoves(Board* b, Square src);

// indexed by getBitboardIndex(piece) (eg, iWP)
(uint64_t*) pieceGenerator(Board* b, Square src)[12] = {
    generateWhitePawnMoves,
    generateWhiteKnightMoves,
    generateWhiteBishopMoves,
    generateWhiteRookMoves,
    generateWhiteQueenMoves,
    generateWhiteKingMoves,

    generateBlackPawnMoves,
    generateBlackKnightMoves,
    generateBlackBishopMoves,
    generateBlackRookMoves,
    generateBlackQueenMoves,
    generateBlackKingMoves
};

// for precomp

int directions[8][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}, {1, 0}, {-1, 0}, {0, 1}, {0, -1}};  // bishops use 0-3 rook 4-7.

void precomputeKnights();
void precomputeKingMoves();

#endif