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
extern uint64_t (*const pieceGenerator[12]) (Board*, Square);

// for precomp
void precomputeKnights();
void precomputeKingMoves();

// debug
uint64_t debug_getKingMove(Square src);
uint64_t debug_getKnightMove(Square src);

#endif