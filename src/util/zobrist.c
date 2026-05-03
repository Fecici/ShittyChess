#include "zobrist.h"


/*
clanker did this. i want to do it differently, so this serves as a reference
check spec in header for more insight into this. 

https://www.chessprogramming.org/Zobrist_Hashing
*/

uint64_t zobristTable[12][64];  // [piece][square]
uint64_t zobristCastle[16];  // 16 possible castling rights combinations
uint64_t zobristEnPassant[8];  // 8 possible en passant files
uint64_t zobristBlackToMove;  // for black to move


void initZobrist() {
    srand(0);  // fixed seed for reproducibility

    for (int piece = 0; piece < 12; piece++) {
        for (int square = 0; square < 64; square++) {
            zobristTable[piece][square] = ((uint64_t) rand() << 32) | rand();
        }
    }

    for (int i = 0; i < 16; i++) {
        zobristCastle[i] = ((uint64_t) rand() << 32) | rand();
    }

    for (int i = 0; i < 8; i++) {
        zobristEnPassant[i] = ((uint64_t) rand() << 32) | rand();
    }

    zobristBlackToMove = ((uint64_t) rand() << 32) | rand();
}

uint64_t generateZobristHash(Board* b) {
    
    uint64_t hash = 0;

    for (int i = 0; i < 12; i++) {
        uint64_t bitboard = b->bitboards[i];
        while (bitboard) {
            uint64_t k = bitboard & -bitboard;  // get least significant bit
            uint8_t squareIndex = __builtin_ctzll(k);  // get index of least significant bit
            hash ^= zobristTable[i][squareIndex];  // xor with corresponding zobrist value
            bitboard &= bitboard - 1;  // clear least significant bit
        }
    }

    // castling rights
    uint8_t castlingRights = getCastlingRights(b->gameState);
    hash ^= zobristCastle[castlingRights];

    // en passant
    uint8_t epSquare = getEnPassantSquare(b->gameState);
    if (epSquare) {
        hash ^= zobristEnPassant[epSquare % 8];  // only the file matters for en passant
    }

    // colour to move
    if (isBlackToMove(b->gameState)) {
        hash ^= zobristBlackToMove;
    }

    return hash;
}
