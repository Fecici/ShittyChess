#ifndef BIT_UTIL_HEADER
#define BIT_UTIL_HEADER


#include "definitions.h"


// function signatures
uint8_t getBitboardIndex(uint8_t piece);
uint8_t getPieceType(uint8_t piece);
uint8_t getPiecesColour(uint8_t piece);
uint8_t getSrc(Move move);
uint8_t getDst(Move move);
uint8_t getEnPassant(Move move);
bool isCastled(Move move);
uint8_t getPromotion(Move move);
uint8_t getCapturedPieceCode(Move move);
uint8_t getCapturedType(Move move);
uint8_t getCapturedColour(Move move);
 

uint8_t getCastlingRights(uint32_t gamestate);
uint8_t isBlackToMove(uint32_t gamestate);
uint8_t getHalfmoveClock(uint32_t gamestate);
uint8_t getEnPassantSquare(uint32_t gamestate);

void setCastlingRights(uint32_t* gamestate, uint8_t state);
void setColourToMove(uint32_t* gamestate, uint8_t state);
void setHalfmoveClock(uint32_t* gamestate, uint8_t state);
void setEnPassantSquare(uint32_t* gamestate, uint8_t state);
void incrHalfmoveClock(uint32_t* gamestate);

// castling rights: _ _ _ _ | black short, black long, white short, white long
void orCastlingRights(uint32_t* gamestate, uint8_t field);
bool canWhiteCastleLong(uint32_t gamestate);
bool canWhiteCastleShort(uint32_t gamestate);
bool canBlackCastleLong(uint32_t gamestate);
bool canBlackCastleShort(uint32_t gamestate);
unsigned int getMoveCount(unsigned int ply);

#endif