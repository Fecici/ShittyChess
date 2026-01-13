#ifndef BIT_UTIL_HEADER
#define BIT_UTIL_HEADER


#include "definitions.h"


// function signatures
static inline uint8_t getBitboardIndex(uint8_t piece);
static inline uint8_t getPieceType(uint8_t piece);
static inline uint8_t getPiecesColour(uint8_t piece);
static inline uint8_t getSrc(Move move);
static inline uint8_t getDst(Move move);
static inline uint8_t getEnPassant(Move move);
static inline bool isCastled(Move move);
static inline uint8_t getPromotion(Move move);
static inline uint8_t getCapturedPieceCode(Move move);
static inline uint8_t getCapturedType(Move move);
static inline uint8_t getCapturedColour(Move move);
 

static inline uint8_t getCastlingRights(uint32_t gamestate);
static inline uint8_t isBlackToMove(uint32_t gamestate);
static inline uint8_t getHalfmoveClock(uint32_t gamestate);
static inline uint8_t getEnPassantSquare(uint32_t gamestate);

static inline void setCastlingRights(uint32_t* gamestate, uint8_t state);
static inline void setColourToMove(uint32_t* gamestate, uint8_t state);
static inline void setHalfmoveClock(uint32_t* gamestate, uint8_t state);
static inline void setEnPassantSquare(uint32_t* gamestate, uint8_t state);
static inline void incrHalfmoveClock(uint32_t* gamestate);

// castling rights: _ _ _ _ | black short, black long, white short, white long
static inline void orCastlingRights(uint32_t* gamestate, uint8_t field);
static inline bool canWhiteCastleLong(uint32_t gamestate);
static inline bool canWhiteCastleShort(uint32_t gamestate);
static inline bool canBlackCastleLong(uint32_t gamestate);
static inline bool canBlackCastleShort(uint32_t gamestate);
static inline int getMoveCount(int ply);

#endif