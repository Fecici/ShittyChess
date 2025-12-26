#include "../../include/bitUtil.h"

static inline uint8_t getBitboardIndex(uint8_t piece) {

    return (getPieceType(piece) - 1 + 6 * getPieceColour(piece));

};

static inline uint8_t getPieceType(uint8_t piece) {
    return piece & 7;
}

static inline uint8_t getPieceColour(uint8_t piece) {
    return piece >> 3;
};

static inline uint8_t getSrc(Move move) {
    return move & sourceMask;
};

static inline uint8_t getDst(Move move) {
    return (move & targetMask) >> 6;
};

static inline uint8_t getEnPassant(Move move) {
    return (move & enPassantMask) >> 15;
};

static inline bool isCastled(Move move) {

    return (move & castleMask);
}

static inline uint8_t getPromotion(Move move) {
    
    return (move & promoMask) >> 12;
}

static inline uint8_t getCapturedPieceCode(Move move) {
    return (move & capturedPieceMask) >> 21;
}

static inline uint8_t getCapturedType(Move move) {

    return getCapturedPieceCode(move) & 7;
}

static inline uint8_t getCapturedColour(Move move) {
    return getCapturedPieceCode(move) >> 3;
}



static inline uint8_t getCastlingRights(uint32_t gamestate) {


    return gamestate & GS_castlingRightsMask;
}

static inline uint8_t isBlackToMove(uint32_t gamestate) {

    return (gamestate & GS_colourtoMoveMask);  // since this would return 0 for white's turn, we can just keep this since itll act as a bool anyways

}

static inline uint8_t getHalfmoveClock(uint32_t gamestate) {

    return (gamestate & GS_halfmoveClockMask) >> 10;

}

static inline uint8_t getEnPassantSquare(uint32_t gamestate) {

    return (gamestate & GS_enpassantSquareMask) >> 4;
}

static inline void setCastlingRights(uint32_t* gamestate, uint8_t state) {

    *gamestate = (*gamestate & ~GS_castlingRightsMask) | (state & 0xFU);

}

static inline void setColourToMove(uint32_t* gamestate, uint8_t state) {

    *gamestate = (*gamestate & ~GS_colourtoMoveMask) | ((state & 0x1U) << 17);

}

static inline void setHalfmoveClock(uint32_t* gamestate, uint8_t state) {

    *gamestate = (*gamestate & ~GS_halfmoveClockMask) | ((state & 0x7FU) << 10);

}

static inline void setEnPassantSquare(uint32_t* gamestate, uint8_t state) {

    *gamestate = (*gamestate & ~GS_enpassantSquareMask) | ((state & 0x3FU) << 10);

}

static inline void incrHalfmoveClock(uint32_t* gamestate) {
    setHalfmoveClock(gamestate, getHalfmoveClock(gamestate) + 1);
}

static inline void orCastlingRights(uint32_t* gamestate, uint8_t field) {

}

static inline void canWhiteCastleLong(uint32_t gamestate) {

}

static inline void canWhiteCastleShort(uint32_t gamestate) {

}

static inline void canBlackCastleLong(uint32_t gamestate) {

}

static inline void canBlackCastleShort(uint32_t gamestate) {

}