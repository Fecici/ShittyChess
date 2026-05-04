#ifndef BIT_UTIL_HEADER
#define BIT_UTIL_HEADER


#include "definitions.h"

static inline uint8_t getPieceType(uint8_t piece) {
    return piece & 7;
}

static inline uint8_t getPiecesColour(uint8_t piece) {
    return piece >> 3;
};

static inline uint8_t getSrc(Move move) {
    return (uint8_t) (move & sourceMask);
};

static inline uint8_t getDst(Move move) {
    return (uint8_t) ((move & targetMask) >> 6);
};

static inline uint8_t getEnPassant(Move move) {
    return (uint8_t) ((move & enPassantMask) >> 15);
};

static inline bool isCastled(Move move) {

    return (move & castleMask);
}

static inline uint8_t getPromotion(Move move) {
    
    return (uint8_t) ((move & promoMask) >> 12);
}

static inline uint8_t getCapturedPieceCode(Move move) {
    return (uint8_t) ((move & capturedPieceMask) >> 21);
}

static inline uint8_t getCapturedType(Move move) {

    return (uint8_t) (getCapturedPieceCode(move) & 7);
}

static inline uint8_t getCapturedColour(Move move) {
    return (uint8_t) (getCapturedPieceCode(move) >> 3);
}

static inline void setSrc(Move* move, uint8_t src) {
    *move = (*move & ~sourceMask) | (src & sourceMask);
}

static inline void setDst(Move* move, uint8_t dst) {
    *move = (*move & ~targetMask) | ((dst << 6) & targetMask);
}

static inline void setPromotion(Move* move, uint8_t promo) {
    *move = (*move & ~promoMask) | ((promo << 12) & promoMask);
}

static inline void setEnPassant(Move* move, uint8_t ep) {
    *move = (*move & ~enPassantMask) | ((ep << 15) & enPassantMask);
}

static inline void setCapturedPieceCode(Move* move, uint8_t captured) {
    *move = (*move & ~capturedPieceMask) | ((captured << 21) & capturedPieceMask);
}

static inline void setCapturedColour(Move* move, uint8_t colour) {
    *move |= (colour & 1) << 24;
}

static inline void setCapturedType(Move* move, uint8_t type) {
    uint8_t captured = getCapturedPieceCode(*move);
    captured = (captured & 0x18) | (type & 7);
    setCapturedPieceCode(move, captured);
}

static inline uint8_t getCastlingRights(uint32_t gamestate) {
    return (uint8_t) (gamestate & GS_castlingRightsMask);
}


static inline uint8_t getHalfmoveClock(uint32_t gamestate) {
    
    return (uint8_t) ((gamestate & GS_halfmoveClockMask) >> 10);
    
}

static inline uint8_t getEnPassantSquare(uint32_t gamestate) {
    
    return (uint8_t) ((gamestate & GS_enpassantSquareMask) >> 4);
}

static inline void setCastlingRights(uint32_t* gamestate, uint8_t state) {
    
    *gamestate = (*gamestate & ~GS_castlingRightsMask) | (state & 0xFU);
    
}

static inline void setColourToMove(uint32_t* gamestate, uint8_t state) {
    
    *gamestate = (*gamestate & ~GS_colourtoMoveMask) | ((state & 0x1U) << 17);
    
}

static inline uint8_t getColourToMove(uint32_t gamestate) {
    
    return (uint8_t) ((gamestate & GS_colourtoMoveMask) >> 17);
}

static inline bool isBlackToMove(uint32_t gamestate) {

    return getColourToMove(gamestate) == 1;  // since this would return 0 for white's turn, we can just keep this since itll act as a bool anyways

}

static inline void setHalfmoveClock(uint32_t* gamestate, uint8_t state) {

    *gamestate = (*gamestate & ~GS_halfmoveClockMask) | ((state & 0x7FU) << 10);

}

static inline void setEnPassantSquare(uint32_t* gamestate, uint8_t state) {

    *gamestate = (*gamestate & ~GS_enpassantSquareMask) | ((state & 0x3FU) << 4);

}

static inline void incrHalfmoveClock(uint32_t* gamestate) {
    setHalfmoveClock(gamestate, getHalfmoveClock(*gamestate) + 1);
}

static inline void orCastlingRights(uint32_t* gamestate, uint8_t field) {
    *gamestate |= (field & 0xf);
}

static inline bool canWhiteCastleLong(uint32_t gamestate) {

    return gamestate & whiteLongCastleMask;

}

static inline bool canWhiteCastleShort(uint32_t gamestate) {

    return gamestate & whiteShortCastleMask;

}

static inline bool canBlackCastleLong(uint32_t gamestate) {

    return gamestate & blackLongCastleMask;

}

static inline bool canBlackCastleShort(uint32_t gamestate) {

    return gamestate & blackShortCastleMask;

}

static inline unsigned int getMoveCount(unsigned int ply) {
    return ply / 2 + 1;
}

static inline uint8_t getBitboardIndex(uint8_t piece) {

    return (uint8_t) ((getPieceType(piece) - 1 + 6 * getPiecesColour(piece)));

};

#endif