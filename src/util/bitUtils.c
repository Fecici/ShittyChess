#include "bitUtils.h"

uint8_t getBitboardIndex(uint8_t piece) {

    return (uint8_t) ((getPieceType(piece) - 1 + 6 * getPiecesColour(piece)));

};

uint8_t getPieceType(uint8_t piece) {
    return piece & 7;
}

uint8_t getPiecesColour(uint8_t piece) {
    return piece >> 3;
};

uint8_t getSrc(Move move) {
    return (uint8_t) (move & sourceMask);
};

uint8_t getDst(Move move) {
    return (uint8_t) ((move & targetMask) >> 6);
};

uint8_t getEnPassant(Move move) {
    return (uint8_t) ((move & enPassantMask) >> 15);
};

bool isCastled(Move move) {

    return (move & castleMask);
}

uint8_t getPromotion(Move move) {
    
    return (uint8_t) ((move & promoMask) >> 12);
}

uint8_t getCapturedPieceCode(Move move) {
    return (uint8_t) ((move & capturedPieceMask) >> 21);
}

uint8_t getCapturedType(Move move) {

    return (uint8_t) (getCapturedPieceCode(move) & 7);
}

uint8_t getCapturedColour(Move move) {
    return (uint8_t) (getCapturedPieceCode(move) >> 3);
}



uint8_t getCastlingRights(uint32_t gamestate) {


    return (uint8_t) (gamestate & GS_castlingRightsMask);
}

uint8_t isBlackToMove(uint32_t gamestate) {

    return (uint8_t) ((gamestate & GS_colourtoMoveMask));  // since this would return 0 for white's turn, we can just keep this since itll act as a bool anyways

}

uint8_t getHalfmoveClock(uint32_t gamestate) {

    return (uint8_t) ((gamestate & GS_halfmoveClockMask) >> 10);

}

uint8_t getEnPassantSquare(uint32_t gamestate) {

    return (uint8_t) ((gamestate & GS_enpassantSquareMask) >> 4);
}

void setCastlingRights(uint32_t* gamestate, uint8_t state) {

    *gamestate = (*gamestate & ~GS_castlingRightsMask) | (state & 0xFU);

}

void setColourToMove(uint32_t* gamestate, uint8_t state) {

    *gamestate = (*gamestate & ~GS_colourtoMoveMask) | ((state & 0x1U) << 17);

}

void setHalfmoveClock(uint32_t* gamestate, uint8_t state) {

    *gamestate = (*gamestate & ~GS_halfmoveClockMask) | ((state & 0x7FU) << 10);

}

void setEnPassantSquare(uint32_t* gamestate, uint8_t state) {

    *gamestate = (*gamestate & ~GS_enpassantSquareMask) | ((state & 0x3FU) << 10);

}

void incrHalfmoveClock(uint32_t* gamestate) {
    setHalfmoveClock(gamestate, getHalfmoveClock(*gamestate) + 1);
}

void orCastlingRights(uint32_t* gamestate, uint8_t field) {
    *gamestate |= (field & 0xf);
}

bool canWhiteCastleLong(uint32_t gamestate) {

    return gamestate & whiteLongCastleMask;

}

bool canWhiteCastleShort(uint32_t gamestate) {

    return gamestate & whiteShortCastleMask;

}

bool canBlackCastleLong(uint32_t gamestate) {

    return gamestate & blackLongCastleMask;

}

bool canBlackCastleShort(uint32_t gamestate) {

    return gamestate & blackShortCastleMask;

}

unsigned int getMoveCount(unsigned int ply) {
    return ply / 2 + 1;
}