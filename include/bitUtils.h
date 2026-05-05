#ifndef BIT_UTIL_HEADER
#define BIT_UTIL_HEADER


#include "definitions.h"

static inline PieceType getPieceType(Piece piece) {
    return piece & 7;
}

static inline char getCharFromPiece(Piece piece) {
    if (piece == EMPTY) return ' ';
    
    switch (type) {
        case WP: return 'P';
        case WN: return 'N';
        case WB: return 'B';
        case WR: return 'R';
        case WQ: return 'Q';
        case WK: return 'K';
        case BP: return 'p';
        case BN: return 'n';
        case BB: return 'b';
        case BR: return 'r';
        case BQ: return 'q';
        case BK: return 'k';
        default: return ' ';
    }

}

static inline Piece getPieceFromChar(char c) {
    switch (c) {
        case 'P': return WP;
        case 'N': return WN;
        case 'B': return WB;
        case 'R': return WR;
        case 'Q': return WQ;
        case 'K': return WK;
        case 'p': return BP;
        case 'n': return BN;
        case 'b': return BB;
        case 'r': return BR;
        case 'q': return BQ;
        case 'k': return BK;
        default: return EMPTY;  // empty or invalid
    }
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

static inline Piece getCapturedPiece(Move move) {
    return (Piece) ((move & capturedPieceMask) >> 21);
}

static inline PieceType getCapturedType(Move move) {

    return (PieceType) (getCapturedPiece(move) & 7);
}

static inline uint8_t getCapturedColour(Move move) {
    return (uint8_t) (getCapturedPiece(move) >> 3);
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

static inline void setCapturedPiece(Move* move, Piece captured) {
    *move = (*move & ~capturedPieceMask) | ((captured << 21) & capturedPieceMask);
}

static inline void setCapturedColour(Move* move, uint8_t colour) {
    *move |= (colour & 1) << 24;
}

static inline void setCastled(Move* move, bool castled) {
    if (castled) {
        *move |= castleMask;
    } else {
        *move &= ~castleMask;
    }
}

static inline void setCapturedType(Move* move, PieceType type) {
    uint8_t captured = getCapturedPiece(*move);
    captured = (captured & 0x18) | (type & 7);
    setCapturedPiece(move, captured);
}

static inline uint8_t getCastlingRights(Gamestate gamestate) {
    return (uint8_t) (gamestate & GS_castlingRightsMask);
}


static inline uint8_t getHalfmoveClock(Gamestate gamestate) {
    
    return (uint8_t) ((gamestate & GS_halfmoveClockMask) >> 10);
    
}

static inline uint8_t getEnPassantSquare(Gamestate gamestate) {
    
    return (uint8_t) ((gamestate & GS_enpassantSquareMask) >> 4);
}

static inline void setCastlingRights(Gamestate* gamestate, uint8_t state) {
    
    *gamestate = (*gamestate & ~GS_castlingRightsMask) | (state & 0xFU);
    
}

static inline void setColourToMove(Gamestate* gamestate, uint8_t state) {
    
    *gamestate = (*gamestate & ~GS_colourtoMoveMask) | ((state & 0x1U) << 17);
    
}

static inline uint8_t getColourToMove(Gamestate gamestate) {
    
    return (uint8_t) ((gamestate & GS_colourtoMoveMask) >> 17);
}

static inline bool isBlackToMove(Gamestate gamestate) {

    return getColourToMove(gamestate) == 1;  // since this would return 0 for white's turn, we can just keep this since itll act as a bool anyways

}

static inline void setHalfmoveClock(Gamestate* gamestate, uint8_t state) {

    *gamestate = (*gamestate & ~GS_halfmoveClockMask) | ((state & 0x7FU) << 10);

}

static inline void setEnPassantSquare(Gamestate* gamestate, uint8_t state) {

    *gamestate = (*gamestate & ~GS_enpassantSquareMask) | ((state & 0x3FU) << 4);

}

static inline void incrHalfmoveClock(Gamestate* gamestate) {
    setHalfmoveClock(gamestate, getHalfmoveClock(*gamestate) + 1);
}

static inline void orCastlingRights(Gamestate* gamestate, uint8_t field) {
    *gamestate |= (field & 0xf);
}

static inline bool canWhiteCastleLong(Gamestate gamestate) {

    return gamestate & whiteLongCastleMask;

}

static inline bool canWhiteCastleShort(Gamestate gamestate) {

    return gamestate & whiteShortCastleMask;

}

static inline bool canBlackCastleLong(Gamestate gamestate) {

    return gamestate & blackLongCastleMask;

}

static inline bool canBlackCastleShort(Gamestate gamestate) {

    return gamestate & blackShortCastleMask;

}

static inline unsigned int getMoveCount(unsigned int ply) {
    return ply / 2 + 1;
}

static inline uint8_t getBitboardIndex(uint8_t piece) {

    return (uint8_t) ((getPieceType(piece) - 1 + 6 * getPiecesColour(piece)));

};

#endif