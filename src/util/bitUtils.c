#include "../../include/bitUtil.h"

uint8_t getBitboardIndex(uint8_t piece) {

    return (getPieceType(piece) - 1 + 6 * getPieceColour(piece));

};

uint8_t getPieceType(uint8_t piece) {
    return piece & 7;
}

uint8_t getPieceColour(uint8_t piece) {
    return piece >> 3;
};

uint8_t getSrc(Move move) {
    return move & sourceMask;
};

uint8_t getDst(Move move) {
    return (move & targetMask) >> 6;
};

uint8_t getEnPassant(Move move) {
    return (move & enPassantMask) >> 15;
};

bool isCastled(Move move) {

    return (move & castleMask);
}

uint8_t getPromotion(Move move) {
    return (move & promoMask) >> 12;
}

uint8_t getCapturedPieceCode(Move move) {
    return (move & capturedPieceMask) >> 21;
}

uint8_t getCapturedType(Move move) {

    return getCapturedPieceCode(move) & 7;
}

uint8_t getCapturedColour(Move move) {
    return getCapturedPieceCode(move) >> 3;
}