#include "legal.h"


bool kingInCheck(Board* b, int blackToMove) {
    uint64_t opponentAttacks = 0;

    uint64_t kingBitboard = b->bitboards[6 + 6 * blackToMove];  // get bitboard of the king of the side to move

    // or all attacks
    for (int i = 6 * blackToMove; i < 6 + 6 * blackToMove; i++) {
        uint64_t pieceBitboard = b->bitboards[i];
        while (pieceBitboard) {
            uint64_t moveMask = pieceBitboard & -pieceBitboard;  // get least significant bit
            pieceBitboard &= pieceBitboard - 1;  // clear least significant bit

            Square src = (Square) __builtin_ctzll(moveMask);  // get source square

            opponentAttacks |= pieceGenerator[i](b, src);  // add attacks from this piece to opponentAttacks
        }
    }

    return kingBitboard & opponentAttacks;
}

// for now, play, check king, unmove
bool isLegalMove(Board* b, Move move) {

    Gamestate gamestate = b->gamestate;

    bool blackToMove = isBlackToMove(gamestate);

    Undo64 undo = createUndo64(move, gamestate);
    makeMove(b, move);  // now white to move after this

    // check if the king is in check after the move
    if (kingInCheck(b, blackToMove ? 1 : 0)) {
        // revert the move
        performUndo(b, undo);
        return false;
    }

    // revert the move
    performUndo(b, undo);
    return true;
}

uint64_t getLegalFromPseudo(Board* b, uint64_t pesudoMoves, Square src) {

    uint64_t legalMoves = 0;
    Piece piece = b->pieces[src];
    bool promo = ((piece == WP && squareBitboards[src] & rank7) || (piece == BP && squareBitboards[src] & rank2)) ? true : false;  // for now, we will not generate promo moves, so this is just a placeholder. we can set this to the correct promo type when we implement promo move generation

    // play each move (each bit) and or it to 0 if legal
    while (pesudoMoves) {
        uint64_t moveMask = pesudoMoves & -pesudoMoves;  // get least significant bit
        pesudoMoves &= pesudoMoves - 1;  // clear least significant bit

        Square dst = (Square) __builtin_ctzll(moveMask);  // get dst

        // promo decision does not happen here. Decision does not affect legality
        Move move = getMoveFromSquare(b, src, dst, promo);
        if (isLegalMove(b, move)) {
            legalMoves |= moveMask;  // add move to legal moves
        }   
    }

    return legalMoves;
}