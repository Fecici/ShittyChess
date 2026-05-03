#ifndef FEN_HEADER
#define FEN_HEADER

#include "definitions.h"
#include "bitUtils.h"
#include "zobrist.h"

bool loadFromFen(Board* b, char* fen);
bool isCharInt(const char c);
bool validFen(const char* fen);
Piece getPieceFromChar(const char c);
char getCharFromPiece(Piece piece);

// convert position to fen (lets call this with a flag in the fen cmd)
char* convertToFen(Board* b);

#endif
