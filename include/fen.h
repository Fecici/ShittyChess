#ifndef FEN_HEADER
#define FEN_HEADER

#include "definitions.h"
#include "bitUtils.h"
#include "zobrist.h"

bool loadFromFen(Board* b, char* fen);
bool isCharInt(const char c);
Piece getPieceFromChar(const char c);

// convert position to fen (lets call this with a flag in the fen cmd)
char* convertToFen(Board* b);

#endif
