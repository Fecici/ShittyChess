#ifndef PARSE_HEADER
#define PARSE_HEADER

#include "definitions.h"
#include "command.h"

// this needs to turn all whitespace into a '\0' and count the args. this also mutates argv
int tokenize(char* line, char** argv);
void getInput(char* input, size_t size);

static inline bool validSquareNotation(char* squareStr) {
    return strlen(squareStr) == 2 &&
           squareStr[0] >= 'a' && squareStr[0] <= 'h' &&
           squareStr[1] >= '1' && squareStr[1] <= '8';
}

static inline Square getSquareFromNotation(char* squareStr) {
    if (!validSquareNotation(squareStr)) {
        fprintf(stderr, "Error: Invalid square notation: %s\n", squareStr);
        return -1;  // invalid square
    }

    char file = squareStr[0];
    char rank = squareStr[1];

    return (Square) ((rank - '1') * 8 + (file - 'a'));
}

#endif
