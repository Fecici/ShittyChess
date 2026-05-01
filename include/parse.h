#ifndef PARSE_HEADER
#define PARSE_HEADER

#include "definitions.h"
#include "command.h"

Move getMoveFromNotation(Board* b, char* moveStr);
bool validMoveNotation(char* moveStr);

// this needs to turn all whitespace into a '\0' and count the args. this also mutates argv
int tokenize(char* line, char** argv);
void getInput(char* input, size_t size);

#endif
