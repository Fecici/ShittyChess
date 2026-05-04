#include "parse.h"

Move getMoveFromNotation(Board* b, char* moveStr) {

    // we assume valid notation at this point

    Move m;

    return m;
}

bool validMoveNotation(char* moveStr) {

    char promo;

    if (strncmp(moveStr, "O-O-O", 5) == 0) {
        return true;
    }

    if (strncmp(moveStr, "O-O", 3) == 0) {
        return true;
    }

    if (strnlen(moveStr, 4) != 4) {
        if (strnlen(moveStr, 5) == 5) {
            goto handlePromotion;
        }
        return false;
    }

    normalAlgebraMoveStrHandle:
    char srcFile, srcRank, dstFile, dstRank;
    srcFile = moveStr[0];
    dstFile = moveStr[2];
    srcRank = moveStr[1];
    dstRank = moveStr[3];

    if (!(srcFile >= 'a' && srcFile <= 'h' && dstFile >= 'a' && dstFile <= 'h')) { return false; }
    if (!(srcRank >= '1' && srcRank <= '8' && dstRank >= '1' && dstRank <= '8')) { return false; }

    return true;  // we will not worry about the fact that the rank must be 1 or 8 depending on the promo, but we will just let the application deal with this

    handlePromotion:
    promo = moveStr[4];
    char tester = promo ^ 'q' ^ 'b' ^ 'r' ^ 'n';
    if (tester == 0 && promo != 0) {
        goto normalAlgebraMoveStrHandle;
    }

    return false;
}

// this needs to turn all whitespace into a '\0' and count the args. this also mutates argv
int tokenize(char* line, char** argv) {

    int argc = 0;
    char* split = line;
    while (*split) {
        // skip until whitespace
        while (*split && isspace((unsigned char) *split)) split++;
        if (!*split) break;
        if (argc >= MAX_ARG - 1) break;

        if (*split == '"') {
            split++;  // skip opening quote
            argv[argc++] = split;  // token starts after quote

            while (*split && *split != '"') {
                split++;
            }

            if (*split == '"') {
                *split = '\0';  // terminate quoted argument
                split++;
            } else {
                fprintf(stderr, "Error: unmatched quote\n");
                return -1;
            } 
        }
        else {

            argv[argc] = split;
            argc++;
    
            while (*split && !isspace((unsigned char) *split)) split++;
            // fill whitespace with \0
            if (*split) {
                *split = '\0';
                split++;
            }
        }
    }
    argv[argc] = NULL;  // null terminate the argv array
    return argc;

}

void getInput(char* input, size_t size) {

    printf("\n>>> ");
    if (!fgets(input, (int) size, stdin)) {
        fprintf(stderr, "Error reading command, try again...\n"); 
        return getInput(input, size);
    }

    // strip
    input[strcspn(input, "\r\n")] = '\0';
}
