#include "main.h"

#ifdef _WIN32
#include <windows.h>
#endif

char* const startFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";


int main(int argc, char** argv) {

    #ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    #endif

    char* fen;
    fen = startFen;  // default

    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  --help, -h       Show this help message and exit\n");
            printf("  --fen <FEN>     Start a game with the given FEN string\n");
            return 0;
        }

        if (strcmp(argv[i], "--fen") == 0) {
            if (i + 1 < argc) {

                if (validFen(argv[i + 1])) {

                    fen = argv[i + 1];

                } else if (isdigit(argv[i + 1][0])) {

                    int fenIndex = strtol(argv[i + 1], NULL, 10);
                    
                    if (fenIndex >= 0 && fenIndex < sizeOfTestFens) {  // TODO: this will fail sometimes because the sizes are not uniform

                        fen = (char*)testFens[fenIndex];

                    } else {
                        fprintf(stderr, "Error: Invalid FEN index\n");
                        return 1;
                    }
                } else {
                    fprintf(stderr, "Error: Invalid FEN string\n");
                    return 1;
                }
                i++; // Skip the FEN argument

            } else {
                fprintf(stderr, "Error: --fen option requires a FEN string argument or number\n");
                return 1;
            }
        }
    }

    // init board

    // init game state

    // init pieces
    precomputeKnights();
    precomputeKingMoves();

    // init "clock"

    // init "players"

    // print board

    // enter game loop


    // on exit, print exit information

    Player white = {HUMAN, WHITE, NULL};
    Player black = {HUMAN, BLACK, NULL};

    Game* game = initGame(fen, white, black, HvH);
    cliMainLoop(game, NULL);
    

    return 0;
}
