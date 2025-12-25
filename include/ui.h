#ifndef UI_HEADER
#define UI_HEADER

typedef enum {HvE, EvE, HvH} GameType;

typedef struct {

    char* name;
    GameType gametype;
    void (*message)();
    void (*render)();

} UI;

void initUI(UI* ui, char* name, GameType gt, void (*messager)(), void (*renderer)());

// if i choose to make different renderers or messagers, this might help.
// i am also in complete violation of the software engineering principles
// that i learned in class :)
void ascii_render();
void stdout_message();


#endif