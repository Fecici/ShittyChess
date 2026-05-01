#include "ui.h"

void initUI(UI* ui, char* name, GameType gt, void (*messager)(), void (*renderer)()) {
    ui->name = name;
    ui->gametype = gt;
    ui->message = messager;
    ui->render = renderer;
}

void ascii_render() {}
void stdout_messager() {}

