
#include "mainwindow.h"
#include <QApplication>
#include <QDesktopWidget>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    a.setWindowIcon(QIcon("../data/yolo.ico"));

    int W = 500;
    int H = 500;
    int screenWidth;
    int screenHeight;
    int x, y;

    QDesktopWidget desktop;

    screenWidth = desktop.screen()->width();