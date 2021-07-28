
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QtGui>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QFileDialog>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "yolo_v2_class.h"
#include "thread"

using namespace std;
using namespace cv;

std::vector<std::string> objects_names_from_file(std::string const filename) {
    std::ifstream file(filename);
    std::vector<std::string> file_lines;
    if (!file.is_open()) return file_lines;
    for (std::string line; file >> line;) file_lines.push_back(line);
    std::cout << "object names loaded \n";
    return file_lines;
}
void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names) {

    for (auto &i : result_vec) {
            cv::Scalar color(60, 160, 260);
            cv::Scalar color2(120, 60, 150);
            cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 3);

            if (obj_names.size() >= i.obj_id)
            {
                string w = to_string(i.w);
                string h = to_string(i.h);
                putText(mat_img, obj_names[i.obj_id], cv::Point2f(i.x, i.y - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, color2);
                //putText(mat_img, h, cv::Point2f(i.x, i.y - 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, color);
                //putText(mat_img, w, cv::Point2f(i.x, i.y - 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, color);
            }
            /*if (i.track_id > 0)
                    putText(mat_img, std::to_string(i.track_id), cv::Point2f(i.x + 5, i.y + 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, color);*/
    }
    namedWindow("Result",WINDOW_AUTOSIZE);
    moveWindow("Result",300,50);
    cv::imshow("Result", mat_img);
}

void show_result(std::vector<bbox_t> const result_vec, std::vector<std::string> const obj_names) {
    for (auto &i : result_vec) {
            if (obj_names.size() > i.obj_id) std::cout << obj_names[i.obj_id] << " - ";
            std::cout << "obj_id = " << i.obj_id << ",  x = " << i.x << ", y = " << i.y
                    << ", w = " << i.w << ", h = " << i.h
                    << std::setprecision(3) << ", prob = " << i.prob << std::endl;

    }
}

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{    
    ui->setupUi(this);

    QPalette pal = palette();
    pal.setColor(QPalette::Background, Qt::white);
    this->setAutoFillBackground(true);
    this->setPalette(pal);

    try
    {
        this->detector = new Detector("../data/lightDenseYOLO.cfg", "../data/lightDenseYOLO.weights");
        obj_names = objects_names_from_file("../data/coco.names");
    }
    catch(std::exception &e){ std::cerr << "exception: " << e.what() << "\n"; getchar(); }
    catch (...) { std::cerr << "unknown exception \n"; getchar(); }

    QPalette sample_palette;
    sample_palette.setColor(QPalette::Window, Qt::white);
    sample_palette.setColor(QPalette::WindowText, Qt::blue);

    ui->lblStatus->setAutoFillBackground(true);
    ui->lblStatus->setPalette(sample_palette);
    ui->lblStatus->setText("Ready");
}

MainWindow::~MainWindow()
{
    delete ui;
}



void MainWindow::on_actionReset_triggered()
{
    ui->txtPath->clear();
}

void MainWindow::on_actionExit_Program_triggered()
{
    QApplication::quit();
}

void MainWindow::on_btnBrowse_clicked()
{