
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