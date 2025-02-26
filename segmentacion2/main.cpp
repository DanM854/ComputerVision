#include<iostream>
#include<vector>
#include<opencv2/opencv.hpp>

//variables globales
cv::Mat src_img;
cv::Mat img;

//rectangulo actual
cv::Rect rect(0,0,0,0);

//start and end points del rectangulo
cv::Point p1(0,0);
cv::Point p2(0,0);

//uso del mouse
static bool clicked = false;

//funcion para evitar que el rectangulo se salga de la imagen original

void arreglar_bordes(){
    if(rect.width > img.cols - rect.x)
        rect.width = img.cols - rect.x;
    if(rect.height > img.rows - rect.y)
        rect.height = img.rows - rect.y;
    if(rect.x < 0)
        rect.x = 0;
    if(rect.y < 0)
        rect.y = 0;
}

//funcion para dibujar el rectangulo
void dibujar(){
    img = src_img.clone();
    arreglar_bordes();
    cv::rectangle(img, rect, cv::Scalar(0,255,0),1,8,0);
    cv::imshow("Imagen original", img);
}

//funcion para controlar el rectangulo usando el mouse

void mouse(int event, int x, int y, int flag, void* user_data){
    switch (event) {
    case cv::EVENT_LBUTTONDOWN:
        clicked = true;
        p1.x = x;
        p1.y = y;
        p2.x = x;
        p2.y = y;
        break;
    case cv::EVENT_LBUTTONUP:
        clicked = false;
        p2.x = x;
        p2.y = y;
        break;
    case cv::EVENT_MOUSEMOVE:
        if(clicked){
            p2.x = x;
            p2.y = y;
        }
        break;

    default:
        break;
    }
    if(p1.x > p2.x){
        rect.x = p2.x;
        rect.width = p1.x - p2.x;
    }
    else{
        rect.x = p1.x;
        rect.width = p2.x - p1.x;
    }

    if(p1.y > p2.y){
        rect.y = p2.y;
        rect.height = p1.y - p2.y;
    }
    else{
        rect.y = p1.y;
        rect.height = p2.y - p1.y;
    }
    dibujar();
}

int main(){
    src_img = cv::imread("../Data/pajaro.jpg");
    cv::namedWindow("Imagen original", cv::WINDOW_NORMAL);
    cv::setMouseCallback("Imagen original", mouse, NULL);
    cv::imshow("Imagen Original", src_img);


    //variable para guardar el resultado de la segmentacion
    cv::Mat result;

    cv::Mat bgmodel, fgmodel;

    while(1){
        char c = cv::waitKey();
        cv::grabCut(src_img, result, rect, bgmodel, fgmodel, 5, cv::GC_INIT_WITH_RECT);

        cv::compare(result, cv::GC_PR_FGD, result, cv::CMP_EQ);
        cv::Mat foreground(src_img.size(), CV_8UC3, cv::Scalar(255,255,255));
        src_img.copyTo(foreground, result);
        cv::namedWindow("Imagen Segmentada");
        cv::imshow("Imagen segmentada", foreground);
    }

    return 0;
}
