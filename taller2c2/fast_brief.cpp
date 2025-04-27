#include <stdint.h>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

int main() {
    // Cargar imágenes
    string objectImagePath = "../Data/box.png";
    string sceneImagePath = "../Data/box_in_scene.png";
    
    Mat img_object = imread(objectImagePath, IMREAD_GRAYSCALE);
    Mat img_scene = imread(sceneImagePath, IMREAD_GRAYSCALE);
    
    if (img_object.empty() || img_scene.empty()) {
        cerr << "Error al cargar las imágenes. Probando rutas alternativas..." << endl;
        
        // Intentar con rutas alternativas
        objectImagePath = "../Data/ima1.png";
        sceneImagePath = "../Data/ima21.png";
        
        img_object = imread(objectImagePath, IMREAD_GRAYSCALE);
        img_scene = imread(sceneImagePath, IMREAD_GRAYSCALE);
        
        if (img_object.empty() || img_scene.empty()) {
            objectImagePath = "Data/box.png";
            sceneImagePath = "Data/box_in_scene.png";
            
            img_object = imread(objectImagePath, IMREAD_GRAYSCALE);
            img_scene = imread(sceneImagePath, IMREAD_GRAYSCALE);
            
            if (img_object.empty() || img_scene.empty()) {
                cerr << "No se pudieron cargar las imágenes. Verifica las rutas." << endl;
                return -1;
            }
        }
    }
    
    cout << "Imágenes cargadas correctamente." << endl;
    cout << "Analizando con FAST (detector) + BRIEF (descriptor) + BF (matcher)" << endl;
    
    // Redimensionar imágenes si son muy grandes (para evitar problemas de memoria)
    const int MAX_SIZE = 800;
    if (img_object.cols > MAX_SIZE || img_object.rows > MAX_SIZE) {
        double scale = min(double(MAX_SIZE)/img_object.cols, double(MAX_SIZE)/img_object.rows);
        resize(img_object, img_object, Size(), scale, scale, INTER_AREA);
    }
    
    if (img_scene.cols > MAX_SIZE || img_scene.rows > MAX_SIZE) {
        double scale = min(double(MAX_SIZE)/img_scene.cols, double(MAX_SIZE)/img_scene.rows);
        resize(img_scene, img_scene, Size(), scale, scale, INTER_AREA);
    }
    
    // Iniciar el cronómetro
    auto start = chrono::high_resolution_clock::now();
    
    // Crear detector FAST
    Ptr<FastFeatureDetector> fast = FastFeatureDetector::create(20); // Umbral más bajo = más puntos
    
    // Crear descriptor BRIEF
    Ptr<BriefDescriptorExtractor> brief = BriefDescriptorExtractor::create(32); // 32 bytes = 256 bits
    
    // Detectar keypoints con FAST
    vector<KeyPoint> keypoints_object, keypoints_scene;
    fast->detect(img_object, keypoints_object);
    fast->detect(img_scene, keypoints_scene);
    
    // Limitar el número de keypoints si hay demasiados
    const int MAX_KEYPOINTS = 1000;
    if (keypoints_object.size() > MAX_KEYPOINTS) {
        keypoints_object.resize(MAX_KEYPOINTS);
    }
    
    if (keypoints_scene.size() > MAX_KEYPOINTS) {
        keypoints_scene.resize(MAX_KEYPOINTS);
    }
    
    cout << "Keypoints en imagen objeto: " << keypoints_object.size() << endl;
    cout << "Keypoints en imagen escena: " << keypoints_scene.size() << endl;
    
    // Calcular descriptores con BRIEF
    Mat descriptors_object, descriptors_scene;
    brief->compute(img_object, keypoints_object, descriptors_object);
    brief->compute(img_scene, keypoints_scene, descriptors_scene);
    
    // Si no hay suficientes keypoints o descriptores, salir
    if (descriptors_object.empty() || descriptors_scene.empty()) {
        cerr << "No se pudieron calcular los descriptores. Verifica que hay suficientes keypoints." << endl;
        return -1;
    }
    
    // Matcher Brute Force para descriptores binarios (BRIEF)
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    vector<vector<DMatch>> knn_matches;
    matcher->knnMatch(descriptors_object, descriptors_scene, knn_matches, 2);
    
    // Filtrar matches usando el test de ratio de Lowe
    const float RATIO_THRESHOLD = 0.8f;
    vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i].size() >= 2 && 
            knn_matches[i][0].distance < RATIO_THRESHOLD * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    
    cout << "Total matches: " << knn_matches.size() << ", Good matches: " << good_matches.size() << endl;
    
    // Encontrar homografía si hay suficientes buenos matches
    Mat homography;
    bool homographySuccess = false;
    
    if (good_matches.size() >= 4) {
        vector<Point2f> obj;
        vector<Point2f> scene;
        
        for (auto& match : good_matches) {
            if (match.queryIdx < (int)keypoints_object.size() && 
                match.trainIdx < (int)keypoints_scene.size()) {
                obj.push_back(keypoints_object[match.queryIdx].pt);
                scene.push_back(keypoints_scene[match.trainIdx].pt);
            }
        }
        
        if (obj.size() >= 4 && scene.size() >= 4) {
            // Usar RANSAC para encontrar homografía robusta
            homography = findHomography(obj, scene, RANSAC);
            homographySuccess = !homography.empty();
        }
    }
    
    // Medir tiempo total
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    
    cout << "Tiempo de procesamiento: " << duration << " ms" << endl;
    cout << "Homografía exitosa: " << (homographySuccess ? "Sí" : "No") << endl;
    
    // Visualización de resultados
    Mat img_matches;
    drawMatches(img_object, keypoints_object, img_scene, keypoints_scene, good_matches, img_matches, 
               Scalar::all(-1), Scalar::all(-1), vector<char>(), 
               DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
    if (homographySuccess) {
        // Obtener las esquinas del objeto
        vector<Point2f> obj_corners(4);
        obj_corners[0] = Point2f(0, 0);
        obj_corners[1] = Point2f((float)img_object.cols, 0);
        obj_corners[2] = Point2f((float)img_object.cols, (float)img_object.rows);
        obj_corners[3] = Point2f(0, (float)img_object.rows);
        
        vector<Point2f> scene_corners(4);
        perspectiveTransform(obj_corners, scene_corners, homography);
        
        // Dibujar el objeto encontrado
        line(img_matches, scene_corners[0] + Point2f((float)img_object.cols, 0),
            scene_corners[1] + Point2f((float)img_object.cols, 0), Scalar(0, 255, 0), 4);
        line(img_matches, scene_corners[1] + Point2f((float)img_object.cols, 0),
            scene_corners[2] + Point2f((float)img_object.cols, 0), Scalar(0, 255, 0), 4);
        line(img_matches, scene_corners[2] + Point2f((float)img_object.cols, 0),
            scene_corners[3] + Point2f((float)img_object.cols, 0), Scalar(0, 255, 0), 4);
        line(img_matches, scene_corners[3] + Point2f((float)img_object.cols, 0),
            scene_corners[0] + Point2f((float)img_object.cols, 0), Scalar(0, 255, 0), 4);
    }
    
    // Mostrar y guardar resultados
    namedWindow("FAST_BRIEF_Matches", WINDOW_NORMAL);
    imshow("FAST_BRIEF_Matches", img_matches);
    imwrite("result_FAST_BRIEF_BF.jpg", img_matches);
    
    cout << "Análisis completo. Presiona cualquier tecla para salir." << endl;
    waitKey(0);
    
    return 0;
}
