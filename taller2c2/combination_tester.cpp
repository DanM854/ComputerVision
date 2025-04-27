#include <stdint.h>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <iomanip>
#include <tuple>
#include <algorithm>

#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

// Estructura para almacenar resultados
struct MatchResult {
    int numMatches;
    int numGoodMatches;
    double processingTime;
    bool homographySuccess;
};

// Función para crear un detector
Ptr<Feature2D> createDetector(const string& detectorName) {
    if (detectorName == "SIFT") {
        return SIFT::create(500);
    } else if (detectorName == "SURF") {
        return SURF::create(100, 3, 3, false);
    } else if (detectorName == "ORB") {
        return ORB::create(700);
    } else if (detectorName == "BRISK") {
        return BRISK::create(30, 3, 1.0f);
    } else if (detectorName == "FAST") {
        return FastFeatureDetector::create(20);
    } else {
        cerr << "Detector no reconocido: " << detectorName << endl;
        return nullptr;
    }
}

// Función para crear un descriptor
Ptr<Feature2D> createDescriptor(const string& descriptorName) {
    if (descriptorName == "SIFT") {
        return SIFT::create(500);
    } else if (descriptorName == "SURF") {
        return SURF::create(100, 3, 3, false);
    } else if (descriptorName == "ORB") {
        return ORB::create(700);
    } else if (descriptorName == "BRISK") {
        return BRISK::create(30, 3, 1.0f);
    } else if (descriptorName == "BRIEF") {
        return BriefDescriptorExtractor::create(32);
    } else if (descriptorName == "FREAK") {
        return FREAK::create();
    } else {
        cerr << "Descriptor no reconocido: " << descriptorName << endl;
        return nullptr;
    }
}

// Función para crear un matcher
Ptr<DescriptorMatcher> createMatcher(const string& matcherName, bool isBinaryDescriptor) {
    if (matcherName == "BF") {
        if (isBinaryDescriptor) {
            return DescriptorMatcher::create("BruteForce-Hamming");
        } else {
            return DescriptorMatcher::create("BruteForce");
        }
    } else if (matcherName == "FLANN") {
        if (isBinaryDescriptor) {
            // Para descriptores binarios en FLANN
            Ptr<flann::IndexParams> indexParams = makePtr<flann::LshIndexParams>(6, 12, 1);
            Ptr<flann::SearchParams> searchParams = makePtr<flann::SearchParams>(50);
            return makePtr<FlannBasedMatcher>(indexParams, searchParams);
        } else {
            // Para descriptores flotantes en FLANN
            return DescriptorMatcher::create("FlannBased");
        }
    } else {
        cerr << "Matcher no reconocido: " << matcherName << endl;
        return nullptr;
    }
}

// Función para procesar una combinación específica
MatchResult processCombination(const Mat& img1, const Mat& img2, 
                               const string& detectorName, const string& descriptorName, 
                               const string& matcherName, bool saveResult = true,
                               bool isSpecificCombination = false) {  // Añadido parámetro para saber si es combinación específica
    MatchResult result;
    result.numMatches = 0;
    result.numGoodMatches = 0;
    result.processingTime = 0;
    result.homographySuccess = false;
    
    cout << "Procesando: " << detectorName << " (detector) + " 
         << descriptorName << " (descriptor) + " << matcherName << " (matcher)" << endl;
    
    // Iniciar cronómetro
    auto start = chrono::high_resolution_clock::now();
    
    try {
        // Crear detector y descriptor
        Ptr<Feature2D> detector = createDetector(detectorName);
        Ptr<Feature2D> descriptor = createDescriptor(descriptorName);
        
        if (!detector || !descriptor) {
            return result;
        }
        
        // Detectar keypoints
        vector<KeyPoint> keypoints1, keypoints2;
        
        if (detectorName == "BRIEF" || detectorName == "FREAK") {
            // BRIEF y FREAK son solo descriptores, usar FAST como detector
            Ptr<FastFeatureDetector> fastDetector = FastFeatureDetector::create(20);
            fastDetector->detect(img1, keypoints1);
            fastDetector->detect(img2, keypoints2);
        } else {
            detector->detect(img1, keypoints1);
            detector->detect(img2, keypoints2);
        }
        
        // Limitar keypoints
        const int MAX_KEYPOINTS = 500;
        if (keypoints1.size() > MAX_KEYPOINTS) {
            keypoints1.resize(MAX_KEYPOINTS);
        }
        if (keypoints2.size() > MAX_KEYPOINTS) {
            keypoints2.resize(MAX_KEYPOINTS);
        }
        
        cout << "Keypoints en imagen 1: " << keypoints1.size() << endl;
        cout << "Keypoints en imagen 2: " << keypoints2.size() << endl;
        
        // Calcular descriptores
        Mat descriptors1, descriptors2;
        descriptor->compute(img1, keypoints1, descriptors1);
        descriptor->compute(img2, keypoints2, descriptors2);
        
        if (descriptors1.empty() || descriptors2.empty()) {
            cerr << "No se pudieron calcular los descriptores" << endl;
            return result;
        }
        
        // Verificar si es descriptor binario
        bool isBinaryDescriptor = descriptorName == "ORB" || 
                                 descriptorName == "BRIEF" || 
                                 descriptorName == "BRISK" || 
                                 descriptorName == "FREAK";
        
        // Para algunos descriptores, convertir a CV_32F para FLANN
        if (matcherName == "FLANN" && !isBinaryDescriptor) {
            if (descriptors1.type() != CV_32F) {
                descriptors1.convertTo(descriptors1, CV_32F);
            }
            if (descriptors2.type() != CV_32F) {
                descriptors2.convertTo(descriptors2, CV_32F);
            }
        }
        
        // Crear matcher
        Ptr<DescriptorMatcher> matcher = createMatcher(matcherName, isBinaryDescriptor);
        if (!matcher) {
            return result;
        }
        
        // Matching
        vector<vector<DMatch>> knnMatches;
        try {
            matcher->knnMatch(descriptors1, descriptors2, knnMatches, 2);
        } catch (const Exception& e) {
            cerr << "Error en knnMatch: " << e.what() << endl;
            // Intentar con match regular como alternativa
            vector<DMatch> regularMatches;
            matcher->match(descriptors1, descriptors2, regularMatches);
            
            // Convertir a formato knnMatches
            knnMatches.resize(regularMatches.size());
            for (size_t i = 0; i < regularMatches.size(); i++) {
                knnMatches[i].push_back(regularMatches[i]);
                // Agregar match ficticio
                DMatch fictitiousMatch;
                fictitiousMatch.distance = regularMatches[i].distance * 1.5f;
                knnMatches[i].push_back(fictitiousMatch);
            }
        }
        
        result.numMatches = knnMatches.size();
        
        // Filtrar buenos matches
        vector<DMatch> goodMatches;
        const float RATIO_THRESHOLD = isBinaryDescriptor ? 0.8f : 0.75f;
        
        for (size_t i = 0; i < knnMatches.size(); i++) {
            if (knnMatches[i].size() >= 2 && 
                knnMatches[i][0].distance < RATIO_THRESHOLD * knnMatches[i][1].distance) {
                goodMatches.push_back(knnMatches[i][0]);
            }
        }
        
        result.numGoodMatches = goodMatches.size();
        
        cout << "Total matches: " << result.numMatches << ", Good matches: " << result.numGoodMatches << endl;
        
        // Encontrar homografía
        Mat homography;
        if (goodMatches.size() >= 4) {
            vector<Point2f> obj;
            vector<Point2f> scene;
            
            for (size_t i = 0; i < goodMatches.size(); i++) {
                if (goodMatches[i].queryIdx < (int)keypoints1.size() && 
                    goodMatches[i].trainIdx < (int)keypoints2.size()) {
                    obj.push_back(keypoints1[goodMatches[i].queryIdx].pt);
                    scene.push_back(keypoints2[goodMatches[i].trainIdx].pt);
                }
            }
            
            if (obj.size() >= 4 && scene.size() >= 4) {
                homography = findHomography(obj, scene, RANSAC);
                result.homographySuccess = !homography.empty();
            }
        }
        
        // Guardar y mostrar resultado visual
        if (saveResult && !goodMatches.empty()) {
            Mat imgMatches;
            drawMatches(img1, keypoints1, img2, keypoints2, goodMatches, imgMatches,
                       Scalar::all(-1), Scalar::all(-1), vector<char>(),
                       DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            
            if (result.homographySuccess) {
                // Dibujar contorno del objeto
                vector<Point2f> objCorners(4);
                objCorners[0] = Point2f(0, 0);
                objCorners[1] = Point2f((float)img1.cols, 0);
                objCorners[2] = Point2f((float)img1.cols, (float)img1.rows);
                objCorners[3] = Point2f(0, (float)img1.rows);
                
                vector<Point2f> sceneCorners(4);
                perspectiveTransform(objCorners, sceneCorners, homography);
                
                line(imgMatches, sceneCorners[0] + Point2f((float)img1.cols, 0),
                    sceneCorners[1] + Point2f((float)img1.cols, 0), Scalar(0, 255, 0), 4);
                line(imgMatches, sceneCorners[1] + Point2f((float)img1.cols, 0),
                    sceneCorners[2] + Point2f((float)img1.cols, 0), Scalar(0, 255, 0), 4);
                line(imgMatches, sceneCorners[2] + Point2f((float)img1.cols, 0),
                    sceneCorners[3] + Point2f((float)img1.cols, 0), Scalar(0, 255, 0), 4);
                line(imgMatches, sceneCorners[3] + Point2f((float)img1.cols, 0),
                    sceneCorners[0] + Point2f((float)img1.cols, 0), Scalar(0, 255, 0), 4);
            }
            
            string fileName = "result_" + detectorName + "_" + descriptorName + "_" + matcherName + ".jpg";
            imwrite(fileName, imgMatches);
            
            // Mostrar el resultado
            string windowTitle = detectorName + "_" + descriptorName + "_" + matcherName;
            namedWindow(windowTitle, WINDOW_NORMAL);
            imshow(windowTitle, imgMatches);
            
            // Si es una combinación específica, esperar a que el usuario cierre la ventana
            if (isSpecificCombination) {
                cout << "Presiona cualquier tecla para continuar..." << endl;
                waitKey(0);
            } else {
                // Si estamos procesando todas las combinaciones, solo mostrar brevemente
                waitKey(500);
            }
            
            destroyWindow(windowTitle);
        }
    } catch (const Exception& e) {
        cerr << "Error de OpenCV: " << e.what() << endl;
    } catch (const exception& e) {
        cerr << "Error de C++: " << e.what() << endl;
    } catch (...) {
        cerr << "Error desconocido" << endl;
    }
    
    // Medir tiempo
    auto end = chrono::high_resolution_clock::now();
    result.processingTime = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    
    cout << "Tiempo de procesamiento: " << result.processingTime << " ms" << endl;
    cout << "Homografía exitosa: " << (result.homographySuccess ? "Sí" : "No") << endl;
    cout << "--------------------------------" << endl;
    
    return result;
}

// Función para verificar si una combinación es válida
bool isCombinationValid(const string& detector, const string& descriptor) {
    // BRIEF y FREAK solo son descriptores, no detectores
    if ((detector == "BRIEF" || detector == "FREAK")) {
        return false;
    }
    
    return true;
}

int main(int argc, char* argv[]) {
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
    
    // Redimensionar imágenes si son muy grandes
    const int MAX_SIZE = 800;
    if (img_object.cols > MAX_SIZE || img_object.rows > MAX_SIZE) {
        double scale = min(double(MAX_SIZE)/img_object.cols, double(MAX_SIZE)/img_object.rows);
        resize(img_object, img_object, Size(), scale, scale, INTER_AREA);
    }
    
    if (img_scene.cols > MAX_SIZE || img_scene.rows > MAX_SIZE) {
        double scale = min(double(MAX_SIZE)/img_scene.cols, double(MAX_SIZE)/img_scene.rows);
        resize(img_scene, img_scene, Size(), scale, scale, INTER_AREA);
    }
    
    // Definir detectores, descriptores y matchers
    vector<string> detectors = {"SIFT", "SURF", "ORB", "FAST", "BRISK"};
    vector<string> descriptors = {"SIFT", "SURF", "ORB", "BRIEF", "FREAK", "BRISK"};
    vector<string> matchers = {"BF", "FLANN"};
    
    // Determinar qué combinación procesar
    string requestedDetector, requestedDescriptor, requestedMatcher;
    bool processAll = false;
    
    if (argc >= 4) {
        // Si se proporcionan argumentos, procesar la combinación específica
        requestedDetector = argv[1];
        requestedDescriptor = argv[2];
        requestedMatcher = argv[3];
    } else {
        // De lo contrario, mostrar menú
        cout << "Selecciona una opción:" << endl;
        cout << "1. Procesar todas las combinaciones válidas" << endl;
        cout << "2. Seleccionar una combinación específica" << endl;
        
        int option;
        cin >> option;
        
        if (option == 1) {
            processAll = true;
        } else if (option == 2) {
            // Mostrar detectores disponibles
            cout << "Detectores disponibles:" << endl;
            for (size_t i = 0; i < detectors.size(); i++) {
                cout << i+1 << ". " << detectors[i] << endl;
            }
            
            int detectorIdx;
            cout << "Selecciona un detector (1-" << detectors.size() << "): ";
            cin >> detectorIdx;
            
            if (detectorIdx < 1 || detectorIdx > (int)detectors.size()) {
                cerr << "Índice de detector inválido" << endl;
                return -1;
            }
            
            requestedDetector = detectors[detectorIdx-1];
            
            // Mostrar descriptores disponibles
            cout << "Descriptores disponibles:" << endl;
            for (size_t i = 0; i < descriptors.size(); i++) {
                cout << i+1 << ". " << descriptors[i] << endl;
            }
            
            int descriptorIdx;
            cout << "Selecciona un descriptor (1-" << descriptors.size() << "): ";
            cin >> descriptorIdx;
            
            if (descriptorIdx < 1 || descriptorIdx > (int)descriptors.size()) {
                cerr << "Índice de descriptor inválido" << endl;
                return -1;
            }
            
            requestedDescriptor = descriptors[descriptorIdx-1];
            
            // Mostrar matchers disponibles
            cout << "Matchers disponibles:" << endl;
            for (size_t i = 0; i < matchers.size(); i++) {
                cout << i+1 << ". " << matchers[i] << endl;
            }
            
            int matcherIdx;
            cout << "Selecciona un matcher (1-" << matchers.size() << "): ";
            cin >> matcherIdx;
            
            if (matcherIdx < 1 || matcherIdx > (int)matchers.size()) {
                cerr << "Índice de matcher inválido" << endl;
                return -1;
            }
            
            requestedMatcher = matchers[matcherIdx-1];
        } else {
            cerr << "Opción inválida" << endl;
            return -1;
        }
    }
    
    // Mapa para almacenar resultados
    map<tuple<string, string, string>, MatchResult> results;
    
    if (processAll) {
        cout << "Procesando todas las combinaciones válidas..." << endl;
        
        for (const string& detector : detectors) {
            for (const string& descriptor : descriptors) {
                if (!isCombinationValid(detector, descriptor)) {
                    continue;
                }
                
                for (const string& matcher : matchers) {
                    // Para FLANN con descriptores binarios, se necesita manejo especial
                    bool isBinaryDescriptor = descriptor == "ORB" || descriptor == "BRIEF" || 
                                            descriptor == "BRISK" || descriptor == "FREAK";
                                            
                    // Omitir FLANN con descriptores binarios en versiones antiguas de OpenCV
                    if (matcher == "FLANN" && isBinaryDescriptor) {
                        continue;
                    }
                    
                    auto key = make_tuple(detector, descriptor, matcher);
                    results[key] = processCombination(img_object, img_scene, detector, descriptor, matcher, true, false);
                    
                    // Liberar recursos
                    waitKey(500);
                    destroyAllWindows();
                }
            }
        }
    } else {
        // Procesar solo la combinación seleccionada
        if (!isCombinationValid(requestedDetector, requestedDescriptor)) {
            cerr << "Combinación inválida: " << requestedDetector << " + " << requestedDescriptor << endl;
            return -1;
        }
        
        auto key = make_tuple(requestedDetector, requestedDescriptor, requestedMatcher);
        results[key] = processCombination(img_object, img_scene, requestedDetector, requestedDescriptor, requestedMatcher, true, true);
    }
    
    // Mostrar tabla de resultados
    cout << "\n=== RESULTADOS COMPARATIVOS ===" << endl;
    cout << setw(25) << "Combinación" << setw(12) << "Matches" << setw(12) << "Good" 
         << setw(12) << "Tiempo (ms)" << setw(15) << "Homografía" << endl;
    cout << string(76, '-') << endl;
    
    for (const auto& result : results) {
        string combination = get<0>(result.first) + "_" + get<1>(result.first) + "_" + get<2>(result.first);
        cout << setw(25) << combination 
             << setw(12) << result.second.numMatches 
             << setw(12) << result.second.numGoodMatches
             << setw(12) << result.second.processingTime
             << setw(15) << (result.second.homographySuccess ? "Sí" : "No") 
             << endl;
    }
    
    // Encontrar mejor combinación basada en buenos matches
    auto bestMatch = max_element(results.begin(), results.end(),
        [](const pair<tuple<string, string, string>, MatchResult>& a, 
           const pair<tuple<string, string, string>, MatchResult>& b) {
            return a.second.numGoodMatches < b.second.numGoodMatches;
        });
    
    // Encontrar combinación más rápida
    auto fastestMatch = min_element(results.begin(), results.end(),
        [](const pair<tuple<string, string, string>, MatchResult>& a, 
           const pair<tuple<string, string, string>, MatchResult>& b) {
            return a.second.processingTime < b.second.processingTime;
        });
    
    if (bestMatch != results.end()) {
        cout << "\nMejor combinación (más matches): " 
             << get<0>(bestMatch->first) << "_" << get<1>(bestMatch->first) << "_" << get<2>(bestMatch->first)
             << " con " << bestMatch->second.numGoodMatches << " buenos matches" << endl;
    }
    
    if (fastestMatch != results.end()) {
        cout << "Combinación más rápida: " 
             << get<0>(fastestMatch->first) << "_" << get<1>(fastestMatch->first) << "_" << get<2>(fastestMatch->first)
             << " con " << fastestMatch->second.processingTime << " ms" << endl;
    }
    
    if (processAll) {
        // Mostrar las imágenes de las mejores combinaciones
        cout << "\nMostrando resultado de la mejor combinación. Presiona cualquier tecla para cerrar..." << endl;
        
        if (bestMatch != results.end()) {
            string bestFileName = "result_" + get<0>(bestMatch->first) + "_" + 
                                 get<1>(bestMatch->first) + "_" + 
                                 get<2>(bestMatch->first) + ".jpg";
            
            Mat bestImage = imread(bestFileName);
            if (!bestImage.empty()) {
                string windowTitle = "Mejor combinación: " + get<0>(bestMatch->first) + "_" + 
                                    get<1>(bestMatch->first) + "_" + 
                                    get<2>(bestMatch->first);
                namedWindow(windowTitle, WINDOW_NORMAL);
                imshow(windowTitle, bestImage);
                waitKey(0);
                destroyWindow(windowTitle);
            }
        }
    }
    
    cout << "\nPrograma finalizado." << endl;
    
    return 0;
}
