#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <map>
#include <cmath>

using namespace cv;
using namespace std;

// Estructura para almacenar información de las monedas
struct CoinInfo {
    int value;           // Valor de la moneda
    double diameter_mm;  // Diámetro en mm
    Scalar color;        // Color para visualización
};

// Función para clasificar una moneda según su diámetro en mm
int classifyCoin(double diameter_mm, const vector<CoinInfo>& coin_types) {
    // Encontrar la moneda más cercana basada en el diámetro
    int best_value = 0;
    double min_diff = INFINITY;

    for (const auto& coin : coin_types) {
        double diff = abs(diameter_mm - coin.diameter_mm);
        if (diff < min_diff) {
            min_diff = diff;
            best_value = coin.value;
        }
    }

    return best_value;
}

int main() {
    // Definir tipos de monedas con sus valores y diámetros
    vector<CoinInfo> coin_types = {
        {1, 20.0, Scalar(200, 200, 200)},  // 1 Kč - diámetro 20mm
        {2, 21.5, Scalar(200, 255, 200)},  // 2 Kč - diámetro 21.5mm
        {5, 23.0, Scalar(200, 200, 255)},  // 5 Kč - diámetro 23mm
        {10, 24.5, Scalar(100, 100, 255)}, // 10 Kč - diámetro 24.5mm
        {20, 26.0, Scalar(255, 200, 100)}  // 20 Kč - diámetro 26mm
    };

    // Intentar diferentes rutas de acceso para la imagen
    vector<string> possible_image_paths = {
        "koruny_black.jpg",
        "../koruny_black.jpg",
        "../../koruny_black.jpg",
        "Data/koruny_black.jpg",
        "../Data/koruny_black.jpg",
        "Image2.jpg",
        "Data/Image2.jpg"
    };

    Mat img;
    for (const auto& path : possible_image_paths) {
        img = imread(path);
        if (!img.empty()) {
            cout << "Imagen cargada desde: " << path << endl;
            break;
        }
    }

    if (img.empty()) {
        cout << "No se pudo abrir la imagen. Verifica la ruta." << endl;
        return -1;
    }

    // Crear copia para visualización
    Mat img_display = img.clone();

    // PASO 1: PREPROCESAMIENTO

    // Convertir a escala de grises
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // Aplicar filtro bilateral para suavizar el ruido conservando los bordes
    Mat bilateral;
    bilateralFilter(gray, bilateral, 9, 75, 75);

    // Aplicar corrección gamma para mejorar contraste en áreas oscuras
    Mat gamma_corrected;
    double gamma = 2.0; // Valor gamma > 1 aclara áreas oscuras
    Mat lookup_table(1, 256, CV_8U);
    for (int i = 0; i < 256; i++)
        lookup_table.at<uchar>(0, i) = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
    LUT(bilateral, lookup_table, gamma_corrected);

    // Guardar imagen con corrección gamma para verificación
    imwrite("gamma_corregida.jpg", gamma_corrected);

    // PASO 2: APLICAR THRESHOLD DE 50 PARA ELIMINAR RUIDO DEL FONDO

    // Aplicar umbralización con valor de 50 como solicitado
    Mat binary;
    threshold(gamma_corrected, binary, 50, 255, THRESH_BINARY);

    // Guardar imagen binaria con threshold de 50
    imwrite("threshold_50.jpg", binary);

    // Aplicar filtro de mediana para eliminar cualquier ruido residual tipo sal y pimienta
    Mat median_filtered;
    medianBlur(binary, median_filtered, 5);

    // Operaciones morfológicas para mejorar la segmentación
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));

    // Erosión para eliminar pequeños puntos blancos (ruido)
    Mat eroded;
    erode(median_filtered, eroded, kernel);

    // Dilatación para recuperar tamaño original y rellenar huecos
    Mat dilated;
    dilate(eroded, dilated, kernel, Point(-1,-1), 2); // Dilatar dos veces

    // Guardar imágenes intermedias para verificación
    imwrite("filtrada_mediana.jpg", median_filtered);
    imwrite("erosionada.jpg", eroded);
    imwrite("dilatada.jpg", dilated);

    // PASO 3: DETECCIÓN DE CÍRCULOS

    // Aplicar transformada de Hough sobre la imagen binaria procesada
    vector<Vec3f> circles;
    HoughCircles(dilated, circles, HOUGH_GRADIENT, 1, 40, 100, 15, 50, 120);

    cout << "Se detectaron " << circles.size() << " círculos" << endl;

    // Si no se detectaron suficientes círculos, intentar con otro enfoque
    if (circles.size() < 10) {
        cout << "Intentando detección alternativa..." << endl;

        // Buscar contornos en la imagen binaria
        vector<vector<Point>> contours;
        findContours(dilated, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        cout << "Se encontraron " << contours.size() << " contornos" << endl;

        // Procesar cada contorno para encontrar círculos
        for (const auto& contour : contours) {
            // Filtrar contornos muy pequeños
            double area = contourArea(contour);
            if (area < 1000) continue;

            // Encontrar círculo mínimo que encierra el contorno
            Point2f center;
            float radius;
            minEnclosingCircle(contour, center, radius);

            // Verificar si es lo suficientemente circular
            double circle_area = M_PI * radius * radius;
            double circularity = area / circle_area;

            if (circularity > 0.6) {
                circles.push_back(Vec3f(center.x, center.y, radius));
            }
        }

        cout << "Después de buscar por contornos: " << circles.size() << " círculos" << endl;
    }

    // PASO 4: CALIBRACIÓN Y CLASIFICACIÓN

    double px_per_mm = 1.0;

    if (!circles.empty()) {
        // Ordenar círculos por radio
        sort(circles.begin(), circles.end(),
             [](const Vec3f& a, const Vec3f& b) {
                 return a[2] < b[2];
             });

        // Usar los radios extremos para calibración
        double min_diameter = 2 * circles.front()[2];
        double max_diameter = 2 * circles.back()[2];

        // La moneda más pequeña es 1 Kč (20mm) y la más grande 20 Kč (26mm)
        double min_scale = min_diameter / 20.0;
        double max_scale = max_diameter / 26.0;

        // Promedio ponderado
        px_per_mm = (min_scale + max_scale) / 2.0;

        cout << "Calibración: " << px_per_mm << " píxeles por mm" << endl;
        cout << "Diámetro mínimo: " << min_diameter << " px (" << min_diameter/px_per_mm << " mm)" << endl;
        cout << "Diámetro máximo: " << max_diameter << " px (" << max_diameter/px_per_mm << " mm)" << endl;

        // Imprime información detallada sobre todos los diámetros detectados
        cout << "\nDiámetros detectados (mm):" << endl;
        for (const auto& circle : circles) {
            double diameter_mm = 2 * circle[2] / px_per_mm;
            cout << diameter_mm << " ";
        }
        cout << endl;
    }

    // PASO 5: CLASIFICACIÓN Y VISUALIZACIÓN

    int total_value = 0;
    map<int, int> coin_counts;

    Mat circles_img = img.clone();
    Mat contours_img = Mat::zeros(img.size(), CV_8UC3);

    // Dibujar la imagen binaria procesada como fondo para los contornos
    for (int y = 0; y < dilated.rows; y++) {
        for (int x = 0; x < dilated.cols; x++) {
            if (dilated.at<uchar>(y, x) > 0) {
                contours_img.at<Vec3b>(y, x) = Vec3b(255, 255, 255);
            }
        }
    }

    for (size_t i = 0; i < circles.size(); i++) {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);

        // Calcular diámetro en mm
        double diameter_mm = 2 * radius / px_per_mm;

        // Clasificar según el diámetro
        int value = classifyCoin(diameter_mm, coin_types);

        // Actualizar conteo y total
        coin_counts[value]++;
        total_value += value;

        // Obtener color para esta denominación
        Scalar color = Scalar(0, 0, 255);  // Default: rojo
        for (const auto& coin_info : coin_types) {
            if (coin_info.value == value) {
                color = coin_info.color;
                break;
            }
        }

        // Dibujar círculo en la imagen
        cv::circle(img_display, center, 3, Scalar(0, 255, 0), -1);
        cv::circle(img_display, center, radius, color, 2);

        // Etiquetar con el valor
        putText(img_display, to_string(value) + " Kc",
                Point(center.x - radius/2, center.y),
                FONT_HERSHEY_SIMPLEX, 0.5, color, 2);

        // Mostrar también el diámetro para verificación
        string diameter_text = to_string(int(diameter_mm * 10) / 10.0) + "mm";
        putText(img_display, diameter_text,
                Point(center.x - radius/2, center.y + 20),
                FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255), 1);

        // Dibujar también en la imagen de círculos
        cv::circle(circles_img, center, radius, color, 2);
        cv::circle(contours_img, center, radius, color, 2);
    }

    // Mostrar resultados
    cout << "\nMonedas detectadas:" << endl;
    for (const auto& entry : coin_counts) {
        cout << entry.second << " x " << entry.first << " Kc = "
             << entry.second * entry.first << " Kc" << endl;
    }
    cout << "\nValor total: " << total_value << " Kc" << endl;

    // Añadir texto con el total
    string total_text = "Total: " + to_string(total_value) + " Kc";
    putText(img_display, total_text, Point(30, 30),
            FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2);

    // Mostrar imágenes
    namedWindow("Original", WINDOW_NORMAL);
    imshow("Original", img);

    namedWindow("Corregida Gamma", WINDOW_NORMAL);
    imshow("Corregida Gamma", gamma_corrected);

    namedWindow("Threshold 50", WINDOW_NORMAL);
    imshow("Threshold 50", binary);

    namedWindow("Dilatada", WINDOW_NORMAL);
    imshow("Dilatada", dilated);

    namedWindow("Contornos y Círculos", WINDOW_NORMAL);
    imshow("Contornos y Círculos", contours_img);

    namedWindow("Resultado", WINDOW_NORMAL);
    imshow("Resultado", img_display);

    // Guardar resultados
    imwrite("threshold_50.jpg", binary);
    imwrite("contornos_circulos.jpg", contours_img);
    imwrite("circulos.jpg", circles_img);
    imwrite("resultado.jpg", img_display);

    waitKey(0);
    destroyAllWindows();

    return 0;
}
