#!/bin/bash

# Script para compilar y ejecutar los diferentes algoritmos de detección de características

# Comprobar si OpenCV está instalado
if ! pkg-config --exists opencv4; then
    echo "Error: OpenCV 4 no está instalado o no se encuentra en pkg-config"
    echo "Instala OpenCV con: sudo apt-get install libopencv-dev"
    exit 1
fi

# Función para compilar y ejecutar un algoritmo
compile_and_run() {
    algorithm=$1
    echo "===========================================" 
    echo "Compilando $algorithm.cpp..."
    
    g++ -std=c++11 -O3 $algorithm.cpp -o $algorithm `pkg-config --cflags --libs opencv4`
    
    if [ $? -eq 0 ]; then
        echo "Compilación exitosa. Ejecutando $algorithm..."
        ./$algorithm
        echo "Ejecución completada para $algorithm"
    else
        echo "Error al compilar $algorithm.cpp"
    fi
    echo "===========================================" 
    echo ""
}

# Verificar y crear el directorio de resultados
mkdir -p results

# Compilar y ejecutar cada algoritmo
algorithms=("sift_sift" "surf_surf" "orb_orb" "fast_brief" "brisk_brisk")

echo "Selecciona un algoritmo para ejecutar:"
echo "1. SIFT + SIFT + BF"
echo "2. SURF + SURF + BF"
echo "3. ORB + ORB + BF"
echo "4. FAST + BRIEF + BF"
echo "5. BRISK + BRISK + BF"
echo "6. Todos los algoritmos (uno tras otro)"
echo ""
read -p "Ingresa tu elección (1-6): " choice

case $choice in
    1) compile_and_run "sift_sift" ;;
    2) compile_and_run "surf_surf" ;;
    3) compile_and_run "orb_orb" ;;
    4) compile_and_run "fast_brief" ;;
    5) compile_and_run "brisk_brisk" ;;
    6) 
        for algo in "${algorithms[@]}"; do
            compile_and_run "$algo"
            sleep 1  # Dar tiempo para liberar recursos
        done
        ;;
    *) echo "Opción inválida" ;;
esac

echo "Proceso completo. Los resultados están disponibles en archivos JPG."
