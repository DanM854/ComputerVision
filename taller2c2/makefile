# Makefile para el proyecto de análisis de algoritmos de características

# Compilador y flags
CXX = g++
CXXFLAGS = -std=c++11 -O3 -Wall
OPENCV = `pkg-config --cflags --libs opencv4`

# Archivos fuente y ejecutables
INDIVIDUAL_SOURCES = sift_sift.cpp surf_surf.cpp orb_orb.cpp fast_brief.cpp brisk_brisk.cpp
INDIVIDUAL_BINARIES = $(INDIVIDUAL_SOURCES:.cpp=)

# Programa principal para todas las combinaciones
TESTER = combination_tester
TESTER_SRC = combination_tester.cpp

# Objetivo principal
all: $(INDIVIDUAL_BINARIES) $(TESTER)

# Regla para compilar los programas individuales
%: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(OPENCV)

# Compilar el tester de combinaciones
$(TESTER): $(TESTER_SRC)
	$(CXX) $(CXXFLAGS) $< -o $@ $(OPENCV)

# Crear carpeta para resultados
results:
	mkdir -p results

# Limpiar archivos generados
clean:
	rm -f $(INDIVIDUAL_BINARIES) $(TESTER)
	rm -f result_*.jpg

# Ejecutar el tester de combinaciones
run_tester: $(TESTER) results
	./$(TESTER)

# Ejecutar un algoritmo específico
run_sift: sift_sift results
	./sift_sift

run_surf: surf_surf results
	./surf_surf

run_orb: orb_orb results
	./orb_orb

run_fast_brief: fast_brief results
	./fast_brief

run_brisk: brisk_brisk results
	./brisk_brisk

# Menú interactivo
menu:
	@echo "Selecciona un algoritmo para ejecutar:"
	@echo "1. SIFT + SIFT + BF"
	@echo "2. SURF + SURF + BF"
	@echo "3. ORB + ORB + BF"
	@echo "4. FAST + BRIEF + BF"
	@echo "5. BRISK + BRISK + BF"
	@echo "6. Probar todas las combinaciones"
	@echo "7. Limpiar archivos generados"
	@read -p "Ingresa tu elección (1-7): " choice; \
	case $$choice in \
		1) make run_sift ;; \
		2) make run_surf ;; \
		3) make run_orb ;; \
		4) make run_fast_brief ;; \
		5) make run_brisk ;; \
		6) make run_tester ;; \
		7) make clean ;; \
		*) echo "Opción inválida" ;; \
	esac

.PHONY: all clean results menu run_tester run_sift run_surf run_orb run_fast_brief run_brisk
