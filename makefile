# To add to all C++ projects that use OpenCV or SFML (top level dir).
# Use: in the terminal, run "make" to create a .exe, then ".\main" to run it. Run "make clean" to clean up.

all: compile link

compile:
	g++ -c main.cpp \
	-IC:/msys64/mingw64/include/opencv4 \
	-DSFML_STATIC

link:
	g++ main.o -o main \
	-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs \
	-lsfml-graphics-s -lsfml-window-s -lsfml-system-s -lopengl32 -lfreetype -lwinmm -lgdi32 -lsfml-main

clean:
	rm -f main *.o
