# To add to all C++ projects that use OpenCV or SFML (top level dir).
# Use: in the terminal, run "make" to create a .exe, then ".\main" to run it. Run "make clean" to clean up.

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra $(shell pkg-config --cflags opencv4 sfml-graphics sfml-window sfml-system)
LIBS = $(shell pkg-config --libs opencv4 sfml-graphics sfml-window sfml-system)

all: main

main: main.o
	$(CXX) main.o -o main $(LIBS)

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp

clean:
	rm -f main *.o
