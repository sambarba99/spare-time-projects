# To add to all C++ SFML projects (top level dir).
# Use: in the terminal, run "make" to create a .exe, then ".\main" to run it. Run "make clean" to clean up.

all: compile link

compile:
	g++ -c main.cpp -IC:\SFML-2.5.1\include -DSFML_STATIC

link:
	g++ main.o -o main -LC:\SFML-2.5.1\lib -lsfml-graphics-s -lsfml-window-s -lsfml-system-s -lopengl32 -lfreetype -lwinmm -lgdi32 -lsfml-main

clean:
	rm -f main *.o
