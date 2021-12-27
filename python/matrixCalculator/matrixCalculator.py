# Matrix Calculator
# Author: Sam Barba
# Created 03/09/2021

from matrix import Matrix

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

while True:
	choice = input("Enter: A to add/subtract matrices,"
		+ "\n M to multiply matrices,"
		+ "\n D to divide matrices,"
		+ "\n P to do matrix to a power,"
		+ "\n R to convert a matrix to reduced row echelon form,"
		+ "\n I to find the matrix of cofactors, inverse & determinant of a matrix,"
		+ "\n G to do geometric transformations,"
		+ "\n or X to exit: ").upper()
	print()

	if choice == "A":
		rows, cols = map(int, input("Input the number of rows & columns for the matrices: ").split())

		matAline = input(f"\nInput matrix A ({rows * cols} entries): ")
		matBline = input(f"Input matrix B ({rows * cols} entries): ")

		matA = Matrix(matAline, rows, cols)
		matB = Matrix(matBline, rows, cols)

		print("\nA + B =\n" + str(matA.addSubtract(matB, True)))
		print("\nA - B =\n" + str(matA.addSubtract(matB, False)))

	elif choice == "M":
		rowsA, colsA = map(int, input("Input the number of rows & columns for matrix A: ").split())
		colsB = int(input(f"Input the number of columns for matrix B (rows = {colsA}): "))
		rowsB = colsA

		matAline = input(f"\nInput matrix A ({rowsA * colsA} entries): ")
		matBline = input(f"Input matrix B ({rowsB * colsB} entries): ")

		matA = Matrix(matAline, rowsA, colsA)
		matB = Matrix(matBline, rowsB, colsB)

		print("\nA x B =\n" + str(matA.mult(matB)))

	elif choice == "D":
		rowsA, colsA = map(int, input("Input the number of rows & columns for matrix A: ").split())
		rowsB = colsB = colsA
		print(f"B is {rowsB} x {colsB}")

		matAline = input(f"\nInput matrix A ({rowsA * colsA} entries): ")
		matBline = input(f"Input matrix B ({rowsB * colsB} entries): ")

		matA = Matrix(matAline, rowsA, colsA)
		matB = Matrix(matBline, rowsB, colsB)

		if matB.determinant(rowsB) == 0:
			print("\nCan't divide, determinant of B is 0")
		else:
			print("\nA / B =\n" + str(matA.mult(matB.inverse())))

	elif choice == "P":
		size = int(input("Input the size of the square matrix: "))
		p = int(input("Input the power: "))

		matLine = input(f"\nInput matrix M ({size ** 2} entries): ")
		mat = Matrix(matLine, size, size)

		print(f"\nM^{p} =\n{str(mat.power(p))}")

	elif choice == "R":
		rows, cols = map(int, input("Input the number of rows & columns for the matrix: ").split())

		matLine = input(f"\nInput matrix M ({rows * cols} entries): ")
		mat = Matrix(matLine, rows, cols)

		print("\nRREF(M) =\n" + str(mat.rref()))

	elif choice == "I":
		size = int(input("Input the size of the square matrix: "))

		matLine = input(f"\nInput matrix M ({size ** 2} entries): ")
		mat = Matrix(matLine, size, size)

		print("\nMatrix of cofactors:\n" + str(mat.comatrix()))

		det = mat.determinant(size)
		if det == 0:
			print("\nNo inverse, determinant = 0")
		else:
			print("\nInverse:\n" + str(mat.inverse()))
			print("\nDeterminant =", det)

	elif choice == "G":
		choice = int(input("Enter 1 for translation, 2 for enlargement, 3 for reflection or 4 for rotation: "))

		numV = int(input("\nHow many vertices? "))
		coords = [[0] * 2] * numV

		for i in range(numV):
			coords[i] = list(map(float, input(f"Input x and y coords {i + 1} / {numV}: ").split()))

		coords = Matrix(coords)

		if choice == 1:
			dx, dy = map(float, input("Input the change in x and change in y: ").split())
			print("\nResultant coords:\n" + str(coords.translate(dx, dy)))
		elif choice == 2:
			k, x, y = map(float, input("Input the enlargement factor and x, y: ").split())
			print("\nResultant coords:\n" + str(coords.enlarge(k, x, y)))
		elif choice == 3:
			m, c = map(float, input("Input m, c for reflection line y = mx + c: ").split())
			print("\nResultant coords:\n" + str(coords.reflect(m, c)))
		elif choice == 4:
			a, x, y = map(float, input("Input the clockwise rotation angle (°) and x, y: ").split())
			print("\nResultant coords:\n" + str(coords.rotate(a, x, y)))

	elif choice == "X":
		break

	print()
