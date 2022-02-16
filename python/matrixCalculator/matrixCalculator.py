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

		mat_a_line = input(f"\nInput matrix A ({rows * cols} entries): ")
		mat_b_line = input(f"Input matrix B ({rows * cols} entries): ")

		mat_a = Matrix(mat_a_line, rows, cols)
		mat_b = Matrix(mat_b_line, rows, cols)

		print("\nA + B =\n" + str(mat_a.add_subtract(mat_b, True)))
		print("\nA - B =\n" + str(mat_a.add_subtract(mat_b, False)))

	elif choice == "M":
		rows_a, cols_a = map(int, input("Input the number of rows & columns for matrix A: ").split())
		cols_b = int(input(f"Input the number of columns for matrix B (rows = {cols_a}): "))
		rows_b = cols_a

		mat_a_line = input(f"\nInput matrix A ({rows_a * cols_a} entries): ")
		mat_b_line = input(f"Input matrix B ({rows_b * cols_b} entries): ")

		mat_a = Matrix(mat_a_line, rows_a, cols_a)
		mat_b = Matrix(mat_b_line, rows_b, cols_b)

		print("\nA x B =\n" + str(mat_a.mult(mat_b)))

	elif choice == "D":
		rows_a, cols_a = map(int, input("Input the number of rows & columns for matrix A: ").split())
		rows_b = cols_b = cols_a
		print(f"B is {rows_b} x {cols_b}")

		mat_a_line = input(f"\nInput matrix A ({rows_a * cols_a} entries): ")
		mat_b_line = input(f"Input matrix B ({rows_b * cols_b} entries): ")

		mat_a = Matrix(mat_a_line, rows_a, cols_a)
		mat_b = Matrix(mat_b_line, rows_b, cols_b)

		if mat_b.determinant(rows_b) == 0:
			print("\nCan't divide, determinant of B is 0")
		else:
			print("\nA / B =\n" + str(mat_a.mult(mat_b.inverse())))

	elif choice == "P":
		size = int(input("Input the size of the square matrix: "))
		p = int(input("Input the power: "))

		mat_line = input(f"\nInput matrix M ({size ** 2} entries): ")
		mat = Matrix(mat_line, size, size)

		print(f"\nM^{p} =\n{str(mat.power(p))}")

	elif choice == "R":
		rows, cols = map(int, input("Input the number of rows & columns for the matrix: ").split())

		mat_line = input(f"\nInput matrix M ({rows * cols} entries): ")
		mat = Matrix(mat_line, rows, cols)

		print("\nRREF(M) =\n" + str(mat.rref()))

	elif choice == "I":
		size = int(input("Input the size of the square matrix: "))

		mat_line = input(f"\nInput matrix M ({size ** 2} entries): ")
		mat = Matrix(mat_line, size, size)

		print("\nMatrix of cofactors:\n" + str(mat.comatrix()))

		det = mat.determinant(size)
		if abs(det) < 10 ** -6:
			print("\nNo inverse, determinant = 0")
		else:
			print("\nInverse:\n" + str(mat.inverse()))
			print("\nDeterminant =", det)

	elif choice == "G":
		choice = int(input("Enter 1 for translation, 2 for enlargement, 3 for reflection or 4 for rotation: "))

		num_v = int(input("\nHow many vertices? "))
		coords = [None] * num_v

		for i in range(num_v):
			coords[i] = list(map(float, input(f"Input x and y coords {i + 1} / {num_v}: ").split()))

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
