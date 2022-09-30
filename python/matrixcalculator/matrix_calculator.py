"""
Matrix Calculator

Author: Sam Barba
Created 03/09/2021
"""

from matrix import Matrix

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def get_example(length):
	return ' '.join(map(str, range(1, length + 1)))

def main():
	while True:
		choice = input('Enter: A to add/subtract matrices,'
			+ '\n M to multiply matrices,'
			+ '\n D to divide matrices,'
			+ '\n P to do matrix to a power,'
			+ '\n R to convert a matrix to reduced row echelon form,'
			+ '\n I to find the matrix of cofactors, inverse & determinant of a matrix,'
			+ '\n G to do geometric transformations,'
			+ '\n or X to exit\n>>> ').upper()
		print()

		match choice:
			case 'A':
				rows, cols = map(int, input('Input the number of rows & columns for the matrices\n>>> ').split())

				mat_a_line = input(f'\nInput matrix A ({rows * cols} entries) e.g. {get_example(rows * cols)}\n>>> ')
				mat_b_line = input(f'Input matrix B ({rows * cols} entries) e.g. {get_example(rows * cols)}\n>>> ')

				mat_a = Matrix(mat_a_line, rows, cols)
				mat_b = Matrix(mat_b_line, rows, cols)

				print(f'\nA + B =\n{mat_a.add_subtract(mat_b, True)}')
				print(f'\nA - B =\n{mat_a.add_subtract(mat_b, False)}')
			case 'M':
				rows_a, cols_a = map(int, input('Input the number of rows & columns for matrix A\n>>> ').split())
				cols_b = int(input(f'Input the number of columns for matrix B (rows = {cols_a})\n>>> '))
				rows_b = cols_a

				mat_a_line = input(f'\nInput matrix A ({rows_a * cols_a} entries) e.g. {get_example(rows_a * cols_a)}\n>>> ')
				mat_b_line = input(f'Input matrix B ({rows_b * cols_b} entries) e.g. {get_example(rows_b * cols_b)}\n>>> ')

				mat_a = Matrix(mat_a_line, rows_a, cols_a)
				mat_b = Matrix(mat_b_line, rows_b, cols_b)

				print(f'\nA x B =\n{mat_a.mult(mat_b)}')
			case 'D':
				rows_a, cols_a = map(int, input('Input the number of rows & columns for matrix A\n>>> ').split())
				rows_b = cols_b = cols_a
				print(f'B is {rows_b} x {cols_b}')

				mat_a_line = input(f'\nInput matrix A ({rows_a * cols_a} entries) e.g. {get_example(rows_a * cols_a)}\n>>> ')
				mat_b_line = input(f'Input matrix B ({rows_b * cols_b} entries) e.g. {get_example(rows_b * cols_b)}\n>>> ')

				mat_a = Matrix(mat_a_line, rows_a, cols_a)
				mat_b = Matrix(mat_b_line, rows_b, cols_b)

				if mat_b.determinant(rows_b) == 0:
					print("\nCan't divide, determinant of B is 0")
				else:
					print(f'\nA / B =\n{mat_a.mult(mat_b.inverse())}')
			case 'P':
				size = int(input('Input the size of the square matrix\n>>> '))
				p = int(input('Input the power\n>>> '))

				mat_line = input(f'\nInput matrix M ({size ** 2} entries) e.g. {get_example(size ** 2)}\n>>> ')
				mat = Matrix(mat_line, size, size)

				print(f'\nM^{p} =\n{mat.power(p)}')
			case 'R':
				rows, cols = map(int, input('Input the number of rows & columns for the matrix\n>>> ').split())

				mat_line = input(f'\nInput matrix M ({rows * cols} entries) e.g. {get_example(rows * cols)}\n>>> ')
				mat = Matrix(mat_line, rows, cols)

				print(f'\nRREF(M) =\n{mat.rref()}')
			case 'I':
				size = int(input('Input the size of the square matrix\n>>> '))

				mat_line = input(f'\nInput matrix M ({size ** 2} entries) e.g. {get_example(size ** 2)}\n>>> ')
				mat = Matrix(mat_line, size, size)

				print(f'\nMatrix of cofactors:\n{mat.comatrix()}')

				det = mat.determinant(size)
				if abs(det) < 1e-6:
					print('\nNo inverse, determinant = 0')
				else:
					print(f'\nInverse:\n{mat.inverse()}')
					print('\nDeterminant =', det)
			case 'G':
				choice = int(input('Enter 1 for translation, 2 for enlargement, 3 for reflection, or 4 for rotation\n>>> '))

				n_verts = int(input('\nHow many vertices?\n>>> '))
				coords = [None] * n_verts

				for i in range(n_verts):
					coords[i] = list(map(float, input(f'Input x and y coords {i + 1}/{n_verts} e.g. {get_example(2)}\n>>> ').split()))

				coords = Matrix(coords)

				match choice:
					case 1:
						dx, dy = map(float, input(f'Input the change in x and change in y e.g. {get_example(2)}\n>>> ').split())
						print(f'\nResultant coords:\n{coords.translate(dx, dy)}')
					case 2:
						k, x, y = map(float, input(f'Input the enlargement factor and x, y e.g. 2 3 4\n>>> ').split())
						print(f'\nResultant coords:\n{coords.enlarge(k, x, y)}')
					case 3:
						m, c = map(float, input(f'Input m, c for reflection line y = mx + c e.g. 2 3\n>>> ').split())
						print(f'\nResultant coords:\n{coords.reflect(m, c)}')
					case 4:
						a, x, y = map(float, input(f'Input the clockwise rotation angle (deg) and x, y e.g. 90 0 2\n>>> ').split())
						print(f'\nResultant coords:\n{coords.rotate(a, x, y)}')
			case 'X':
				break

		print()

if __name__ == '__main__':
	main()
