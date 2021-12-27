# Sudoku solver using depth-first search
# Author: Sam Barba
# Created 07/09/2021

SIZE = 9

# Most difficult sudoku
BOARD_DIFFICULT = "800000000003600000070090200050007000000045700000100030001000068008500010090000400"

solved = False
board = [[0] * SIZE for _ in range(SIZE)]

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def solve():
	global solved
	if isFull():
		solved = True
	else:
		x, y = findFreeSquare()
		for n in range(1, 10):
			if legal(n, x, y):
				board[x][y] = n
				solve()

		# If we're here, no numbers were legal
		# So the previous attempt in the loop must be invalid
		# So we reset the square in order to backtrack, so next number is tried
		if not solved:
			board[x][y] = 0

def isFull():
	return all(n != 0 for row in board for n in row)

def findFreeSquare():
	for x in range(SIZE):
		for y in range(SIZE):
			if board[x][y] == 0:
				return x, y

	return -1, -1

def valid():
	return all(all(legal(board[x][y], x, y) for x in range(SIZE)) for y in range(SIZE))

def legal(n, x, y):
	if n == 0: return True

	bigSquareX = x - (x % 3) # Smallest coords of big square
	bigSquareY = y - (y % 3)

	# Check big square
	for checkX in range(bigSquareX, bigSquareX + 3):
		for checkY in range(bigSquareY, bigSquareY + 3):
			if board[checkX][checkY] == n and not (checkX == x and checkY == y):
				return False

	# Check column and row
	for i in range(9):
		if board[x][i] == n and i != y: return False
		if board[i][y] == n and i != x: return False

	return True

def formatBoard():
	s = ""
	for i in range(SIZE):
		if i in [3, 6]:
			s += " ------+-------+------\n"
		for j in range(SIZE):
			if j in [3, 6]:
				s += " |"
			s += " " + str(board[i][j])
		s += "\n"
	return s

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

choice = input("Enter 1 to use 'all zeroes' sudoku, 2 to use most difficult sudoku: ")

if choice == "2":
	for idx, n in enumerate(BOARD_DIFFICULT):
		row, col = idx // SIZE, idx % SIZE
		board[row][col] = int(n)

solve()

print("\nSolved:\n")
print(formatBoard())
