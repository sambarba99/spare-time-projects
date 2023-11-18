#ifndef MATRIX
#define MATRIX

#include <cmath>
#include <string>
#include <vector>

using std::string;
using std::to_string;
using std::vector;

class Matrix {
	public:
		vector<vector<long double>> grid;
		int rows;
		int cols;

		Matrix(const vector<vector<long double>> grid) {
			this->grid = grid;
			rows = grid.size();
			cols = grid[0].size();
		}

		Matrix* addSubtract(const Matrix* other, const bool isAdd) {
			vector<vector<long double>> result(other->rows, vector<long double> (other->cols));

			for (int i = 0; i < rows; i++)
				for (int j = 0; j < cols; j++)
					if (isAdd) result[i][j] = grid[i][j] + other->grid[i][j];
					else result[i][j] = grid[i][j] - other->grid[i][j];

			return new Matrix(result);
		}

		Matrix* mult(const Matrix* other) {
			vector<vector<long double>> result(rows, vector<long double> (other->cols));

			for (int i = 0; i < rows; i++)
				for (int j = 0; j < other->cols; j++)
					for (int k = 0; k < cols; k++)
						result[i][j] += grid[i][k] * other->grid[k][j];

			return new Matrix(result);
		}

		long double determinant(int n = -1) {
			if (n == -1) n = rows;

			long double det = 0;
			vector<vector<long double>> temp(rows, vector<long double>(rows));

			if (n == 1) return grid[0][0];

			int a, b, sign;
			for (int i = 0; i < n; i++) {
				a = 0;
				b = 0;
				for (int j = 1; j < n; j++) {
					for (int k = 0; k < n; k++) {
						if (k == i) continue;
						temp[a][b] = grid[j][k];
						b++;
						if (b == n - 1) {
							a++;
							b = 0;
						}
					}
				}
				sign = i % 2 == 0 ? 1 : -1;
				Matrix* tempMat = new Matrix(temp);
				det += sign * grid[0][i] * tempMat->determinant(n - 1);
			}

			return det;
		}

		Matrix* removeRowAndCol(const Matrix* mat, const int row, const int col) {
			vector<vector<long double>> subGrid(mat->rows - 1, vector<long double>(mat->cols - 1));
			int subRow = 0, subCol = 0;

			for (int i = 0; i < mat->rows; i++) {
				if (i == row) continue;
				for (int j = 0; j < mat->cols; j++) {
					if (j == col) continue;
					subGrid[subRow][subCol] = mat->grid[i][j];
					subCol = (subCol + 1) % subGrid.size();
					if (subCol == 0) subRow++;
				}
			}

			return new Matrix(subGrid);
		}

		Matrix* comatrix() {
			vector<vector<long double>> result(rows, vector<long double>(cols));

			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					Matrix* submatrix = this->removeRowAndCol(this, i, j);
					int sign = (i + j) % 2 == 0 ? 1 : -1;
					result[i][j] = sign * submatrix->determinant();
				}
			}

			return new Matrix(result);
		}

		Matrix* inverse() {
			/*
			Inverse of a square matrix = 1/determinant * adjugate matrix
			= 1/determinant * transposed cofactor matrix
			*/

			long double det = this->determinant();
			Matrix* comatrix = this->comatrix();
			vector<vector<long double>> adjugateGrid(rows, vector<long double> (cols));
			for (int i = 0; i < rows; i++)
				for (int j = 0; j < cols; j++)
					adjugateGrid[j][i] = comatrix->grid[i][j];  // Transpose

			vector<vector<long double>> result(rows, vector<long double> (cols));
			for (int i = 0; i < rows; i++)
				for (int j = 0; j < cols; j++)
					result[i][j] = adjugateGrid[i][j] / det;

			return new Matrix(result);
		}

		Matrix* power(int p) {
			Matrix* m = new Matrix(this->grid);
			if (p < 0) {
				m = m->inverse();
				p = -p;
			}

			Matrix* result = new Matrix(m->grid);
			for (int i = 0; i < p - 1; i++)
				result = result->mult(m);

			return result;
		}

		Matrix* rref() {
			vector<vector<long double>> rrefGrid = grid;
			int pivotRow, pivotCol = 0;

			for (int i = 0; i < rows; i++) {
				// 1. Find left-most nonzero entry (pivot)
				pivotRow = i;
				while (rrefGrid[pivotRow][pivotCol] == 0) {
					pivotRow++;
					if (pivotRow == rows) {
						// Back to initial row but next column
						pivotRow == i;
						pivotCol++;
						if (pivotCol == cols)  // No pivot
							return new Matrix(rrefGrid);
					}
				}

				// 2. If pivot row is below current row, swap these rows (push 0s to bottom)
				if (pivotRow > i) {
					vector<long double> temp = rrefGrid[pivotRow];
					rrefGrid[pivotRow] = rrefGrid[i];
					rrefGrid[i] = temp;
				}

				// 3. Scale current row such that pivot becomes 1
				long double scale = rrefGrid[i][pivotCol];
				vector<long double> scaledRow(cols);
				for (int j = 0; j < cols; j++)
					scaledRow[j] = rrefGrid[i][j] / scale;
				rrefGrid[i] = scaledRow;

				// 4. Make entries above/below equal 0
				for (int r = 0; r < rows; r++) {
					scale = rrefGrid[r][pivotCol];
					if (r != i) {
						vector<long double> newRow(cols);
						for (int j = 0; j < cols; j++)
							newRow[j] = rrefGrid[r][j] - scale * rrefGrid[i][j];
						rrefGrid[r] = newRow;
					}
				}

				// 5. Move to next col
				pivotCol++;
				if (pivotCol == cols) break;
			}

			return new Matrix(rrefGrid);
		}

		// -------------------- Geometric transformations --------------------

		Matrix* translate(const long double dx, const long double dy) {
			vector<vector<long double>> result = grid;
			for (int i = 0; i < rows; i++) {
				result[i][0] += dx;
				result[i][1] += dy;
			}
			return new Matrix(result);
		}

		Matrix* enlarge(const long double k, const long double x, const long double y) {
			Matrix* temp = new Matrix(grid);
			if (x != 0 || y != 0)
				temp = temp->translate(-x, -y);  // Enlarge from origin (0,0)

			vector<vector<long double>> enlargeGrid = {{k, 0}, {0, k}};
			Matrix* enlargeMatrix = new Matrix(enlargeGrid);
			Matrix* result = temp->mult(enlargeMatrix);

			if (x != 0 || y != 0)
				return result->translate(x, y);  // Undo first translation if necessary
			else
				return result;
		}

		Matrix* reflect(const long double m, const long double c) {
			Matrix* temp = new Matrix(grid);
			if (c != 0)
				temp = temp->translate(0, -c);  // Reflect in y = mx

			long double r = 1 / (1 + m * m);
			vector<vector<long double>> reflectGrid = {
				{r * (1 - m * m), r * 2 * m},
				{r * 2 * m, r * (m * m - 1)}
			};
			Matrix* reflectMatrix = new Matrix(reflectGrid);
			Matrix* result = temp->mult(reflectMatrix);

			if (c != 0)
				return result->translate(0, c);  // Undo first translation if necessary
			else
				return result;
		}

		Matrix* rotate(const long theta, const long x, const long y) {
			// Rotate by theta (deg) clockwise about (x,y)
			Matrix* temp = new Matrix(grid);
			if (x != 0 || y != 0)
				temp = temp->translate(-x, -y);  // Rotate about origin

			long double thetaRad = theta * M_PI / 180;
			long double sinTheta = sin(thetaRad), cosTheta = cos(thetaRad);
			vector<vector<long double>> rotateGrid = {{cosTheta, -sinTheta}, {sinTheta, cosTheta}};
			Matrix* rotateMatrix = new Matrix(rotateGrid);
			Matrix* result = temp->mult(rotateMatrix);

			if (x != 0 || y != 0)
				return result->translate(x, y);  // Undo first translation if necessary
			else
				return result;
		}

		string toString() {
			string s = "";
			for (vector<long double> row : grid) {
				for (long double n : row)
					s += to_string(n) + ' ';
				s += '\n';
			}
			return s;
		}
};

#endif
