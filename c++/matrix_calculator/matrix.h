#ifndef MATRIX
#define MATRIX

#include <cmath>
#include <vector>

using std::vector;


class Matrix {
	public:
		vector<vector<long double>> grid;
		int rows;
		int cols;

		Matrix(const vector<vector<long double>>& grid) {
			this->grid = grid;
			rows = grid.size();
			cols = grid[0].size();
		}

		Matrix add_subtract(const Matrix& other, const bool is_add) {
			vector<vector<long double>> result(other.rows, vector<long double> (other.cols));

			for (int i = 0; i < rows; i++)
				for (int j = 0; j < cols; j++)
					if (is_add)
						result[i][j] = grid[i][j] + other.grid[i][j];
					else
						result[i][j] = grid[i][j] - other.grid[i][j];

			return Matrix(result);
		}

		Matrix mult(const Matrix& other) {
			vector<vector<long double>> result(rows, vector<long double> (other.cols));

			for (int i = 0; i < rows; i++)
				for (int j = 0; j < other.cols; j++)
					for (int k = 0; k < cols; k++)
						result[i][j] += grid[i][k] * other.grid[k][j];

			return Matrix(result);
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
				det += sign * grid[0][i] * Matrix(temp).determinant(n - 1);
			}

			return det;
		}

		Matrix remove_row_and_col(const Matrix& mat, const int row, const int col) {
			vector<vector<long double>> sub_grid(mat.rows - 1, vector<long double>(mat.cols - 1));
			int sub_row = 0, sub_col = 0;

			for (int i = 0; i < mat.rows; i++) {
				if (i == row)
					continue;
				for (int j = 0; j < mat.cols; j++) {
					if (j == col)
						continue;
					sub_grid[sub_row][sub_col] = mat.grid[i][j];
					sub_col = (sub_col + 1) % sub_grid.size();
					if (sub_col == 0)
						sub_row++;
				}
			}

			return Matrix(sub_grid);
		}

		Matrix comatrix() {
			vector<vector<long double>> result(rows, vector<long double>(cols));

			for (int i = 0; i < rows; i++)
				for (int j = 0; j < cols; j++) {
					Matrix submatrix = remove_row_and_col(*this, i, j);
					int sign = (i + j) % 2 == 0 ? 1 : -1;
					result[i][j] = sign * submatrix.determinant();
				}

			return Matrix(result);
		}

		Matrix inverse() {
			/*
			Inverse of a square matrix = 1/determinant * adjugate matrix
			= 1/determinant * transposed cofactor matrix
			*/

			long double det = determinant();
			Matrix comat = comatrix();
			vector<vector<long double>> adjugate_grid(rows, vector<long double> (cols));
			for (int i = 0; i < rows; i++)
				for (int j = 0; j < cols; j++)
					adjugate_grid[j][i] = comat.grid[i][j];  // Transpose

			vector<vector<long double>> result(rows, vector<long double> (cols));
			for (int i = 0; i < rows; i++)
				for (int j = 0; j < cols; j++)
					result[i][j] = adjugate_grid[i][j] / det;

			return Matrix(result);
		}

		Matrix power(int p) {
			Matrix m(grid);
			if (p < 0) {
				m = m.inverse();
				p = -p;
			}

			Matrix result(m.grid);
			for (int i = 0; i < p - 1; i++)
				result = result.mult(m);

			return result;
		}

		Matrix rref() {
			vector<vector<long double>> rref_grid = grid;
			int pivot_row, pivot_col = 0;

			for (int i = 0; i < rows; i++) {
				// 1. Find left-most nonzero entry (pivot)
				pivot_row = i;
				while (rref_grid[pivot_row][pivot_col] == 0) {
					pivot_row++;
					if (pivot_row == rows) {
						// Back to initial row but next column
						pivot_row == i;
						pivot_col++;
						if (pivot_col == cols)  // No pivot
							return Matrix(rref_grid);
					}
				}

				// 2. If pivot row is below current row, swap these rows (push 0s to bottom)
				if (pivot_row > i) {
					vector<long double> temp = rref_grid[pivot_row];
					rref_grid[pivot_row] = rref_grid[i];
					rref_grid[i] = temp;
				}

				// 3. Scale current row such that pivot becomes 1
				long double scale = rref_grid[i][pivot_col];
				vector<long double> scaled_row(cols);
				for (int j = 0; j < cols; j++)
					scaled_row[j] = rref_grid[i][j] / scale;
				rref_grid[i] = scaled_row;

				// 4. Make entries above/below equal 0
				for (int r = 0; r < rows; r++) {
					scale = rref_grid[r][pivot_col];
					if (r != i) {
						vector<long double> new_row(cols);
						for (int j = 0; j < cols; j++)
							new_row[j] = rref_grid[r][j] - scale * rref_grid[i][j];
						rref_grid[r] = new_row;
					}
				}

				// 5. Move to next col
				pivot_col++;
				if (pivot_col == cols) break;
			}

			return Matrix(rref_grid);
		}

		// -------------------- Geometric transformations --------------------

		Matrix translate(const long double dx, const long double dy) {
			vector<vector<long double>> result = grid;
			for (int i = 0; i < rows; i++) {
				result[i][0] += dx;
				result[i][1] += dy;
			}
			return Matrix(result);
		}

		Matrix enlarge(const long double k, const long double x, const long double y) {
			Matrix temp(grid);
			if (x != 0 || y != 0)
				temp = temp.translate(-x, -y);  // Enlarge from origin (0,0)

			Matrix enlarge_matrix({
				{k, 0},
				{0, k}
			});
			Matrix result = temp.mult(enlarge_matrix);

			if (x != 0 || y != 0)
				return result.translate(x, y);  // Undo first translation if necessary
			else
				return result;
		}

		Matrix reflect(const long double m, const long double c) {
			Matrix temp(grid);
			if (c != 0)
				temp = temp.translate(0, -c);  // Reflect in y = mx

			long double r = 1 / (1 + m * m);
			Matrix reflect_matrix({
				{r * (1 - m * m), r * 2 * m},
				{r * 2 * m, r * (m * m - 1)}
			});
			Matrix result = temp.mult(reflect_matrix);

			if (c != 0)
				return result.translate(0, c);  // Undo first translation if necessary
			else
				return result;
		}

		Matrix rotate(const long theta, const long x, const long y) {
			// Rotate by theta (deg) clockwise about (x,y)
			Matrix temp(grid);
			if (x != 0 || y != 0)
				temp = temp.translate(-x, -y);  // Rotate about origin

			long double theta_rad = theta * M_PI / 180;
			long double sin_theta = sin(theta_rad), cos_theta = cos(theta_rad);
			Matrix rotate_matrix({
				{cos_theta, -sin_theta},
				{sin_theta, cos_theta}
			});
			Matrix result = temp.mult(rotate_matrix);

			if (x != 0 || y != 0)
				return result.translate(x, y);  // Undo first translation if necessary
			else
				return result;
		}

		std::string to_string() {
			std::string s = "";
			for (vector<long double> row : grid) {
				for (long double n : row)
					s += std::to_string(n) + ' ';
				s += '\n';
			}
			return s;
		}
};

#endif
