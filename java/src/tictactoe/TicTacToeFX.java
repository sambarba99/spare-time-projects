package tictactoe;

import javafx.application.Application;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Pane;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.shape.Circle;
import javafx.scene.shape.Line;
import javafx.stage.Stage;

import utils.BoxType;
import utils.PaneBuilder;

/**
 * Tic Tac Toe player using minimax
 * 
 * @author Sam Barba
 */
public class TicTacToeFX extends Application {

	private static final int SIZE = 3;

	private static final char AI = 'X';

	private static final char HUMAN = 'O';

	private static final char TIE = 'T';

	private static final char NONE = 'N';

	private static final char EMPTY = ' ';

	private Label lblStatus = new Label("Human's turn (O)");

	private Button btnAiFirst = new Button("AI first?");

	private Cell[][] board = new Cell[SIZE][SIZE];

	@Override
	public void start(Stage primaryStage) {
		initialiseGame();

		lblStatus.setStyle("-fx-font-size: 20px");

		Button btnReset = new Button("Reset");
		btnReset.setOnAction(action -> start(primaryStage));
		btnAiFirst.setOnAction(action -> {
			makeBestAiMove();
			btnAiFirst.setDisable(true);
		});

		HBox hboxBoard = (HBox) new PaneBuilder(BoxType.HBOX)
			.withAlignment(Pos.CENTER)
			.withSpacing(10)
			.build();

		for (int x = 0; x < SIZE; x++) {
			VBox vboxRow = (VBox) new PaneBuilder(BoxType.VBOX)
				.withAlignment(Pos.CENTER)
				.withSpacing(10)
				.build();

			for (int y = 0; y < SIZE; y++) {
				vboxRow.getChildren().add(board[x][y]);
			}
			hboxBoard.getChildren().add(vboxRow);
		}

		VBox root = (VBox) new PaneBuilder(BoxType.VBOX)
			.withAlignment(Pos.CENTER)
			.withSpacing(10)
			.withNodes(hboxBoard, lblStatus, btnAiFirst, btnReset)
			.build();
		root.getStyleClass().add("root-no-gradient");

		Scene scene = new Scene(root, 400, 500);
		scene.getStylesheets().add("style.css");
		primaryStage.setScene(scene);
		primaryStage.setTitle("Tic Tac Toe");
		primaryStage.setResizable(false);
		primaryStage.show();
	}

	private void initialiseGame() {
		for (int x = 0; x < SIZE; x++) {
			for (int y = 0; y < SIZE; y++) {
				board[x][y] = new Cell();
			}
		}
		btnAiFirst.setDisable(false);
		lblStatus.setText("Human's turn (O)");
	}

	private void makeBestAiMove() {
		double bestScore = -Double.MAX_VALUE;
		int bestX = 0, bestY = 0;

		for (int x = 0; x < SIZE; x++) {
			for (int y = 0; y < SIZE; y++) {
				if (board[x][y].getToken() == EMPTY) {
					board[x][y].setToken(AI, false);
					double score = minimax(1, false);
					board[x][y].setToken(EMPTY, false);

					if (score > bestScore) {
						bestScore = score;
						bestX = x;
						bestY = y;
					}
				}
			}
		}

		if (findWinner() == NONE) {
			board[bestX][bestY].setToken(AI, true);
		}
		lblStatus.setText("Human's turn (O)");
	}

	private double minimax(int depth, boolean maximising) {
		char result = findWinner();
		if (result == AI) {
			return 1;
		} else if (result == HUMAN) {
			return -1;
		} else if (result == TIE) {
			return 0;
		}

		double bestScore = maximising ? -Double.MAX_VALUE : Double.MAX_VALUE;

		for (int x = 0; x < SIZE; x++) {
			for (int y = 0; y < SIZE; y++) {
				if (board[x][y].getToken() == EMPTY) {
					if (maximising) {
						board[x][y].setToken(AI, false);
						bestScore = Math.max(minimax(depth + 1, false), bestScore);
					} else {
						board[x][y].setToken(HUMAN, false);
						bestScore = Math.min(minimax(depth + 1, true), bestScore);
					}
					board[x][y].setToken(EMPTY, false);
				}
			}
		}
		return bestScore / (double) depth; // The deeper we have to search, the worse the score
	}

	private char findWinner() {
		int freeSpots = 0;
		int sumRowsAI = 0, sumColsAI = 0, sumDiagsAI = 0, sumRdiagsAI = 0;
		int sumRowsHuman = 0, sumColsHuman = 0, sumDiagsHuman = 0, sumRdiagsHuman = 0;

		for (int x = 0; x < SIZE; x++) {
			for (int y = 0; y < SIZE; y++) {
				// Check columns
				if (board[x][y].getToken() == AI) {
					sumColsAI++;
				} else if (board[x][y].getToken() == HUMAN) {
					sumColsHuman--;
				}

				// Check rows
				if (board[y][x].getToken() == AI) {
					sumRowsAI++;
				} else if (board[y][x].getToken() == HUMAN) {
					sumRowsHuman--;
				}

				if (board[x][y].getToken() == EMPTY) {
					freeSpots++;
				}
			}

			if (sumRowsAI == SIZE || sumColsAI == SIZE) {
				return AI;
			} else if (sumRowsHuman == -SIZE || sumColsHuman == -SIZE) {
				return HUMAN;
			}
			sumRowsAI = sumColsAI = sumRowsHuman = sumColsHuman = 0;

			// Check main diagonal
			if (board[x][x].getToken() == AI) {
				sumDiagsAI++;
			} else if (board[x][x].getToken() == HUMAN) {
				sumDiagsHuman--;
			}

			// Check right diagonal
			if (board[2 - x][x].getToken() == AI) {
				sumRdiagsAI++;
			} else if (board[2 - x][x].getToken() == HUMAN) {
				sumRdiagsHuman--;
			}
		}

		if (sumDiagsAI == SIZE || sumRdiagsAI == SIZE) {
			return AI;
		} else if (sumDiagsHuman == -SIZE || sumRdiagsHuman == -SIZE) {
			return HUMAN;
		}

		return freeSpots == 0 ? TIE : NONE;
	}

	public class Cell extends Pane {

		private char token;

		public Cell() {
			token = EMPTY;
			setOnMouseClicked(e -> handleMouseClick());
			setPrefSize(100, 100);
			setStyle("-fx-border-color: #dcdcdc");
		}

		private void setToken(char token, boolean draw) {
			this.token = token;

			if (draw && getToken() == AI) { // Draw 'X'
				Line x1 = new Line(30, 30, getWidth() - 30, getHeight() - 30);
				Line x2 = new Line(30, getHeight() - 30, getWidth() - 30, 30);
				x1.setStroke(Color.web("#e61414"));
				x2.setStroke(Color.web("#e61414"));
				x1.setStrokeWidth(20);
				x2.setStrokeWidth(20);

				getChildren().addAll(x1, x2);
			} else if (draw && getToken() == HUMAN) { // Draw 'O'
				Circle o = new Circle(getWidth() / 2, getHeight() / 2, getWidth() / 1.8 - 30, Color.TRANSPARENT);
				o.setStroke(Color.web("#0080ff"));
				o.setStrokeWidth(20);

				getChildren().add(o);
			}
		}

		private void handleMouseClick() {
			if (getToken() == EMPTY && findWinner() == NONE) {
				setToken(HUMAN, true);
				btnAiFirst.setDisable(true);

				lblStatus.setText("AI's turn (X)");
				makeBestAiMove();

				char result = findWinner();
				if (result == AI) {
					lblStatus.setText("AI wins!");
				} else if (result == HUMAN) {
					lblStatus.setText("Human wins!");
				} else if (result == TIE) {
					lblStatus.setText("It's a tie!");
				}
			}
		}

		private char getToken() {
			return token;
		}
	}

	public static void main(String[] args) {
		launch(args);
	}
}
