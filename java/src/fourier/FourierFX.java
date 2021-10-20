package fourier;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import javafx.animation.Animation;
import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.input.KeyEvent;
import javafx.scene.input.MouseButton;
import javafx.scene.layout.Pane;
import javafx.scene.paint.Color;
import javafx.scene.shape.Circle;
import javafx.scene.shape.Line;
import javafx.stage.Stage;
import javafx.util.Duration;

/**
 * Fourier transform visualisation
 * 
 * Keys 1-3: change no. epicycles; F: draw flower; H: draw heart; P: draw pi; right-click: toggle user drawing mode
 * 
 * @author Sam Barba
 */
public class FourierFX extends Application {

	private int numEpicycles = 3;

	private boolean userDrawingMode = false;

	private List<int[]> userCoords = new ArrayList<>();

	private List<Complex> fourier = new ArrayList<>();

	private List<Circle> epicycleCircles = new ArrayList<>();

	private List<Line> epicycleLines = new ArrayList<>();

	private double time = 0;

	private Canvas canvas = new Canvas(1000, 1000);

	private GraphicsContext gc = canvas.getGraphicsContext2D();

	private Timeline timeline;

	private Pane pane = new Pane();

	@Override
	public void start(Stage primaryStage) {
		gc.setFill(Color.BLACK);
		setFourier(Arrays.asList(Presets.PI));
		refresh();

		Scene scene = new Scene(pane, 1000, 1000);
		scene.setOnKeyPressed(this::handleKeyPress);
		scene.setOnMouseClicked(e -> {
			if (e.getButton() == MouseButton.PRIMARY && userDrawingMode) {
				int x = (int) Math.round(e.getSceneX());
				int y = (int) Math.round(e.getSceneY());
				userCoords.add(new int[] { x - 500, y - 500 });
				gc.getPixelWriter().setColor(x, y, Color.RED);
			}

			if (e.getButton() == MouseButton.SECONDARY) {
				userDrawingMode = !userDrawingMode;
				refresh();

				if (userDrawingMode) {
					userCoords.clear();
					timeline.pause();
				} else {
					// User finishes drawing
					if (userCoords.size() > 1) {
						setFourier(userCoords);
					}
					timeline.play();
				}
			}
		});
		scene.setOnMouseDragged(e -> {
			if (!userDrawingMode || e.getButton() != MouseButton.PRIMARY) {
				return;
			}

			int x = (int) Math.round(e.getSceneX());
			int y = (int) Math.round(e.getSceneY());
			userCoords.add(new int[] { x - 500, y - 500 });
			gc.getPixelWriter().setColor(x, y, Color.RED);
		});
		scene.setFill(Color.BLACK);
		primaryStage.setScene(scene);
		primaryStage.setTitle("Fourier transform visualisation");
		primaryStage.setResizable(false);
		primaryStage.show();

		timeline = new Timeline(new KeyFrame(Duration.millis(4), event -> draw()));
		timeline.setCycleCount(Animation.INDEFINITE);
		timeline.play();
	}

	private void setFourier(List<int[]> coords) {
		List<Complex> complexCoords = new ArrayList<>();
		int re1 = 0, im1 = 0, re2 = 0, im2 = 0;

		for (int i = 0; i < coords.size() - 1; i++) {
			re1 = coords.get(i)[0];
			im1 = coords.get(i)[1];
			re2 = coords.get(i + 1)[0];
			im2 = coords.get(i + 1)[1];

			complexCoords.addAll(getCoordsBetween(re1, im1, re2, im2));
		}
		int firstRe = coords.get(0)[0];
		int firstIm = coords.get(0)[1];
		complexCoords.addAll(getCoordsBetween(re2, im2, firstRe, firstIm));

		fourier = discreteFourierTransform(complexCoords);
		time = 0;
	}

	/**
	 * Bresenham's algorithm
	 */
	private List<Complex> getCoordsBetween(int re1, int im1, int re2, int im2) {
		int dRe = Math.abs(re2 - re1);
		int dIm = -Math.abs(im2 - im1);
		int sRe = re1 < re2 ? 1 : -1;
		int sIm = im1 < im2 ? 1 : -1;
		int err = dRe + dIm;

		List<Complex> complex = new ArrayList<>();

		while (true) {
			complex.add(new Complex(re1, im1));

			if (re1 == re2 && im1 == im2) {
				return complex;
			}
			int e2 = 2 * err;
			if (e2 >= dIm) {
				err += dIm;
				re1 += sRe;
			}
			if (e2 <= dRe) {
				err += dRe;
				im1 += sIm;
			}
		}
	}

	private List<Complex> discreteFourierTransform(List<Complex> values) {
		int n = values.size();
		List<Complex> result = new ArrayList<>();

		for (int i = 0; i < n; i++) {
			Complex sum = new Complex(0, 0);

			for (int j = 0; j < n; j++) {
				double phi = 2 * Math.PI * i * j / n;
				Complex c = new Complex(Math.cos(phi), -Math.sin(phi));
				sum = sum.add(values.get(j).mult(c));
			}

			sum.setRe(sum.getRe() / n);
			sum.setIm(sum.getIm() / n);
			sum.setFreq(i);

			result.add(sum);
		}

		result.sort(Comparator.comparing(Complex::getAmp));
		Collections.reverse(result);
		return result;
	}

	private void refresh() {
		epicycleCircles.clear();
		epicycleLines.clear();
		for (int i = 0; i < numEpicycles; i++) {
			epicycleCircles.add(new Circle());
			epicycleLines.add(new Line());
		}

		gc.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());
		pane.getChildren().setAll(canvas);
		pane.getChildren().addAll(epicycleCircles);
		pane.getChildren().addAll(epicycleLines);
	}

	private void handleKeyPress(KeyEvent e) {
		switch (e.getCode()) {
			case DIGIT1:
				if (userDrawingMode) {
					return;
				}
				numEpicycles = 3;
				refresh();
				break;
			case DIGIT2:
				if (userDrawingMode) {
					return;
				}
				numEpicycles = 8;
				refresh();
				break;
			case DIGIT3:
				if (userDrawingMode) {
					return;
				}
				numEpicycles = 200;
				refresh();
				break;
			case F:
				setFourier(Arrays.asList(Presets.FLOWER));
				refresh();
				userDrawingMode = false;
				timeline.play();
				break;
			case H:
				setFourier(Arrays.asList(Presets.HEART));
				refresh();
				userDrawingMode = false;
				timeline.play();
				break;
			case P:
				setFourier(Arrays.asList(Presets.PI));
				refresh();
				userDrawingMode = false;
				timeline.play();
				break;
			default:
				return;
		}
	}

	private void draw() {
		// Resultant x,y coords of epicycles
		double[] coordsEpicycle = epicycles(500, 500);
		int drawX = (int) Math.round(coordsEpicycle[0]);
		int drawY = (int) Math.round(coordsEpicycle[1]);

		gc.getPixelWriter().setColor(drawX, drawY, Color.RED);

		time += 2 * Math.PI / fourier.size();
		time %= 2 * Math.PI;
	}

	private double[] epicycles(double x, double y) {
		int lim = Math.min(numEpicycles, fourier.size());

		for (int i = 0; i < lim; i++) {
			double prevX = x;
			double prevY = y;
			double radius = fourier.get(i).getAmp();
			double freq = fourier.get(i).getFreq();
			double phase = fourier.get(i).getPhase();
			x += radius * Math.cos(freq * time + phase);
			y += radius * Math.sin(freq * time + phase);

			epicycleCircles.get(i).setCenterX(prevX);
			epicycleCircles.get(i).setCenterY(prevY);
			epicycleCircles.get(i).setRadius(radius);
			epicycleCircles.get(i).setFill(Color.TRANSPARENT);
			epicycleCircles.get(i).setStroke(Color.web("#585858"));

			epicycleLines.get(i).setStartX(prevX);
			epicycleLines.get(i).setStartY(prevY);
			epicycleLines.get(i).setEndX(x);
			epicycleLines.get(i).setEndY(y);
			epicycleLines.get(i).setStroke(Color.WHITE);
		}

		return new double[] { x, y };
	}

	public static void main(String[] args) {
		launch(args);
	}
}
