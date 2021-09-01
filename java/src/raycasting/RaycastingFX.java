package raycasting;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Random;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.input.KeyEvent;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.Pane;
import javafx.scene.paint.Color;
import javafx.scene.shape.Line;
import javafx.stage.Stage;

/**
 * Raycasting demo
 * 
 * @author Sam Barba
 */
public class RaycastingFX extends Application {

	private static final Random RAND = new Random();

	private static final int RAY_LENGTH = 1800;

	private static final int WIDTH = 1500;

	private static final int HEIGHT = 900;

	// Start at top-left, looking towards centre
	private double sourceX = 0;

	private double sourceY = 0;

	private double sourceHeading = Math.toDegrees(Math.atan(0.6));

	private double povAngle = 60;

	private List<Line> rays = new ArrayList<>();

	private List<Line> walls = new ArrayList<>();

	private List<Optional<Double>> rayDistances = new ArrayList<>();

	private Pane pane = new Pane();

	@Override
	public void start(Stage primaryStage) {
		Scene scene = new Scene(pane, WIDTH, HEIGHT);
		scene.setOnKeyPressed(this::handleKeyPress);
		scene.setOnMouseMoved(this::moveRaySourceMouse);
		scene.getStylesheets().add("style.css");
		primaryStage.setScene(scene);
		primaryStage.setTitle("Raycasting demo");
		primaryStage.show();

		addWalls();
		generateRays();
		RenderedRaycastingFX.initialise(rayDistances);
	}

	private void addWalls() {
		for (int i = 0; i < 5; i++) {
			int ax = RAND.nextInt(WIDTH);
			int ay = RAND.nextInt(HEIGHT);
			int bx = RAND.nextInt(WIDTH);
			int by = RAND.nextInt(HEIGHT);

			Line wall = new Line(ax, ay, bx, by);
			wall.setStroke(Color.web("#dcdcdc"));
			wall.setStrokeWidth(4);

			walls.add(wall);
		}
		pane.getChildren().addAll(walls);
	}

	private void handleKeyPress(KeyEvent e) {
		switch (e.getCode()) {
			case W:
				sourceX += Math.cos(Math.toRadians(sourceHeading)) * 5;
				sourceY += Math.sin(Math.toRadians(sourceHeading)) * 5;
				break;
			case S:
				sourceX -= Math.cos(Math.toRadians(sourceHeading)) * 5;
				sourceY -= Math.sin(Math.toRadians(sourceHeading)) * 5;
				break;
			case A:
				sourceHeading -= 5;
				break;
			case D:
				sourceHeading += 5;
				break;
			case M:
				if (povAngle + 5 <= 180) {
					povAngle += 5;
				}
				break;
			case N:
				if (povAngle - 5 >= 5) {
					povAngle -= 5;
				}
				break;
			case R: // Reset
				sourceX = 0;
				sourceY = 0;
				sourceHeading = Math.toDegrees(Math.atan(0.6));
				povAngle = 60;
				pane.getChildren().clear();
				walls.clear();
				addWalls();
				break;
			default:
				break;
		}
		sourceX = (sourceX + WIDTH) % WIDTH;
		sourceY = (sourceY + HEIGHT) % HEIGHT;
		generateRays();
	}

	private void moveRaySourceMouse(MouseEvent e) {
		sourceX = e.getSceneX();
		sourceY = e.getSceneY();
		generateRays();
	}

	private void generateRays() {
		// Reset scene
		pane.getChildren().removeAll(rays);
		rays.clear();
		rayDistances.clear();

		for (double a = -povAngle / 2; a < povAngle / 2; a += 0.1) {
			double cosA = Math.cos(Math.toRadians(a + sourceHeading));
			double sinA = Math.sin(Math.toRadians(a + sourceHeading));
			double endX = cosA * RAY_LENGTH + sourceX;
			double endY = sinA * RAY_LENGTH + sourceY;

			Line ray = new Line(sourceX, sourceY, endX, endY);
			ray.setStroke(Color.web("#dcdcdc"));

			Optional<Double> d = Optional.empty();
			for (Line w : walls) {
				Optional<double[]> intersection = findIntersection(ray, w);
				if (intersection.isPresent()) {
					ray.setEndX(intersection.get()[0]);
					ray.setEndY(intersection.get()[1]);
					d = Optional.of(dist(ray.getStartX(), ray.getStartY(), ray.getEndX(), ray.getEndY()));
				}
			}
			rays.add(ray);
			rayDistances.add(d);
		}

		pane.getChildren().addAll(rays);
		RenderedRaycastingFX.refreshView(rayDistances);
	}

	private Optional<double[]> findIntersection(Line ray, Line wall) {
		double axRay = ray.getStartX();
		double ayRay = ray.getStartY();
		double bxRay = ray.getEndX();
		double byRay = ray.getEndY();
		double axWall = wall.getStartX();
		double ayWall = wall.getStartY();
		double bxWall = wall.getEndX();
		double byWall = wall.getEndY();

		double denom = (axRay - bxRay) * (ayWall - byWall) - (ayRay - byRay) * (axWall - bxWall);
		if (denom == 0) {
			return Optional.empty();
		}

		double t = ((axRay - axWall) * (ayWall - byWall) - (ayRay - ayWall) * (axWall - bxWall)) / denom;
		double u = -((axRay - bxRay) * (ayRay - ayWall) - (ayRay - byRay) * (axRay - axWall)) / denom;

		if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
			double[] intersection = new double[2];
			intersection[0] = axRay + t * (bxRay - axRay);
			intersection[1] = ayRay + t * (byRay - ayRay);
			return Optional.of(intersection);
		}
		return Optional.empty();
	}

	private double dist(double x1, double y1, double x2, double y2) {
		return Math.sqrt(Math.pow(x1 - x2, 2) + Math.pow(y1 - y2, 2));
	}

	public static void main(String[] args) {
		launch(args);
	}
}
