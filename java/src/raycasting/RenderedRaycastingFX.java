package raycasting;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import javafx.scene.Scene;
import javafx.scene.layout.Pane;
import javafx.scene.paint.Color;
import javafx.scene.shape.Rectangle;
import javafx.stage.Stage;

/**
 * Raycasting rendering demo
 * 
 * @author Sam Barba
 */
public class RenderedRaycastingFX {

	private static final int WIDTH = 1500;

	private static final int HEIGHT = 900;

	private static final int RAY_LENGTH = 1800;

	private static List<Rectangle> wallViews = new ArrayList<>();

	private static Pane pane = new Pane();

	public static void initialise(List<Optional<Double>> rayDistances) {
		refreshView(rayDistances);

		Scene scene = new Scene(pane, WIDTH, HEIGHT);
		scene.setFill(Color.BLACK);
		Stage stage = new Stage();
		stage.setScene(scene);
		stage.setTitle("Rendered Rays");
		stage.show();
	}

	public static void refreshView(List<Optional<Double>> rayDistances) {
		double w = (double) WIDTH / rayDistances.size();
		wallViews.clear();

		for (int i = 0; i < rayDistances.size(); i++) {
			Optional<Double> d = rayDistances.get(i);

			double h = d.isPresent() ? map(d.get(), 0, RAY_LENGTH, HEIGHT, 0) : 0;
			double y = (HEIGHT - h) / 2; // Draw rectangles from centre
			Rectangle rectangle = new Rectangle(i * w, y, w, h);

			int col = d.isPresent() ? (int) map(d.get() * d.get(), 0, RAY_LENGTH * RAY_LENGTH, 255, 0) : 0;
			rectangle.setFill(Color.rgb(col, col, col));
			rectangle.setStroke(Color.rgb(col, col, col));
			wallViews.add(rectangle);
		}

		pane.getChildren().setAll(wallViews);
	}

	private static double map(double x, double fromLo, double fromHi, double toLo, double toHi) {
		return (x - fromLo) * (toHi - toLo) / (fromHi - fromLo) + toLo;
	}
}
