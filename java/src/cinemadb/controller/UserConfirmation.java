package cinemadb.controller;

import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.text.TextAlignment;
import javafx.stage.Stage;

import utils.BoxType;
import utils.ButtonBuilder;
import utils.PaneBuilder;

public class UserConfirmation {

	private static Stage stage = new Stage();

	private static boolean actionConfirmed;

	public static boolean confirm(String message) {
		actionConfirmed = false;

		Label lbl = new Label(message);
		lbl.setTextAlignment(TextAlignment.CENTER);

		Button btnYes = new ButtonBuilder()
			.withWidth(70)
			.withText("Yes")
			.withActionEvent(e -> {
				actionConfirmed = true;
				stage.close();
			})
			.build();
		Button btnNo = new ButtonBuilder()
			.withWidth(70)
			.withText("No")
			.withActionEvent(e -> stage.close())
			.build();

		HBox hboxBtns = (HBox) new PaneBuilder(BoxType.HBOX)
			.withAlignment(Pos.CENTER)
			.withSpacing(10)
			.withNodes(btnYes, btnNo)
			.build();
		VBox root = (VBox) new PaneBuilder(BoxType.VBOX)
			.withAlignment(Pos.CENTER)
			.withSpacing(20)
			.withNodes(lbl, hboxBtns)
			.build();

		Scene scene = new Scene(root, 600, 300);
		scene.getStylesheets().add("style.css");
		stage.setScene(scene);
		stage.setTitle("Confirmation");
		stage.setResizable(false);
		stage.showAndWait();
		return actionConfirmed;
	}
}
