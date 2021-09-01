package cinemadb.controller;

import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.layout.VBox;
import javafx.scene.text.TextAlignment;
import javafx.stage.Stage;

import utils.BoxType;
import utils.ButtonBuilder;
import utils.PaneBuilder;

public class SystemNotification {

	private static Stage stage = new Stage();

	public static void display(String notificationType, String msg) {
		Label lbl = new Label(msg);
		lbl.setTextAlignment(TextAlignment.CENTER);

		Button btnOk = new ButtonBuilder()
			.withWidth(60)
			.withText("Ok")
			.withActionEvent(e -> stage.close())
			.build();

		VBox root = (VBox) new PaneBuilder(BoxType.VBOX)
			.withAlignment(Pos.CENTER)
			.withSpacing(20)
			.withNodes(lbl, btnOk)
			.build();

		Scene scene = new Scene(root, 600, 160);
		scene.getStylesheets().add("style.css");
		stage.setScene(scene);
		stage.setTitle(notificationType);
		stage.setResizable(false);
		stage.setAlwaysOnTop(true);
		stage.show();
	}
}
