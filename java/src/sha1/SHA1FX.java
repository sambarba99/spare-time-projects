package sha1;

import java.security.MessageDigest;

import javafx.application.Application;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Label;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;

import utils.BoxType;
import utils.PaneBuilder;

/**
 * SHA-1 encryption demo
 * 
 * @author Sam Barba
 */
public class SHA1FX extends Application {

	private Label lblEnterRaw = new Label("Enter raw string to encrypt:");

	private TextField txtRaw = new TextField();

	private TextArea txtAreaResult = new TextArea("'' encrypted: " + sha1(""));

	@Override
	public void start(Stage primaryStage) {
		VBox root = setup();

		Scene scene = new Scene(root, 500, 350);
		scene.getStylesheets().add("style.css");
		primaryStage.setScene(scene);
		primaryStage.setTitle("SHA-1");
		primaryStage.show();
	}

	private String sha1(String text) {
		try {
			MessageDigest md = MessageDigest.getInstance("SHA-1");
			md.update(text.getBytes("iso-8859-1"), 0, text.length());
			byte[] sha1hash = md.digest();
			return convertToHex(sha1hash);
		} catch (Exception e) {
			e.printStackTrace();
			return "Invalid value";
		}
	}

	private String convertToHex(byte[] data) {
		String result = "";
		for (int i = 0; i < data.length; i++) {
			result += Integer.toString((data[i] & 255) + 256, 16).substring(1);
		}
		return result;
	}

	private VBox setup() {
		txtRaw.textProperty().addListener((obs, oldText, newText) -> {
			txtAreaResult.setText("'" + newText + "' encrypted: " + sha1(newText));
		});
		txtRaw.setMaxWidth(200);

		txtAreaResult.setEditable(false);
		txtAreaResult.setMaxWidth(380);

		return (VBox) new PaneBuilder(BoxType.VBOX)
			.withAlignment(Pos.CENTER)
			.withSpacing(20)
			.withNodes(lblEnterRaw, txtRaw, txtAreaResult)
			.build();
	}

	public static void main(String[] args) {
		launch(args);
	}
}
