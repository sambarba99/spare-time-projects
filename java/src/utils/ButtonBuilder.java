package utils;

import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.scene.control.Button;

/**
 * Button builder.
 *
 * @author Sam Barba
 */
public class ButtonBuilder {

	private double width;

	private String text;

	private EventHandler<ActionEvent> actionEvent;

	public ButtonBuilder() {
	}

	public ButtonBuilder withWidth(double width) {
		this.width = width;
		return this;
	}

	public ButtonBuilder withText(String text) {
		this.text = text;
		return this;
	}

	public ButtonBuilder withActionEvent(EventHandler<ActionEvent> actionEvent) {
		this.actionEvent = actionEvent;
		return this;
	}

	public Button build() {
		Button btn = new Button(text);
		btn.setPrefWidth(width);
		btn.setOnAction(actionEvent);
		return btn;
	}
}
