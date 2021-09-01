package utils;

import javafx.geometry.Pos;
import javafx.scene.Node;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Pane;
import javafx.scene.layout.VBox;

/**
 * Box pane builder.
 *
 * @author Sam Barba
 */
public class PaneBuilder {

	private BoxType boxType;

	private Pos alignment;

	private double spacing;

	private Node[] nodes;

	public PaneBuilder(BoxType boxType) {
		this.boxType = boxType;
	}

	public PaneBuilder withAlignment(Pos alignment) {
		this.alignment = alignment;
		return this;
	}

	public PaneBuilder withSpacing(double spacing) {
		this.spacing = spacing;
		return this;
	}

	public PaneBuilder withNodes(Node... nodes) {
		this.nodes = nodes;
		return this;
	}

	public Pane build() {
		switch (boxType) {
			case HBOX:
				HBox hbox = new HBox();
				hbox.setAlignment(alignment);
				hbox.setSpacing(spacing);
				if (nodes != null) {
					hbox.getChildren().setAll(nodes);
				}
				return hbox;
			case VBOX:
				VBox vbox = new VBox();
				vbox.setAlignment(alignment);
				vbox.setSpacing(spacing);
				if (nodes != null) {
					vbox.getChildren().setAll(nodes);
				}
				return vbox;
			default:
				throw new IllegalArgumentException("Invalid box type passed: " + boxType.toString());
		}
	}
}
