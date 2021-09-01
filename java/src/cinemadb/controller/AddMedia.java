package cinemadb.controller;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Accordion;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.Label;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.control.TitledPane;
import javafx.scene.layout.AnchorPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;

import cinemadb.model.Media;
import cinemadb.model.MediaService;
import cinemadb.view.Constants;
import cinemadb.view.enums.Genre;
import cinemadb.view.enums.MediaType;
import utils.BoxType;
import utils.ButtonBuilder;
import utils.PaneBuilder;

public class AddMedia {

	private static MediaService mediaService = MediaService.getInstance();

	private static Stage stage = new Stage();

	private static Media media;

	private static boolean added;

	private static TextField txtTitle = new TextField();

	private static ChoiceBox choiceMediaType = new ChoiceBox();

	private static TextField txtYear = new TextField();

	private static Accordion accGenres = new Accordion();

	private static TextField txtDirector = new TextField();

	private static Button btnAddDirector;

	private static CheckBox cbVariousDirectors = new CheckBox("Various directors");

	private static TextField txtActor = new TextField();

	private static TextArea txtAreaResult = new TextArea();

	public static boolean display() {
		added = false;

		btnAddDirector = new ButtonBuilder()
			.withWidth(100)
			.withText("Add")
			.withActionEvent(e -> {
				String director = txtDirector.getText().trim();
				if (director.isEmpty()) {
					SystemNotification.display(Constants.ERROR_NOTIFICATION, "Director name cannot be empty");
				} else {
					media.getDirectors().add(director);
					txtAreaResult.setText(media.toString());
				}
				txtDirector.clear();
			})
			.build();
		Button btnAddActor = new ButtonBuilder()
			.withWidth(100)
			.withText("Add")
			.withActionEvent(e -> {
				String actor = txtActor.getText().trim();
				if (actor.isEmpty()) {
					SystemNotification.display(Constants.ERROR_NOTIFICATION, "Actor name cannot be empty");
				} else {
					media.getActors().add(actor);
					txtAreaResult.setText(media.toString());
				}
				txtActor.clear();
			})
			.build();
		Button btnAddMedia = new ButtonBuilder()
			.withWidth(100)
			.withText("Add!")
			.withActionEvent(e -> {
				if (validateAndAddMedia()) {
					added = true;
					setupNodes(); // Reset fields for adding media
					SystemNotification.display(Constants.SUCCESS_NOTIFICATION, "Media added!");
				}
			})
			.build();
		Button btnReset = new ButtonBuilder()
			.withWidth(100)
			.withText("Reset")
			.withActionEvent(e -> setupNodes())
			.build();

		HBox hboxTitle = (HBox) new PaneBuilder(BoxType.HBOX)
			.withAlignment(Pos.CENTER)
			.withSpacing(10)
			.withNodes(new Label("Set title:"), txtTitle)
			.build();
		HBox hboxType = (HBox) new PaneBuilder(BoxType.HBOX)
			.withAlignment(Pos.CENTER)
			.withSpacing(10)
			.withNodes(new Label("Select type:"), choiceMediaType)
			.build();
		HBox hboxYear = (HBox) new PaneBuilder(BoxType.HBOX)
			.withAlignment(Pos.CENTER)
			.withSpacing(10)
			.withNodes(new Label("Set year:"), txtYear)
			.build();
		HBox hboxDirector = (HBox) new PaneBuilder(BoxType.HBOX)
			.withAlignment(Pos.CENTER)
			.withSpacing(10)
			.withNodes(new Label("Add director:"), txtDirector, btnAddDirector, cbVariousDirectors)
			.build();
		HBox hboxActor = (HBox) new PaneBuilder(BoxType.HBOX)
			.withAlignment(Pos.CENTER)
			.withSpacing(10)
			.withNodes(new Label("Add actor:"), txtActor, btnAddActor)
			.build();
		HBox hboxGenres = (HBox) new PaneBuilder(BoxType.HBOX)
			.withSpacing(10)
			.withNodes(new Label("Select genres:"), accGenres)
			.build();
		HBox hboxBtns = (HBox) new PaneBuilder(BoxType.HBOX)
			.withAlignment(Pos.CENTER)
			.withSpacing(10)
			.withNodes(btnAddMedia, btnReset)
			.build();
		VBox vboxTxtAreaBtns = (VBox) new PaneBuilder(BoxType.VBOX)
			.withAlignment(Pos.CENTER)
			.withSpacing(10)
			.withNodes(txtAreaResult, hboxBtns)
			.build();

		AnchorPane root = new AnchorPane(hboxTitle, hboxType, hboxYear, hboxDirector, hboxActor, hboxGenres,
			vboxTxtAreaBtns);
		AnchorPane.setTopAnchor(hboxTitle, 30.0);
		AnchorPane.setLeftAnchor(hboxTitle, 61.0);
		AnchorPane.setTopAnchor(hboxType, 70.0);
		AnchorPane.setLeftAnchor(hboxType, 37.0);
		AnchorPane.setTopAnchor(hboxYear, 110.0);
		AnchorPane.setLeftAnchor(hboxYear, 55.0);
		AnchorPane.setTopAnchor(hboxDirector, 147.0);
		AnchorPane.setLeftAnchor(hboxDirector, 30.0);
		AnchorPane.setTopAnchor(hboxActor, 190.0);
		AnchorPane.setLeftAnchor(hboxActor, 46.0);
		AnchorPane.setTopAnchor(hboxGenres, 40.0);
		AnchorPane.setRightAnchor(hboxGenres, 30.0);
		AnchorPane.setBottomAnchor(vboxTxtAreaBtns, 30.0);
		AnchorPane.setLeftAnchor(vboxTxtAreaBtns, 100.0);

		setupNodes();

		Scene scene = new Scene(root, 800, 550);
		scene.getStylesheets().add("style.css");
		stage.setScene(scene);
		stage.setTitle("Add New Media");
		stage.setResizable(false);
		stage.showAndWait();
		return added;
	}

	private static void setupNodes() {
		media = new Media();
		media.setMediaType(MediaType.values()[0]);

		/*
		 * Set up media type selector
		 */
		choiceMediaType.getItems().setAll(Arrays.stream(MediaType.values())
			.map(MediaType::toString)
			.collect(Collectors.toList()));
		choiceMediaType.getSelectionModel().selectFirst();
		choiceMediaType.getSelectionModel().selectedItemProperty().addListener(listener -> {
			Object selectedMediaType = choiceMediaType.getSelectionModel().getSelectedItem();
			if (selectedMediaType != null) {
				media.setMediaType(MediaType.getFromStr(selectedMediaType.toString()));
				txtAreaResult.setText(media.toString());
			}
		});
		choiceMediaType.setPrefWidth(100);

		/*
		 * Set up genre selector
		 */
		List<CheckBox> cbGenres = Arrays.stream(Genre.values())
			.map(genre -> new CheckBox(genre.toString()))
			.collect(Collectors.toList());

		// Update TextArea of result media if these CheckBoxes are toggled
		for (CheckBox cb : cbGenres) {
			cb.getStyleClass().add("check-box-accordion");

			cb.selectedProperty().addListener(listener -> {
				List<Genre> genres = cbGenres.stream()
					.filter(CheckBox::isSelected)
					.map(c -> Genre.getFromStr(c.getText()))
					.collect(Collectors.toList());

				media.setGenres(genres);
				txtAreaResult.setText(media.toString());
			});
		}

		VBox vboxGenres = (VBox) new PaneBuilder(BoxType.VBOX)
			.withAlignment(Pos.CENTER_LEFT)
			.withSpacing(3)
			.build();
		vboxGenres.getChildren().setAll(cbGenres);

		accGenres.getPanes().setAll(new TitledPane("Genres", vboxGenres));

		cbVariousDirectors.setSelected(false);
		cbVariousDirectors.selectedProperty().addListener(listener -> {
			if (cbVariousDirectors.isSelected()) {
				media.setDirectors(Collections.singletonList("Various"));
				btnAddDirector.setDisable(true);
			} else {
				media.setDirectors(Collections.emptyList());
				btnAddDirector.setDisable(false);
			}
			txtAreaResult.setText(media.toString());
		});

		/*
		 * Clear text fields
		 */
		txtTitle.clear();
		txtYear.clear();
		txtDirector.clear();
		txtActor.clear();
		txtTitle.textProperty().addListener(listener -> {
			media.setTitle(txtTitle.getText().trim());
			txtAreaResult.setText(media.toString());
		});
		txtYear.textProperty().addListener(listener -> {
			String yearStr = txtYear.getText().trim();
			if (yearStr.matches("^\\d{4}$")) {
				media.setYear(Integer.parseInt(yearStr));
				txtAreaResult.setText(media.toString());
			}
		});
		txtAreaResult.setText(media.toString());
		txtAreaResult.setEditable(false);
		txtAreaResult.setPrefSize(500, 220);
	}

	private static boolean validateAndAddMedia() {
		if (media.getTitle() == null) {
			SystemNotification.display(Constants.ERROR_NOTIFICATION, "Title cannot be empty");
			return false;
		} else if (media.getYear() == 0) {
			SystemNotification.display(Constants.ERROR_NOTIFICATION, "Year must be 4 digits");
			return false;
		} else if (media.getDirectors().isEmpty()) {
			SystemNotification.display(Constants.ERROR_NOTIFICATION, "Media must have at least 1 director");
			return false;
		} else if (media.getActors().isEmpty()) {
			SystemNotification.display(Constants.ERROR_NOTIFICATION, "Media must have at least 1 actor");
			return false;
		} else if (media.getGenres().isEmpty()) {
			SystemNotification.display(Constants.ERROR_NOTIFICATION, "Media must have at least 1 genre");
			return false;
		}

		mediaService.addMedia(media);
		return true;
	}
}
