package cinemadb.controller;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import javafx.application.Application;
import javafx.collections.ObservableList;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Accordion;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Label;
import javafx.scene.control.SelectionMode;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableView;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.control.TitledPane;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.scene.layout.AnchorPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;

import cinemadb.model.Media;
import cinemadb.model.MediaDTO;
import cinemadb.model.MediaService;
import cinemadb.view.Constants;
import cinemadb.view.enums.Genre;
import cinemadb.view.enums.MediaType;
import utils.BoxType;
import utils.ButtonBuilder;
import utils.PaneBuilder;

public class AllMedia extends Application {

	private MediaService mediaService = MediaService.getInstance();

	private TableView tblMedia = new TableView();

	private Label lblSelectMedia = new Label();

	private Accordion accFilters = new Accordion();

	private List<CheckBox> cbGenres;

	private List<CheckBox> cbMediaTypes;

	private TextField txtTitleSubstringFilter = new TextField();

	private TextField txtDirectorSubstringFilter = new TextField();

	private TextField txtActorSubstringFilter = new TextField();

	private TextArea txtAreaMedia = new TextArea();

	@Override
	public void start(Stage primaryStage) throws Exception {
		Button btnAddMedia = new ButtonBuilder()
			.withWidth(160)
			.withText("Add media")
			.withActionEvent(e -> {
				// If added new media, refresh media TableView
				if (AddMedia.display()) {
					refreshMediaTbl();
				}
			})
			.build();
		Button btnDelMedia = new ButtonBuilder()
			.withWidth(160)
			.withText("Delete media")
			.withActionEvent(e -> {
				ObservableList<MediaDTO> mediaDtos = tblMedia.getSelectionModel().getSelectedItems();

				if (mediaDtos.isEmpty()) {
					SystemNotification.display(Constants.ERROR_NOTIFICATION, "Select at least 1 media");
				} else if (UserConfirmation.confirm("Delete " + mediaDtos.size() + " media?")) {
					List<Integer> deleteIds = mediaDtos.stream().map(MediaDTO::getId).collect(Collectors.toList());

					mediaService.deleteMediaByIds(deleteIds);
					refreshMediaTbl();
					SystemNotification.display(Constants.SUCCESS_NOTIFICATION, "Selected media deleted.");
				}
			})
			.build();

		VBox vboxTbl = (VBox) new PaneBuilder(BoxType.VBOX)
			.withAlignment(Pos.CENTER)
			.withSpacing(10)
			.withNodes(lblSelectMedia, tblMedia)
			.build();
		VBox vboxBtns = (VBox) new PaneBuilder(BoxType.VBOX)
			.withAlignment(Pos.CENTER)
			.withSpacing(10)
			.withNodes(btnAddMedia, btnDelMedia)
			.build();
		HBox hboxTxtBtns = (HBox) new PaneBuilder(BoxType.HBOX)
			.withAlignment(Pos.CENTER)
			.withSpacing(20)
			.withNodes(txtAreaMedia, vboxBtns)
			.build();
		VBox vboxFilters = (VBox) new PaneBuilder(BoxType.VBOX)
			.withAlignment(Pos.TOP_LEFT)
			.withSpacing(10)
			.withNodes(new Label("Filter by title?"), txtTitleSubstringFilter, new Label("Filter by director?"),
				txtDirectorSubstringFilter, new Label("Filter by actor?"), txtActorSubstringFilter,
				new Label("Other filters:"), accFilters)
			.build();

		AnchorPane root = new AnchorPane(vboxTbl, hboxTxtBtns, vboxFilters);
		AnchorPane.setTopAnchor(vboxTbl, 40.0);
		AnchorPane.setLeftAnchor(vboxTbl, 30.0);
		AnchorPane.setBottomAnchor(hboxTxtBtns, 40.0);
		AnchorPane.setLeftAnchor(hboxTxtBtns, 150.0);
		AnchorPane.setTopAnchor(vboxFilters, 40.0);
		AnchorPane.setRightAnchor(vboxFilters, 30.0);

		setupNodes();

		Scene scene = new Scene(root, 1380, 820);
		scene.getStylesheets().add("style.css");
		primaryStage.setScene(scene);
		primaryStage.setTitle("MediaDB");
		primaryStage.setResizable(false);
		primaryStage.show();
	}

	private void setupNodes() {
		/*
		 * Set up TableView of media
		 */
		TableColumn<MediaDTO, Integer> colId = new TableColumn<>("ID");
		TableColumn<MediaDTO, String> colTitle = new TableColumn<>("Title");
		TableColumn<MediaDTO, String> colMediaType = new TableColumn<>("Type");
		TableColumn<MediaDTO, Integer> colYear = new TableColumn<>("Year");
		TableColumn<MediaDTO, String> colGenre = new TableColumn<>("Genres");
		TableColumn<MediaDTO, String> colDirector = new TableColumn<>("Directors");
		TableColumn<MediaDTO, String> colActors = new TableColumn<>("Actors");

		colId.setCellValueFactory(new PropertyValueFactory<>("id"));
		colTitle.setCellValueFactory(new PropertyValueFactory<>("title"));
		colMediaType.setCellValueFactory(new PropertyValueFactory<>("mediaType"));
		colYear.setCellValueFactory(new PropertyValueFactory<>("year"));
		colGenre.setCellValueFactory(new PropertyValueFactory<>("genres"));
		colDirector.setCellValueFactory(new PropertyValueFactory<>("directors"));
		colActors.setCellValueFactory(new PropertyValueFactory<>("actors"));

		colId.setPrefWidth(49);
		colTitle.setPrefWidth(200);
		colMediaType.setPrefWidth(84);
		colYear.setPrefWidth(68);
		colGenre.setPrefWidth(250);
		colDirector.setPrefWidth(200);
		colActors.setPrefWidth(250);

		tblMedia.getColumns().setAll(colId, colTitle, colMediaType, colYear, colGenre, colDirector, colActors);
		tblMedia.getSelectionModel().setSelectionMode(SelectionMode.MULTIPLE);
		tblMedia.getSelectionModel().selectedItemProperty().addListener(listener -> {
			MediaDTO mediaDto = (MediaDTO) tblMedia.getSelectionModel().getSelectedItem();
			if (mediaDto != null) {
				Media media = mediaService.getMediaById(mediaDto.getId()).get();
				txtAreaMedia.setText(media.toString());
			}
		});
		tblMedia.setPrefSize(1120, 400);
		tblMedia.setEditable(false);

		/*
		 * Set up genres filter
		 */
		cbGenres = Arrays.stream(Genre.values())
			.map(genre -> new CheckBox(genre.toString()))
			.collect(Collectors.toList());

		// TableView of media must be refreshed if these CheckBoxes are toggled
		for (CheckBox cb : cbGenres) {
			cb.getStyleClass().add("check-box-accordion");
			cb.selectedProperty().addListener(listener -> refreshMediaTbl());
		}

		VBox vboxGenres = (VBox) new PaneBuilder(BoxType.VBOX)
			.withAlignment(Pos.CENTER_LEFT)
			.withSpacing(3)
			.build();
		vboxGenres.getChildren().setAll(cbGenres);

		/*
		 * Set up media type filter
		 */
		cbMediaTypes = Arrays.stream(MediaType.values())
			.map(mediaType -> new CheckBox(mediaType.toString()))
			.collect(Collectors.toList());

		// TableView of media must be refreshed if these CheckBoxes are toggled
		for (CheckBox cb : cbMediaTypes) {
			cb.getStyleClass().add("check-box-accordion");
			cb.selectedProperty().addListener(listener -> refreshMediaTbl());
		}

		VBox vboxMediaTypes = (VBox) new PaneBuilder(BoxType.VBOX)
			.withAlignment(Pos.CENTER_LEFT)
			.withSpacing(3)
			.build();
		vboxMediaTypes.getChildren().setAll(cbMediaTypes);

		TitledPane tPaneGenres = new TitledPane("Filter by genre", vboxGenres);
		TitledPane tPaneMediaTypes = new TitledPane("Filter by media type", vboxMediaTypes);
		accFilters.getPanes().setAll(tPaneGenres, tPaneMediaTypes);

		/*
		 * Set up substring filters
		 */
		txtTitleSubstringFilter.clear();
		txtDirectorSubstringFilter.clear();
		txtActorSubstringFilter.clear();
		txtTitleSubstringFilter.textProperty().addListener(listener -> refreshMediaTbl());
		txtDirectorSubstringFilter.textProperty().addListener(listener -> refreshMediaTbl());
		txtActorSubstringFilter.textProperty().addListener(listener -> refreshMediaTbl());

		/*
		 * Set up media display TextArea
		 */
		txtAreaMedia.setEditable(false);
		txtAreaMedia.setPrefSize(700, 300);
		txtAreaMedia.setText("No media selected.");

		refreshMediaTbl();
	}

	private void refreshMediaTbl() {
		List<String> genreFilters = cbGenres.stream()
			.filter(CheckBox::isSelected)
			.map(CheckBox::getText)
			.collect(Collectors.toList());

		List<String> mediaTypeFilters = cbMediaTypes.stream()
			.filter(CheckBox::isSelected)
			.map(CheckBox::getText)
			.collect(Collectors.toList());

		List<MediaDTO> mediaDtos = mediaService.getMediaDTOsWithFilters(genreFilters, mediaTypeFilters,
			txtTitleSubstringFilter.getText(), txtDirectorSubstringFilter.getText(), txtActorSubstringFilter.getText());
		tblMedia.getItems().setAll(mediaDtos);

		if (mediaDtos.isEmpty()) {
			lblSelectMedia.setText("No media to show.");
		} else {
			lblSelectMedia.setText("Select one of " + mediaDtos.size() + " media to view!");
		}
		txtAreaMedia.setText("No media selected.");
	}

	public static void main(String[] args) {
		launch(args);
	}
}
