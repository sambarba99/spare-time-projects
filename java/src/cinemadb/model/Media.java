package cinemadb.model;

import java.util.ArrayList;
import java.util.List;

import cinemadb.view.enums.Genre;
import cinemadb.view.enums.MediaType;

public class Media {

	private int id;

	private String title;

	private MediaType mediaType;

	private int year;

	private List<Genre> genres;

	private List<String> directors;

	private List<String> actors;

	public Media() {
		genres = new ArrayList<>();
		directors = new ArrayList<>();
		actors = new ArrayList<>();
	}

	public Media(String title, MediaType mediaType, int year, List<Genre> genres, List<String> directors,
		List<String> actors) {

		this.title = title;
		this.mediaType = mediaType;
		this.year = year;
		this.genres = genres;
		this.directors = directors;
		this.actors = actors;
	}

	public int getId() {
		return id;
	}

	public void setId(int id) {
		this.id = id;
	}

	public String getTitle() {
		return title;
	}

	public void setTitle(String title) {
		this.title = title;
	}

	public MediaType getMediaType() {
		return mediaType;
	}

	public void setMediaType(MediaType mediaType) {
		this.mediaType = mediaType;
	}

	public int getYear() {
		return year;
	}

	public void setYear(int year) {
		this.year = year;
	}

	public List<Genre> getGenres() {
		return genres;
	}

	public void setGenres(List<Genre> genres) {
		this.genres.clear();
		this.genres.addAll(genres);
	}

	public List<String> getDirectors() {
		return directors;
	}

	public void setDirectors(List<String> directors) {
		this.directors.clear();
		this.directors.addAll(directors);
	}

	public List<String> getActors() {
		return actors;
	}

	public void setActors(List<String> actors) {
		this.actors.clear();
		this.actors.addAll(actors);
	}

	@Override
	public String toString() {
		String result = "Title: ";

		result += title == null || title.isEmpty() ? "not set" : title;
		result += "\n\nMedia type: " + mediaType.toString();
		result += "\n\nYear: " + (year != 0 ? year : "not set");

		String genresStr = "";
		for (Genre g : genres) {
			genresStr += g.toString() + ", ";
		}

		result += "\n\nGenres: " + (genresStr.isEmpty() ? "not set" : genresStr.substring(0, genresStr.length() - 2));

		String directorsStr = "";
		for (String d : directors) {
			directorsStr += d + ", ";
		}

		result += "\n\nDirectors: "
			+ (directorsStr.isEmpty() ? "not set" : directorsStr.substring(0, directorsStr.length() - 2));

		String actorsStr = "";
		for (String a : actors) {
			actorsStr += a + ", ";
		}

		result += "\n\nActors: " + (actorsStr.isEmpty() ? "not set" : actorsStr.substring(0, actorsStr.length() - 2));

		return result;
	}
}
