package cinemadb.model;

import static org.junit.Assert.assertNotNull;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

import cinemadb.view.enums.Genre;

public class MediaService {

	private static MediaService instance;

	private MediaDAO mediaDao;

	private MediaService(MediaDAO mediaDao) {
		assertNotNull(mediaDao);
		this.mediaDao = mediaDao;
	}

	public synchronized static MediaService getInstance() {
		if (instance == null) {
			instance = new MediaService(MediaDAO.getInstance());
		}
		return instance;
	}

	public void addMedia(Media media) {
		media.setId(getNewMediaId());
		mediaDao.addMedia(media);
	}

	public void deleteMediaByIds(List<Integer> ids) {
		mediaDao.deleteMediaByIds(ids);
	}

	public List<Media> getAllMedia() {
		return mediaDao.getAllMedia();
	}

	public Optional<Media> getMediaById(int id) {
		return getAllMedia().stream().filter(m -> m.getId() == id).findFirst();
	}

	private int getNewMediaId() {
		List<Media> allMedia = getAllMedia();
		return allMedia.isEmpty() ? 1 : allMedia.stream().max(Comparator.comparing(Media::getId)).get().getId() + 1;
	}

	public List<MediaDTO> getMediaDTOsWithFilters(List<String> genresFilter, List<String> mediaTypesFilter,
		String titleSubstring, String directorSubstring, String actorSubstring) {

		List<Media> allMedia = getAllMedia();
		List<MediaDTO> result = new ArrayList<>();

		for (Media media : allMedia) {
			List<String> mediaGenres = media.getGenres().stream()
				.map(Genre::toString)
				.collect(Collectors.toList());

			boolean genreMatches = genresFilter.isEmpty() || genresFilter.stream().anyMatch(mediaGenres::contains);

			boolean mediaTypeMatches = mediaTypesFilter.isEmpty()
				|| mediaTypesFilter.contains(media.getMediaType().toString());

			boolean titleMatches = media.getTitle().toLowerCase().contains(titleSubstring.toLowerCase());

			boolean directorMatches = directorSubstring.isEmpty() || media.getDirectors()
				.stream()
				.anyMatch(directorName -> directorName.toLowerCase().contains(directorSubstring));

			boolean actorMatches = actorSubstring.isEmpty()
				|| media.getActors().stream().anyMatch(actorName -> actorName.toLowerCase().contains(actorSubstring));

			if (mediaTypeMatches && genreMatches && titleMatches && directorMatches && actorMatches) {
				result.add(convertToMediaDTO(media));
			}
		}

		Comparator byMediaType = Comparator.comparing(MediaDTO::getMediaType);
		Comparator byYear = Comparator.comparing(MediaDTO::getYear).reversed();
		Comparator byTitle = Comparator.comparing(MediaDTO::getTitle);

		result.sort(byMediaType.thenComparing(byYear).thenComparing(byTitle));

		return result;
	}

	private MediaDTO convertToMediaDTO(Media media) {
		MediaDTO mediaDto = new MediaDTO();

		mediaDto.setId(media.getId());
		mediaDto.setTitle(media.getTitle());
		mediaDto.setMediaType(media.getMediaType().toString());
		mediaDto.setYear(media.getYear());

		String genres = "";
		for (Genre g : media.getGenres()) {
			genres += g.toString() + ", ";
		}

		mediaDto.setGenres(genres.substring(0, genres.length() - 2));

		String directors = "";
		for (String d : media.getDirectors()) {
			directors += d + ", ";
		}
		mediaDto.setDirectors(directors.substring(0, directors.length() - 2));

		String actors = "";
		for (String a : media.getActors()) {
			actors += a + ", ";
		}

		mediaDto.setActors(actors.substring(0, actors.length() - 2));

		return mediaDto;
	}
}
