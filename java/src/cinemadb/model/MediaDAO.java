package cinemadb.model;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import cinemadb.view.Constants;

public class MediaDAO {

	private XMLMediaSerialiser mediaSerialiser = XMLMediaSerialiser.getInstance();

	private static MediaDAO instance;

	private MediaDAO() {
	}

	public synchronized static MediaDAO getInstance() {
		if (instance == null) {
			instance = new MediaDAO();
		}
		return instance;
	}

	public void addMedia(Media media) {
		try {
			List<Media> allMedia = getAllMedia();

			File xmlFile = new File(Constants.MEDIA_FILE_PATH);
			if (!xmlFile.exists()) {
				xmlFile.getParentFile().mkdirs();
				xmlFile.createNewFile();
			}

			allMedia.add(media);
			mediaSerialiser.write(allMedia);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public void deleteMediaByIds(List<Integer> ids) {
		try {
			List<Media> allMedia = getAllMedia();
			List<Media> writeMedia = allMedia.stream()
				.filter(m -> !ids.contains(m.getId()))
				.collect(Collectors.toList());

			mediaSerialiser.write(writeMedia);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public List<Media> getAllMedia() {
		List<Media> allMedia = new ArrayList<>();

		File xmlFile = new File(Constants.MEDIA_FILE_PATH);
		if (xmlFile.exists()) {
			try {
				allMedia = mediaSerialiser.readAll();
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

		return allMedia;
	}
}
