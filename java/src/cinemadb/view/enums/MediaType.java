package cinemadb.view.enums;

import java.util.Arrays;

public enum MediaType {

	MOVIE("Movie"),
	SERIES("Series");

	private String strVal;

	private MediaType(String strVal) {
		this.strVal = strVal;
	}

	@Override
	public String toString() {
		return strVal;
	}

	public static MediaType getFromStr(String strVal) {
		return Arrays.stream(values())
			.filter(mediaType -> mediaType.toString().toUpperCase().equals(strVal.toUpperCase()))
			.findFirst()
			.orElseThrow(() -> new IllegalArgumentException("Invalid media type passed: " + strVal));
	}
}
