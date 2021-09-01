package cinemadb.view.enums;

import java.util.Arrays;

public enum Genre {

	ACTION("Action"),
	ADVENTURE("Adventure"),
	ANIMATION("Animation"),
	BIOGRAPHY("Biography"),
	COMEDY("Comedy"),
	CRIME("Crime"),
	DARK_COMEDY("Dark comedy"),
	DOCUMENTARY("Documentary"),
	DRAMA("Drama"),
	FANTASY("Fantasy"),
	HORROR("Horror"),
	MUSICAL("Musical"),
	MYSTERY("Mystery"),
	ROMANCE("Romance"),
	SCI_FI("Sci-fi"),
	THRILLER("Thriller"),
	WAR("War"),
	WESTERN("Western");

	private String strVal;

	private Genre(String strVal) {
		this.strVal = strVal;
	}

	@Override
	public String toString() {
		return strVal;
	}

	public static Genre getFromStr(String strVal) {
		return Arrays.stream(values())
			.filter(g -> g.toString().toUpperCase().equals(strVal.toUpperCase()))
			.findFirst()
			.orElseThrow(() -> new IllegalArgumentException("Invalid genre passed: " + strVal));
	}
}
