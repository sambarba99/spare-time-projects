# Media service for cinema DB
# Author: Sam Barba
# Created 15/04/2022

from model.Media import Media
from model.Singleton import Singleton
from model.XMLMediaSerialiser import XMLMediaSerialiser

@Singleton
class MediaService:
	def __init__(self):
		self.serialiser = XMLMediaSerialiser.get_instance()

	# For main page tabular view
	def get_media_rows_with_filters(self, actor_substring="", director_substring="",
		genres=None, media_type=None):

		result = []

		for media in self.__get_all_media():
			actor_matches = any(actor_substring.lower() in ma.lower() for ma in media.actors)
			director_matches = any(director_substring.lower() in md.lower() for md in media.directors)
			genre_matches = genres is None or genres == ["All"] or any(mg in genres for mg in media.genres)
			media_type_matches = media_type is None or media_type == "All" or media.media_type == media_type

			if all([actor_matches, director_matches, genre_matches, media_type_matches]):
				result.append([media.mid, media.title, media.media_type, media.year,
					", ".join(media.genres), ", ".join(media.directors), ", ".join(media.actors)])

		# Sort by: type (movies then series), then year (descending), then title
		# (super pythonic way to remove leading 'the')
		return sorted(result, key=lambda row: (
			row[2],
			-int(row[3]),
			row[1].lower()[row[1].lower().startswith("the") and 3:].lstrip()
		))

	def get_media_by_id(self, media_id):
		return self.__convert_element_to_media_obj(
			self.serialiser.get_media_element_by_id(media_id)
		)

	def __get_all_media(self):
		return [self.__convert_element_to_media_obj(media_element)
			for media_element in self.serialiser.read_all_media()
			if media_element.get("id") is not None]

	def __convert_element_to_media_obj(self, media_element):
		mid = media_element.get("id")
		title = media_element.find("title").get("value")
		media_type = media_element.find("mediaType").get("value")
		year = media_element.find("year").get("value")
		genres = [g.get("value") for g in media_element.find("genres")]
		directors = [d.get("name") for d in media_element.find("directors")]
		actors = [a.get("name") for a in media_element.find("actors")]

		return Media(mid=mid, title=title, media_type=media_type, year=year,
			genres=genres, directors=directors, actors=actors)

	def convert_table_row_to_media(self, row_values):
		mid, title, media_type, year = row_values[:4]
		genres = row_values[4].split(", ")
		directors = row_values[5].split(", ")
		actors = row_values[6].split(", ")

		return Media(mid=mid, title=title, media_type=media_type, year=year,
			genres=genres, directors=directors, actors=actors)

	def add_media(self, new_media):
		new_media.mid = str(self.__create_new_id())
		self.serialiser.add_media(new_media)

	def __create_new_id(self):
		all_ids = sorted([int(m.mid) for m in self.__get_all_media()])
		if len(all_ids) == 0:
			return 1

		# Fill any gaps (if there aren't any, return the highest ID plus 1)
		for i in range(1, all_ids[-1] + 1):
			if i not in all_ids:
				return i

		return all_ids[-1] + 1

	def update_media(self, media_id, new_media):
		self.serialiser.update_media(media_id, new_media)

	def delete_media(self, media_id):
		self.serialiser.delete_media(media_id)
