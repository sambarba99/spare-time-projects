# Media class
# Author: Sam Barba
# Created 15/04/2022

class Media:
	def __init__(self, *, mid=None, title=None, media_type=None, year=None, genres=None,
		directors=None, actors=None):

		# Media ID and year converted to string so they can be written to XML
		self.mid = str(mid) if mid else None
		self.title = title
		self.media_type = media_type
		self.year = str(year) if year else None
		self.genres = genres if genres else []
		self.directors = directors if directors else []
		self.actors = actors if actors else []

	def __repr__(self):
		return f"{self.title if self.title else 'Title not set'}" \
				+ f"\n\nMedia type: {self.media_type if self.media_type else 'not set'}" \
				+ f"\n\nYear: {self.year if self.year else 'not set'}" \
				+ f"\n\nGenres: {', '.join(self.genres) if self.genres else 'not set'}" \
				+ f"\n\nDirectors: {', '.join(self.directors) if self.directors else 'not set'}" \
				+ f"\n\nActors: {', '.join(self.actors) if self.actors else 'not set'}"
