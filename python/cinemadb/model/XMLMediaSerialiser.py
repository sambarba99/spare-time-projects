# XML media serialiser for cinema DB
# Author: Sam Barba
# Created 15/04/2022

from model.Singleton import Singleton
from view.Constants import MEDIA_PATH
import os
import xml.etree.ElementTree as ET

@Singleton
class XMLMediaSerialiser:
	def __init__(self):
		if not os.path.exists(MEDIA_PATH):
			self.root = ET.Element("mediaList")
			self.__write_to_file()

		tree = ET.parse(MEDIA_PATH)
		self.root = tree.getroot()

	def read_all_media(self):
		# print(ET.tostring(self.root).decode())
		return self.root.findall("media")

	def get_media_element_by_id(self, media_id):
		return self.root.find(f".//media[@id='{media_id}']")

	def add_media(self, new_media):
		new_media_element = ET.SubElement(self.root, "media")
		new_media_element.set("id", new_media.mid)
		ET.SubElement(new_media_element, "title", value=new_media.title)
		ET.SubElement(new_media_element, "mediaType", value=new_media.media_type)
		ET.SubElement(new_media_element, "year", value=new_media.year)
		genres = ET.SubElement(new_media_element, "genres")
		for g in new_media.genres:
			ET.SubElement(genres, "genre", value=g)
		directors = ET.SubElement(new_media_element, "directors")
		for d in new_media.directors:
			ET.SubElement(directors, "director", name=d)
		actors = ET.SubElement(new_media_element, "actors")
		for a in new_media.actors:
			ET.SubElement(actors, "actor", name=a)

		self.__write_to_file()

	def update_media(self, media_id, new_media):
		media = self.get_media_element_by_id(media_id)

		if media is None: raise ValueError(f"Non-existent media ID: {media_id}")

		# Just delete and remake the media element, instead of updating each tag
		self.delete_media(media_id)
		self.add_media(new_media)

	def delete_media(self, media_id):
		media = self.get_media_element_by_id(media_id)

		if media is None: raise ValueError(f"Non-existent media ID: {media_id}")

		self.root.remove(media)
		self.__write_to_file()

	def __write_to_file(self):
		tree = ET.ElementTree(self.root)
		tree.write(MEDIA_PATH, encoding="UTF-8", xml_declaration=True)
