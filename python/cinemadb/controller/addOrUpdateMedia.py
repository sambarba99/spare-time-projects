# Cinema DB add/update media page
# Author: Sam Barba
# Created 22/04/2022

from model.Media import Media
from model.MediaService import MediaService
from view.Constants import GENRES, MEDIA_TYPES
import re
import tkinter as tk
from tkinter import messagebox

class AddOrUpdateMediaPage(tk.Toplevel):
	def __init__(self, *, parent, media=None):
		super().__init__(parent)

		self.parent = parent
		self.media = media if media else Media()
		self.media_service = MediaService.get_instance()

		self.title("Update Media" if self.media.mid else "Add Media")
		self.config(width=820, height=500, bg="#000045")
		self.resizable(False, False)

		self.sv_title = tk.StringVar(value=self.media.title)
		self.sv_media_type = tk.StringVar(value=self.media.media_type)
		self.sv_year = tk.StringVar(value=self.media.year)
		self.sv_genre = tk.StringVar()
		self.sv_director = tk.StringVar()
		self.intv_various_directors = tk.IntVar(value=1 if self.media.directors == ["Various"] else 0)
		self.sv_actor = tk.StringVar()
		self.sv_title.trace_add(mode="write", callback=self.__update_media_and_text_area)
		self.sv_media_type.trace_add(mode="write", callback=self.__update_media_and_text_area)
		self.sv_year.trace_add(mode="write", callback=self.__update_media_and_text_area)
		self.sv_genre.trace_add(mode="write", callback=self.__update_media_and_text_area)

		self.media_txt = tk.Text(self, bg="white", font="arial 12", state="disabled")
		self.__setup_window()

		self.transient(self.parent)
		self.grab_set()
		self.parent.wait_window(self)

		self.mainloop()

	def __setup_window(self):
		lbl_text = "Media to update:" if self.media.mid else "Media to add:"
		media_lbl = tk.Label(self, text=lbl_text, font="arial 12", bg="#000045", fg="white")
		media_lbl.place(width=130, height=30, x=82 if self.media.mid else 71, y=241, anchor="center")

		self.media_txt.place(width=640, height=220, x=342, y=368, anchor="center")

		title_lbl = tk.Label(self, text="Title:", font="arial 12", bg="#000045", fg="white")
		title_lbl.place(width=80, height=30, x=50, y=30, anchor="center")
		title_entry = tk.Entry(self, textvariable=self.sv_title, font="arial 12")
		title_entry.place(width=220, height=30, x=143, y=61, anchor="center")

		media_type_lbl = tk.Label(self, text="Media type:", font="arial 12", bg="#000045", fg="white")
		media_type_lbl.place(width=100, height=30, x=75, y=101, anchor="center")
		media_type_selection = tk.OptionMenu(self, self.sv_media_type, *MEDIA_TYPES)
		media_type_selection.place(width=220, height=30, x=144, y=132, anchor="center")

		year_lbl = tk.Label(self, text="Year:", font="arial 12", bg="#000045", fg="white")
		year_lbl.place(width=80, height=30, x=52, y=169, anchor="center")
		year_entry = tk.Entry(self, textvariable=self.sv_year, font="arial 12")
		year_entry.place(width=220, height=30, x=143, y=200, anchor="center")

		genres_lbl = tk.Label(self, text="Genres:", font="arial 12", bg="#000045", fg="white")
		genres_lbl.place(width=100, height=30, x=328, y=30, anchor="center")
		genre_selection = tk.OptionMenu(self, self.sv_genre, *GENRES)
		genre_selection.place(width=220, height=30, x=410, y=61, anchor="center")
		btn_clear_genres = tk.Button(self, text="Clear all", font="arial 12 bold", bg="#0099ff",
			fg="white", command=lambda: self.__clear_genres())
		btn_clear_genres.place(width=100, height=30, x=588, y=61, anchor="center")

		directors_lbl = tk.Label(self, text="Directors:", font="arial 12", bg="#000045", fg="white")
		directors_lbl.place(width=100, height=30, x=334, y=101, anchor="center")
		director_entry = tk.Entry(self, textvariable=self.sv_director, font="arial 12")
		director_entry.place(width=220, height=30, x=410, y=132, anchor="center")
		btn_add_director = tk.Button(self, text="Add", font="arial 12 bold", bg="#0099ff", fg="white",
			command=lambda: self.__add_director_or_actor(lst=self.media.directors, sv=self.sv_director))
		btn_add_director.place(width=70, height=30, x=573, y=132, anchor="center")
		various_directors_check = tk.Checkbutton(self, text="Various", font="arial 12",
			bg="#000045", fg="white", variable=self.intv_various_directors,
			onvalue=1, offvalue=0, command=lambda: self.__set_various_directors())
		various_directors_check.config(selectcolor="black")
		various_directors_check.place(width=100, height=30, x=660, y=132, anchor="center")

		actors_lbl = tk.Label(self, text="Actors:", font="arial 12", bg="#000045", fg="white")
		actors_lbl.place(width=100, height=30, x=327, y=169, anchor="center")
		actor_entry = tk.Entry(self, textvariable=self.sv_actor, font="arial 12")
		actor_entry.place(width=220, height=30, x=410, y=200, anchor="center")
		btn_add_actor = tk.Button(self, text="Add", font="arial 12 bold", bg="#0099ff", fg="white",
			command=lambda: self.__add_director_or_actor(lst=self.media.actors, sv=self.sv_actor))
		btn_add_actor.place(width=70, height=30, x=573, y=200, anchor="center")

		btn_txt = "Update\nmedia" if self.media.mid else "Add\nmedia"
		btn_add_update_media = tk.Button(self, text=btn_txt, font="arial 12 bold",
			bg="#0099ff", fg="white", command=lambda: self.__add_update_media())
		btn_add_update_media.place(width=120, height=60, x=738, y=332, anchor="center")

		btn_clear_all = tk.Button(self, text="Clear all\nattributes", font="arial 12 bold",
			bg="#ff7000", fg="white", command=lambda: self.__clear_all_attributes())
		btn_clear_all.place(width=120, height=60, x=738, y=407, anchor="center")

		self.__update_media_and_text_area()

	def __clear_genres(self):
		self.media.genres = []
		self.sv_genre.set("")
		self.__update_media_and_text_area()

	def __add_director_or_actor(self, *, lst, sv):
		if lst is self.media.directors and self.intv_various_directors.get() == 1:
			messagebox.showerror(title="Error", message="Uncheck 'Various' to add director(s) manually")
		elif sv.get():
			lst.append(sv.get())
			sv.set("")
			self.__update_media_and_text_area()
		else:
			messagebox.showerror(title="Error", message="Enter a name")

	def __set_various_directors(self):
		if self.intv_various_directors.get() == 0:
			self.media.directors = []
		else:  # 1
			self.media.directors = ["Various"]
		self.__update_media_and_text_area()

	def __clear_all_attributes(self):
		self.sv_title.set("")
		self.sv_media_type.set("")
		self.sv_year.set("")
		self.sv_genre.set("")
		self.sv_director.set("")
		self.intv_various_directors.set(0)
		self.sv_actor.set("")
		mid = self.media.mid
		self.media = Media()
		self.media.mid = mid
		self.__update_media_and_text_area()

	def __add_update_media(self):
		# Verify each media attribute
		for attr, val in self.media.__dict__.items():
			if not val and attr != "mid":
				messagebox.showerror(title="Error",
					message=f"Media is missing attribute '{attr.replace('_', ' ')}'")
				return
			if attr == "year" and re.match("^\\d{4}$", val) is None:
				messagebox.showerror(title="Error", message=f"Year must be a 4-digit number")
				return

		if self.media.mid is None:
			self.media_service.add_media(self.media)
			messagebox.showinfo(title="Success",
				message=f"Successfully added {self.media.media_type.lower()} '{self.media.title}'")
		else:
			self.media_service.update_media(self.media.mid, self.media)
			messagebox.showinfo(title="Success",
				message=f"Successfully updated {self.media.media_type.lower()} '{self.media.title}'")

		self.parent.update_table_with_filtered_results()
		self.destroy()

	def __update_media_and_text_area(self, *args):
		self.media.title = self.sv_title.get()
		self.media.media_type = self.sv_media_type.get()
		self.media.year = self.sv_year.get()

		if self.sv_genre.get():
			self.media.genres.append(self.sv_genre.get())
		self.media.genres = sorted(list(set(self.media.genres)))

		self.media_txt.config(state="normal")
		self.media_txt.delete("1.0", tk.END)
		self.media_txt.insert("1.0", self.media)
		self.media_txt.config(state="disabled")
