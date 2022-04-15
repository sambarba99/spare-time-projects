# Cinema DB main page
# Author: Sam Barba
# Created 23/04/2022

from model.MediaService import MediaService
from view.Constants import TABLE_COLS, GENRES, MEDIA_TYPES
from controller.addOrUpdateMedia import AddOrUpdateMediaPage
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

class MainPage(tk.Tk):
	def __init__(self):
		super().__init__()

		self.selected_media = None
		self.media_service = MediaService.get_instance()

		self.title("Cinema DB")
		self.configure(width=1400, height=790, bg="#141414")
		self.eval("tk::PlaceWindow . center")
		self.resizable(False, False)

		# For navigating table with up/down arrows
		self.bind("<Up>", self.__select_row)
		self.bind("<Down>", self.__select_row)

		self.num_media_lbl = tk.Label(self, font="arial 12", bg="#141414", fg="#dcdcdc")
		self.media_table = ttk.Treeview(self, columns=list(TABLE_COLS), show="headings")
		self.selected_media_txt = tk.Text(self, bg="#dcdcdc", font="arial 12", state="disabled")
		self.showing_genres_txt = tk.Text(self, bg="#dcdcdc", font="arial 12", state="disabled")
		self.genres_to_include = []

		self.sv_actor = tk.StringVar()
		self.sv_director = tk.StringVar()
		self.sv_genre = tk.StringVar(value="All")
		self.sv_media_type = tk.StringVar(value="All")
		self.sv_actor.trace_add(mode="write", callback=self.update_table_with_filtered_results)
		self.sv_director.trace_add(mode="write", callback=self.update_table_with_filtered_results)
		self.sv_genre.trace_add(mode="write", callback=self.update_table_with_filtered_results)
		self.sv_media_type.trace_add(mode="write", callback=self.update_table_with_filtered_results)

		self.__setup_window()

	def __setup_window(self):
		self.num_media_lbl.place(width=200, height=30, relx=0.5, y=33, anchor="center")

		selected_media_lbl = tk.Label(self, text="Selected media:", font="arial 12", bg="#141414", fg="#dcdcdc")
		selected_media_lbl.place(width=130, height=30, x=117, y=510, anchor="center")
		self.selected_media_txt.place(width=640, height=220, x=380, y=640, anchor="center")

		filter_lbl = tk.Label(self, text="Filter table by...", font="arial 12", bg="#141414", fg="#dcdcdc")
		by_actor_lbl = tk.Label(self, text="Actor:", font="arial 12", bg="#141414", fg="#dcdcdc")
		by_director_lbl = tk.Label(self, text="Director:", font="arial 12", bg="#141414", fg="#dcdcdc")
		by_genre_lbl = tk.Label(self, text="Genre:", font="arial 12", bg="#141414", fg="#dcdcdc")
		by_media_type_lbl = tk.Label(self, text="Media type:", font="arial 12", bg="#141414", fg="#dcdcdc")
		filter_lbl.place(width=120, height=30, x=790, y=510, anchor="center")
		by_actor_lbl.place(width=120, height=30, x=810, y=545, anchor="center")
		by_director_lbl.place(width=120, height=30, x=800, y=579, anchor="center")
		by_genre_lbl.place(width=120, height=30, x=808, y=613, anchor="center")
		by_media_type_lbl.place(width=120, height=30, x=790, y=648, anchor="center")

		actor_entry = tk.Entry(self, textvariable=self.sv_actor, font="arial 12")
		director_entry = tk.Entry(self, textvariable=self.sv_director, font="arial 12")
		genre_selection = tk.OptionMenu(self, self.sv_genre, *(["All"] + GENRES))
		media_type_selection = tk.OptionMenu(self, self.sv_media_type, *(["All"] + MEDIA_TYPES))
		actor_entry.place(width=220, height=30, x=950, y=545, anchor="center")
		director_entry.place(width=220, height=30, x=950, y=580, anchor="center")
		genre_selection.place(width=220, height=30, x=950, y=615, anchor="center")
		media_type_selection.place(width=220, height=30, x=950, y=650, anchor="center")

		self.showing_genres_txt.place(width=230, height=158, x=1203, y=585, anchor="center")

		btn_add_media = tk.Button(self, text="Add new media", font="arial 12 bold", bg="#0099ff",
			fg="white", command=lambda: AddOrUpdateMediaPage(parent=self))
		btn_update_media = tk.Button(self, text="Update\nselected media", font="arial 12 bold",
			bg="#0099ff", fg="white", command=lambda: self.__btn_update_media_click())
		btn_delete_media = tk.Button(self, text="Delete\nselected media", font="arial 12 bold",
			bg="#ff7000", fg="white", command=lambda: self.__btn_delete_media_click())
		btn_add_media.place(width=160, height=60, x=860, y=715, anchor="center")
		btn_update_media.place(width=160, height=60, x=1030, y=715, anchor="center")
		btn_delete_media.place(width=160, height=60, x=1200, y=715, anchor="center")

		self.__setup_table()
		self.__select_row()  # To set self.selected_media_txt to "None selected"

	def __btn_update_media_click(self):
		if self.selected_media is None:
			messagebox.showerror(title="Error", message="Please select 1 media item")
			return

		self.__select_row()
		AddOrUpdateMediaPage(parent=self, media=self.selected_media)

	def __btn_delete_media_click(self):
		if self.selected_media is None:
			messagebox.showerror(title="Error", message="Please select 1 media item")
			return

		title = self.selected_media.title
		media_type = self.selected_media.media_type.lower()

		confirm_delete = messagebox.askyesnocancel(title=f"Delete selected {media_type}?",
			message=f"Do you really wish to delete {media_type} '{title}'?")

		if confirm_delete:
			try:
				self.media_service.delete_media(self.selected_media.mid)
				self.update_table_with_filtered_results()
				messagebox.showinfo(title="Success", message=f"Successfully deleted {media_type} '{title}'")
			except Exception:
				messagebox.showerror(title="Error", message="Something went wrong :(")

	def __setup_table(self):
		style = ttk.Style()
		style.configure("Treeview.Heading", font=("Arial", 10, "bold"))

		self.media_table.bind("<Motion>", "break")  # Make columns non-resizable
		self.media_table.bind("<ButtonRelease-1>", self.__select_row)

		for col, width in TABLE_COLS.items():
			self.media_table.column(col, anchor=tk.CENTER, width=width)
			self.media_table.heading(col, text=col, command=lambda _col=col: self.__sort_column(_col))

		self.media_table.place(height=428, relx=0.5, y=265, anchor="center")

		self.update_table_with_filtered_results()

	def update_table_with_filtered_results(self, *args):
		self.media_table.delete(*self.media_table.get_children())
		self.selected_media = None
		self.__select_row()  # To set self.selected_media_txt to "None selected"

		if self.sv_genre.get() == "All":
			self.genres_to_include = ["All"]
		else:
			if "All" in self.genres_to_include:
				self.genres_to_include.remove("All")
			self.genres_to_include.append(self.sv_genre.get())

		if self.genres_to_include == ["All"]:
			genres_str = 12 * " " + "Showing all genres"
		else:
			self.genres_to_include = sorted(list(set(self.genres_to_include)))
			genres_str = 14 * " " + f"Showing genres:\n\n{',  '.join(self.genres_to_include)}"

		self.showing_genres_txt.configure(state="normal")
		self.showing_genres_txt.delete("1.0", tk.END)
		self.showing_genres_txt.insert("1.0", genres_str)
		self.showing_genres_txt.configure(state="disabled")

		media_rows = self.media_service.get_media_rows_with_filters(
			actor_substring=self.sv_actor.get(),
			director_substring=self.sv_director.get(),
			genres=self.genres_to_include,
			media_type=self.sv_media_type.get()
		)
		self.num_media_lbl.configure(text=f"Showing {len(media_rows)} media")

		for row in media_rows:
			self.media_table.insert("", tk.END, values=row)

	def __select_row(self, *args):
		try:
			selected_row = self.media_table.focus()
			media_id = self.media_table.item(selected_row)["values"][0]
			self.selected_media = self.media_service.get_media_by_id(media_id)
		except Exception:  # In case header is clicked instead of row
			self.selected_media = None
		finally:
			# Update text area with selected media
			self.selected_media_txt.configure(state="normal")
			self.selected_media_txt.delete("1.0", tk.END)
			self.selected_media_txt.insert(
				"1.0",
				str(self.selected_media) if self.selected_media else "None selected"
			)
			self.selected_media_txt.configure(state="disabled")

	def __sort_column(self, col, descending=False):
		l = [(self.media_table.set(k, col), k) for k in self.media_table.get_children("")]

		if col in ("ID", "Year"):
			# Convert column to int, then sort by it (then by tkinter row ID)
			l.sort(key=lambda item: (int(item[0]), item[1]), reverse=descending)
		elif col == "Title":
			# Sort by title (super pythonic way to remove leading 'the'), then by tkinter row ID
			l.sort(key=lambda item: (
				item[0].lower()[item[0].lower().startswith("the") and 3:].lstrip(),
				item[1]),
				reverse=descending)
		else:
			l.sort(reverse=descending)

		# Rearrange items into sorted positions
		for index, (val, k) in enumerate(l):
			self.media_table.move(k, "", index)

		# Reverse the sort order for next time
		self.media_table.heading(col, command=lambda _col=col: self.__sort_column(_col, not descending))
