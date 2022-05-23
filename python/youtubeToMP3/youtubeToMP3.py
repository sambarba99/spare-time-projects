# YouTube to mp3 converter and downloader
# Author: Sam Barba
# Created 26/03/2022

import os
from pytube import YouTube
import tkinter as tk
from tkinter import messagebox

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def download():
	global sv

	try:
		yt = YouTube(url=sv.get())
		audio = yt.streams.filter(only_audio=True).first()
		file = audio.download(output_path=os.path.expanduser("~") + "\\Desktop")
		dot_idx = file.rfind(".")
		new_file = file[:dot_idx] + ".mp3"  # Rename from .mp4 to .mp3

		if os.path.exists(new_file):
			# Delete file ending in .mp4 (it can't be renamed anyway)
			os.remove(file)
			messagebox.showerror(title="Error", message="File already exists")
		else:
			os.rename(file, new_file)
			messagebox.showinfo(title="Done", message=f"File saved: {new_file}")
	except Exception:
		messagebox.showerror(title="Error", message="Something went wrong, maybe due to bad URL")
	finally:
		sv.set("")

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == "__main__":
	root = tk.Tk()
	root.title("YouTube to mp3 converter and downloader")
	root.config(width=420, height=180, bg="#141414")
	root.eval("tk::PlaceWindow . center")

	frame = tk.Frame(root, bg="#0080ff")
	frame.place(relwidth=0.9, relheight=0.9, relx=0.5, rely=0.5, anchor="center")

	status_lbl = tk.Label(frame, text="Enter a video URL:", font="consolas", bg="#0080ff")
	status_lbl.place(relwidth=0.8, relheight=0.2, relx=0.5, rely=0.2, anchor="center")

	sv = tk.StringVar()
	entry_box = tk.Entry(frame, textvariable=sv, font="consolas", justify="center")
	entry_box.place(relwidth=0.8, relheight=0.18, relx=0.5, rely=0.43, anchor="center")

	btn_download = tk.Button(frame, text="Download", font="consolas", command=lambda: download())
	btn_download.place(relwidth=0.35, relheight=0.2, relx=0.5, rely=0.75, anchor="center")

	root.mainloop()
