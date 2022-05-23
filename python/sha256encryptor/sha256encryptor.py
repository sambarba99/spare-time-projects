# SHA-256 encryptor
# Author: Sam Barba
# Created 30/03/2022

import hashlib
import tkinter as tk

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def encrypt(*args):
	global sv, output_txt

	hash_obj = hashlib.sha256(sv.get().encode())
	hash_dig = hash_obj.hexdigest()

	output_txt.config(state="normal")
	output_txt.delete("1.0", tk.END)
	output_txt.insert("1.0", f"'{sv.get()}' encrypted with SHA-256:\n{hash_dig}")
	output_txt.tag_add("center", "1.0", tk.END)
	output_txt.config(state="disabled")

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == "__main__":
	root = tk.Tk()
	root.title("SHA-256 encryptor")
	root.config(width=500, height=250, bg="#141414")
	root.eval("tk::PlaceWindow . center")

	frame = tk.Frame(root, bg="#0080ff")
	frame.place(relwidth=0.9, relheight=0.9, relx=0.5, rely=0.5, anchor="center")

	enter_txt_lbl = tk.Label(frame, text="Enter some text:", font="consolas", bg="#0080ff")
	enter_txt_lbl.place(relwidth=0.5, relheight=0.12, relx=0.5, rely=0.12, anchor="center")

	sv = tk.StringVar()
	sv.trace_add(mode="write", callback=encrypt)

	plaintext_entry = tk.Entry(frame, textvariable=sv,font="consolas", justify="center")
	plaintext_entry.place(relwidth=0.9, relheight=0.12, relx=0.5, rely=0.25, anchor="center")

	output_lbl = tk.Label(frame, text="Encrypted:", font="consolas", bg="#0080ff")
	output_lbl.place(relwidth=0.5, relheight=0.12, relx=0.5, rely=0.4, anchor="center")
	output_txt = tk.Text(frame, bg="#dcdcdc", font="consolas", state="disabled")
	output_txt.tag_configure("center", justify="center")
	output_txt.place(relwidth=0.9, relheight=0.4, relx=0.5, rely=0.66, anchor="center")

	encrypt()

	root.mainloop()
