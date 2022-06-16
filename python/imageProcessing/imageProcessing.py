"""
Image Processing demo

Author: Sam Barba
Created 23/09/2021
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import messagebox

IMG_SRCS = [f'test{i}.jpg' for i in range(1, 5)]
MAX_SIZE = 600
MAX_DIST = 441.673  # Max Euclidean dist between 2 colours = root(255^2 * 3)

selected_img = target_r_entry = target_g_entry = target_b_entry = None

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def select_img(img):
	"""Display selected img and its histogram"""
	global selected_img

	selected_img = img
	r, g, b = np.array(selected_img.getdata()).T
	r_count = np.bincount(r, minlength=256)
	g_count = np.bincount(g, minlength=256)
	b_count = np.bincount(b, minlength=256)

	ax1 = plt.subplot()
	ax1.imshow(selected_img)
	plt.figure()
	ax2 = plt.subplot()
	ax2.plot(r_count, color='#ff0000', linewidth=1, label='R')
	ax2.plot(g_count, color='#008000', linewidth=1, label='G')
	ax2.plot(b_count, color='#0000ff', linewidth=1, label='B')
	ax2.set_xlabel('RGB value')
	ax2.set_ylabel('Count')
	ax2.set_title('Histogram for selected image')
	ax2.legend()
	plt.show()

def binary_image():
	global selected_img

	if selected_img is None:
		messagebox.showerror(title='Error', message='Please select an image')
		return

	width, height = selected_img.size

	new_img = Image.new('RGB', (width, height))

	for x in range(width):
		for y in range(height):
			d = dist(selected_img.getpixel((x, y)), (255, 255, 255))
			new_pixel = (255, 255, 255) if d < MAX_DIST / 2 else (0, 0, 0)
			new_img.putpixel((x, y), new_pixel)

	plt.imshow(new_img)
	plt.show()

def nearest_colour():
	global selected_img, target_r_entry, target_g_entry, target_b_entry

	if selected_img is None:
		messagebox.showerror(title='Error', message='Please select an image')
		return

	target_r = target_r_entry.get()
	target_g = target_g_entry.get()
	target_b = target_b_entry.get()

	target_r = 0 if target_r == '' else int(target_r)
	target_g = 0 if target_g == '' else int(target_g)
	target_b = 0 if target_b == '' else int(target_b)

	width, height = selected_img.size

	closest_dist = dist(selected_img.getpixel((0, 0)), (target_r, target_g, target_b))
	r_best, g_best, b_best = selected_img.getpixel((0, 0))
	x_best = y_best = 0

	for x in range(width):
		for y in range(height):
			d = dist(selected_img.getpixel((x, y)), (target_r, target_g, target_b))

			if d < closest_dist:
				closest_dist = d
				r_best, g_best, b_best = selected_img.getpixel((x, y))
				x_best, y_best = x, y

	# Draw lines pointing to pixel with most similar colour
	new_img = selected_img.copy()
	for x in range(width):
		if abs(x - x_best) > 3:
			new_img.putpixel((x, y_best), (255, 0, 0))
	for y in range(height):
		if abs(y - y_best) > 3:
			new_img.putpixel((x_best, y), (255, 0, 0))

	percentage_match = 100 * (1 - (closest_dist / MAX_DIST) ** 0.5)

	plt.imshow(new_img)
	plt.title(f'Best RGB = {r_best} {g_best} {b_best}  ({percentage_match:.2f} % match)')
	plt.show()

def dist(colour1, colour2):
	"""Euclidean distance between 2 colours"""
	return np.linalg.norm(np.array(colour1) - np.array(colour2))

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def main():
	global target_r_entry, target_g_entry, target_b_entry

	imgs = [Image.open(src) for src in IMG_SRCS]
	for idx, img in enumerate(imgs):
		width, height = img.size

		if max(width, height) > MAX_SIZE:
			new_width = MAX_SIZE if width > height else round(width / height * MAX_SIZE)
			new_height = MAX_SIZE if height > width else round(height / width * MAX_SIZE)
			imgs[idx] = img.resize((new_width, new_height))

	root = tk.Tk()
	root.title('Image processing demo')
	root.config(width=500, height=300, bg='#000045')
	root.eval('tk::PlaceWindow . center')

	btn_select_img1 = tk.Button(root, text='Select image 1', font='consolas', command=lambda: select_img(imgs[0]))
	btn_select_img2 = tk.Button(root, text='Select image 2', font='consolas', command=lambda: select_img(imgs[1]))
	btn_select_img3 = tk.Button(root, text='Select image 3', font='consolas', command=lambda: select_img(imgs[2]))
	btn_select_img4 = tk.Button(root, text='Select image 4', font='consolas', command=lambda: select_img(imgs[3]))
	btn_to_binary_img = tk.Button(root, text='Convert this image to binary', font='consolas',
		command=lambda: binary_image())
	btn_find_nearest_colour = tk.Button(root, text='Find nearest RGB in image', font='consolas',
		command=lambda: nearest_colour())
	btn_select_img1.place(relwidth=0.4, relheight=0.12, relx=0.28, rely=0.19, anchor='center')
	btn_select_img2.place(relwidth=0.4, relheight=0.12, relx=0.72, rely=0.19, anchor='center')
	btn_select_img3.place(relwidth=0.4, relheight=0.12, relx=0.28, rely=0.34, anchor='center')
	btn_select_img4.place(relwidth=0.4, relheight=0.12, relx=0.72, rely=0.34, anchor='center')
	btn_to_binary_img.place(relwidth=0.65, relheight=0.12, relx=0.5, rely=0.51, anchor='center')
	btn_find_nearest_colour.place(relwidth=0.65, relheight=0.12, relx=0.5, rely=0.66, anchor='center')

	enter_rgb_lbl = tk.Label(root, text='(target RGB =                   )',
		font='consolas', bg='#000045', fg='white')
	enter_rgb_lbl.place(relwidth=0.7, relheight=0.12, relx=0.5, rely=0.82, anchor='center')

	target_r_entry = tk.Entry(root, font='consolas', justify='center')
	target_g_entry = tk.Entry(root, font='consolas', justify='center')
	target_b_entry = tk.Entry(root, font='consolas', justify='center')
	target_r_entry.place(relwidth=0.1, relheight=0.1, relx=0.5, rely=0.82, anchor='center')
	target_g_entry.place(relwidth=0.1, relheight=0.1, relx=0.61, rely=0.82, anchor='center')
	target_b_entry.place(relwidth=0.1, relheight=0.1, relx=0.72, rely=0.82, anchor='center')

	root.mainloop()

if __name__ == '__main__':
	main()
