"""
OpenCV masking/HSV demo

Author: Sam Barba
Created 23/04/2023
"""

import tkinter as tk

import cv2 as cv
import numpy as np


PATH = 'parrot.png'


def change_hsv():
	img = cv.imread(PATH)

	target_r = slider_target_r.get()
	target_g = slider_target_g.get()
	target_b = slider_target_b.get()
	target_bgr = np.array([target_b, target_g, target_r])

	delta = slider_delta.get()
	h_change = slider_hue.get()
	s_change = slider_sat.get()
	v_change = slider_val.get()

	# Create upper/lower ranges for mask using target_bgr and delta
	lo_range = np.clip(target_bgr - delta, 0, 255)
	hi_range = np.clip(target_bgr + delta, 0, 255)

	# Create mask
	mask = cv.inRange(img, lo_range, hi_range)
	bool_mask = mask.astype(bool)

	# Change HSV values
	hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
	h, s, v = cv.split(hsv)
	h, s, v = (i.astype(np.float32) for i in (h, s, v))  # To support addition
	h[bool_mask] = np.clip(h[bool_mask] + h_change, 0, 255)
	s[bool_mask] = np.clip(s[bool_mask] + s_change, 0, 255)
	v[bool_mask] = np.clip(v[bool_mask] + v_change, 0, 255)
	h, s, v = (i.astype(np.uint8) for i in (h, s, v))

	result_hsv = cv.merge((h, s, v))
	result_img = cv.cvtColor(result_hsv, cv.COLOR_HSV2BGR)

	cv.imshow('Original', img)
	cv.imshow('Target BGR mask', mask)
	cv.imshow('Result', result_img)
	# cv.imwrite('mask.png', mask)
	# cv.imwrite('result.png', result_img)


if __name__ == '__main__':
	root = tk.Tk()
	root.title('OpenCV masking/HSV demo')
	root.config(width=350, height=550, bg='#000024')
	root.eval('tk::PlaceWindow . center')
	root.resizable(False, False)

	set_target_rgb_lbl = tk.Label(root, text='Set target RGB:', font='consolas', bg='#000024', fg='white')
	slider_target_r = tk.Scale(
		root, from_=0, to=255, resolution=1, orient='horizontal', font='consolas',
		command=lambda _: change_hsv()
	)
	slider_target_g = tk.Scale(
		root, from_=0, to=255, resolution=1, orient='horizontal', font='consolas',
		command=lambda _: change_hsv()
	)
	slider_target_b = tk.Scale(
		root, from_=0, to=255, resolution=1, orient='horizontal', font='consolas',
		command=lambda _: change_hsv()
	)
	slider_target_b.set(255)

	set_delta_lbl = tk.Label(root, text='Set delta:', font='consolas', bg='#000024', fg='white')
	slider_delta = tk.Scale(
		root, from_=0, to=255, resolution=1, orient='horizontal', font='consolas',
		command=lambda _: change_hsv()
	)
	slider_delta.set(128)

	set_hsv_change = tk.Label(root, text='Set HSV change:', font='consolas', bg='#000024', fg='white')
	slider_hue = tk.Scale(
		root, from_=-255, to=255, resolution=1, orient='horizontal', font='consolas',
		command=lambda _: change_hsv()
	)
	slider_sat = tk.Scale(
		root, from_=-255, to=255, resolution=1, orient='horizontal', font='consolas',
		command=lambda _: change_hsv()
	)
	slider_val = tk.Scale(
		root, from_=-255, to=255, resolution=1, orient='horizontal', font='consolas',
		command=lambda _: change_hsv()
	)
	slider_hue.set(128)
	slider_val.set(-64)

	set_target_rgb_lbl.place(width=150, height=30, relx=0.5, y=40, anchor='center')
	slider_target_r.place(width=200, height=50, relx=0.5, y=80, anchor='center')
	slider_target_g.place(width=200, height=50, relx=0.5, y=135, anchor='center')
	slider_target_b.place(width=200, height=50, relx=0.5, y=190, anchor='center')
	set_delta_lbl.place(width=150, height=30, relx=0.5, y=240, anchor='center')
	slider_delta.place(width=200, height=50, relx=0.5, y=280, anchor='center')
	set_hsv_change.place(width=150, height=30, relx=0.5, y=330, anchor='center')
	slider_hue.place(width=200, height=50, relx=0.5, y=370, anchor='center')
	slider_sat.place(width=200, height=50, relx=0.5, y=425, anchor='center')
	slider_val.place(width=200, height=50, relx=0.5, y=480, anchor='center')

	change_hsv()

	root.mainloop()
