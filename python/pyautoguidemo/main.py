"""
PyAutoGUI demo

Author: Sam Barba
Created 16/02/2022
"""

import numpy as np
import pyautogui as pag


pag.PAUSE = 1  # 1 sec pause between PyAutoGUI calls


def calculator_test():
	pag.hotkey('win')
	write_msg('calculator')
	a, b = np.random.randint(100, 1000, size=2)
	op = np.random.choice(['+', '-', '*', '/'])
	write_msg(f'{a}{op}{b}')
	pag.hotkey('ctrl', 'c')
	pag.hotkey('alt', 'f4')
	pag.hotkey('win')
	write_msg('notepad')
	pag.hotkey('ctrl', 'n')
	write_msg('calculator answer = ', press_enter_after=False)
	pag.hotkey('ctrl', 'v')


def facebook_messenger_test():
	pag.hotkey('win')
	write_msg('google chrome')
	pag.click(300, 50, duration=0.5)
	write_msg('facebook.com')
	pag.click(120, 300, duration=1.5)
	pag.click(120, 235, duration=1.5)
	write_msg('person or group name here', press_enter_after=False)
	pag.click(120, 330, duration=0.5)
	pag.click(640, 1010, duration=0.5)

	pag.PAUSE = 0.1
	for i in range(1, 6):
		write_msg(f'{i}^{i} = {i**i}')


def whatsapp_web_test():  # Ensure whatsapp web is active via phone
	pag.hotkey('win')
	write_msg('google chrome')
	pag.click(300, 50, duration=0.5)
	write_msg('web.whatsapp.com')
	pag.click(400, 205, duration=2)
	write_msg('person or group name here', press_enter_after=False)
	pag.click(400, 330, duration=0.5)

	pag.PAUSE = 0.1
	for i in range(1, 6):
		write_msg(f'hello{i}')


def wolfram_alpha_test():
	pag.hotkey('win')
	write_msg('google chrome')
	pag.click(300, 50, duration=0.5)
	write_msg('wolframalpha.com')
	pag.click(910, 315, duration=0.5)
	pag.moveRel(0, -50, duration=0.5)
	write_msg('tell me a joke')


def write_msg(msg, total_duration=0.5, press_enter_after=True):
	gap_between_chars = total_duration / len(msg)
	pag.write(msg, interval=gap_between_chars)
	if press_enter_after:
		pag.hotkey('enter')


if __name__ == '__main__':
	calculator_test()
	#facebook_messenger_test()
	#whatsapp_web_test()
	wolfram_alpha_test()
