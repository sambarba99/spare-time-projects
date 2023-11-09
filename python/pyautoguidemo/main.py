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
	write_msg(f'calculator answer: {a} {op} {b} = ', press_enter_after=False)
	pag.hotkey('ctrl', 'v')
	pag.hotkey('alt', 'f4')
	pag.hotkey('right')
	pag.hotkey('enter')


def facebook_messenger_test():
	pag.hotkey('win')
	write_msg('firefox')
	pag.click(450, 60, duration=1)
	write_msg('facebook.com/messages')
	pag.click(120, 330, duration=4)
	pag.click(640, 1010, duration=2)

	for i in range(1, 11):
		msg = 'hi ' * i
		write_msg(msg)


def whatsapp_web_test():
	"""First, ensure whatsapp web is active via phone"""

	pag.hotkey('win')
	write_msg('firefox')
	pag.click(450, 60, duration=1)
	write_msg('web.whatsapp.com')
	pag.click(400, 210, duration=4)
	write_msg('person or group name here', press_enter_after=False)
	pag.click(400, 330, duration=1)

	for i in range(1, 6):
		write_msg(f'hello{i}')


def wolfram_alpha_test():
	pag.hotkey('win')
	write_msg('firefox')
	pag.click(450, 60, duration=1)
	write_msg('wolframalpha.com')
	pag.click(700, 320, duration=4)
	write_msg('tell me a joke')


def write_msg(msg, total_duration=1, press_enter_after=True):
	pag.write(msg, interval=total_duration / len(msg))
	if press_enter_after:
		pag.hotkey('enter')


if __name__ == '__main__':
	calculator_test()
	# facebook_messenger_test()
	# whatsapp_web_test()
	wolfram_alpha_test()
