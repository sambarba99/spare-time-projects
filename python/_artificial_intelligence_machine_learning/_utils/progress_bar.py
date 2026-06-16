"""
Progress Bar class

Author: Sam Barba
Created 2026-06-23
"""

import sys
import time


GREEN = '\033[38;2;0;240;0m'
RESET = '\033[0m'


class ProgressBar:
	def __init__(self, iterable, desc=None, unit='it', auto_finish=True, length=50):
		self.iterable = iterable
		self.total = len(iterable)
		self.desc = desc
		self.unit = unit
		self.auto_finish = auto_finish
		self.length = length

		self.done_count = 0
		self.postfix = ''
		self.start_time = time.time()
		self.finished = False
		self._draw()

	def __iter__(self):
		try:
			for i in self.iterable:
				yield i
				self.done_count += 1
				self._draw()
		finally:
			if self.auto_finish:
				self.finish()

	def set_postfix(self, text):
		self.postfix = text
		self._draw()

	def finish(self, postfix=None):
		if self.finished:
			return
		if postfix:
			self.postfix = postfix
			self._draw()
		print()
		self.finished = True

	def _format_time(self, secs):
		hours, rem = divmod(round(secs), 3600)
		mins, secs = divmod(rem, 60)
		ret = f'{mins:02d}:{secs:02d}'
		if hours:
			ret = f'{hours:02d}:{ret}'
		return ret

	def _draw(self):
		elapsed = time.time() - self.start_time

		progress = self.done_count / self.total
		if progress == 1:
			bar = '█' * self.length
			eta_str = '00:00'
		else:
			sub_units = progress * self.length * 10
			full_blocks, partial = divmod(round(sub_units), 10)
			partial_chr = str(partial) if partial > 0 else '-'
			bar = '█' * full_blocks + partial_chr + '-' * (self.length - full_blocks - 1)

			eta = max(elapsed / progress - elapsed, 0) if progress > 0 else None
			eta_str = '?' if eta is None else self._format_time(eta)

		iters_per_sec = self.done_count / elapsed if elapsed > 0 else 0
		if iters_per_sec == 0:
			rate = f'?{self.unit}/s'
		elif iters_per_sec >= 1:
			rate = f'{iters_per_sec:.1f}{self.unit}/s'
		else:
			rate = f'{(1 / iters_per_sec):.1f}s/{self.unit}'

		line = (
			f'{progress:>4.0%}|{bar}| '
			f'[{self.done_count}/{self.total}, {self._format_time(elapsed)}<{eta_str}, {rate}] '
			f'{self.postfix}{RESET}'
		)
		sys.stdout.write((f'\r{GREEN}{self.desc}: ' if self.desc else f'\r{GREEN}') + line)
		sys.stdout.flush()
