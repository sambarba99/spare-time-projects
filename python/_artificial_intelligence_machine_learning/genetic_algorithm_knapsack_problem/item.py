"""
Item class for GA demo

Author: Sam Barba
Created 2021-09-17
"""

from dataclasses import dataclass


@dataclass
class Item:
	index: int
	weight: float
	value: float
