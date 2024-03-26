"""
Item class for GA demo

Author: Sam Barba
Created 17/09/2021
"""

from dataclasses import dataclass


@dataclass
class Item:
	index: int
	weight: float
	value: float
