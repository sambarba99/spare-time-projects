"""
Generation maths for apollonian_gasket.py

Author: Sam Barba
Created 23/09/2022
"""

from circle import Circle
from cmath import sqrt

def tangent_circles_from_radii(r2, r3, r4):
	"""
	Takes 3 radii and calculates the corresponding externally tangent circles
	as well as a 4th one enclosing them. The enclosing circle is the first one.
	"""

	def outer_tangent_circle(circle1, circle2, circle3):
		"""Given 3 externally tangent circles, calculate the 4th enclosing one"""

		cur1 = circle1.curvature
		cur2 = circle2.curvature
		cur3 = circle3.curvature
		cen1 = circle1.centre
		cen2 = circle2.centre
		cen3 = circle3.centre
		cur4 = -2 * (cur1 * cur2 + cur2 * cur3 + cur1 * cur3) ** 0.5 + cur1 + cur2 + cur3
		c4 = (-2 * sqrt(cur1 * cen1 * cur2 * cen2 + cur2 * cen2 * cur3 * cen3 + cur1 * cen1 * cur3 * cen3)
			  + cur1 * cen1 + cur2 * cen2 + cur3 * cen3) / cur4
		circle4 = Circle(c4.real, c4.imag, 1 / cur4)

		return circle4

	circle2 = Circle(0, 0, r2)
	circle3 = Circle(r2 + r3, 0, r3)
	c4x = (r2 * r2 + r2 * r4 + r2 * r3 - r3 * r4) / (r2 + r3)
	c4y = sqrt((r2 + r4) * (r2 + r4) - c4x * c4x)
	circle4 = Circle(c4x, c4y, r4)
	circle1 = outer_tangent_circle(circle2, circle3, circle4)

	return circle1, circle2, circle3, circle4

def second_solution(fixed, c1, c2, c3):
	"""Given 4 tangent circles, calculate the other one that is tangent to the last 3"""

	cur_fixed = fixed.curvature
	cur1 = c1.curvature
	cur2 = c2.curvature
	cur3 = c3.curvature

	cur = 2 * (cur1 + cur2 + cur3) - cur_fixed
	c = (2 * (cur1 * c1.centre + cur2 * c2.centre + cur3 * c3.centre) - cur_fixed * fixed.centre) / cur
	return Circle(c.real, c.imag, 1 / cur)

class ApollonianGasketGenerator:
	def __init__(self, r1, r2, r3):
		"""
		Creates a basic Apollonian gasket with 4 circles.

		r1, r2, r3: The radii of the three inner circles of the starting set
		(i.e. depth 0 of the recursion). The 4th enclosing circle is calculated from them.
		"""

		self.start = tangent_circles_from_radii(r1, r2, r3)
		self.gen_circles = list(self.start)

	def recurse(self, circles, depth, max_depth):
		"""Recursively calculate the smaller circles of the gasket up to the given depth"""

		if depth == max_depth: return

		c1, c2, c3, c4 = circles

		if depth == 0:
			# First recursive step, the only time we need to calculate 4 new circles
			c_special = second_solution(c1, c2, c3, c4)
			self.gen_circles.append(c_special)
			self.recurse((c_special, c2, c3, c4), 1, max_depth)

		cn2 = second_solution(c2, c1, c3, c4)
		self.gen_circles.append(cn2)
		cn3 = second_solution(c3, c1, c2, c4)
		self.gen_circles.append(cn3)
		cn4 = second_solution(c4, c1, c2, c3)
		self.gen_circles.append(cn4)

		self.recurse((cn2, c1, c3, c4), depth + 1, max_depth)
		self.recurse((cn3, c1, c2, c4), depth + 1, max_depth)
		self.recurse((cn4, c1, c2, c3), depth + 1, max_depth)

	def generate(self, depth):
		"""Wrapper for the recurse function (generates the gasket)"""
		self.recurse(self.start, 0, depth)
		return self.gen_circles
