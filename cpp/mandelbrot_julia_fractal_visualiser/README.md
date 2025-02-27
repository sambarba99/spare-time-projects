## Mandelbrot/Julia Set visualiser

<h3 align="center">Iterative definition</h3>

The Mandelbrot set is defined as the set of all complex numbers $c$ for which the function:

$$z = z^2 + c$$

does not diverge when iterated (starting at the origin, $z_0 = 0 + 0\mathbf{i}$). I.e., $c$ belongs to the Mandelbrot set if the sequence remains bounded.

Julia sets are defined by the same function:

$$z = z^2 + c$$

but here, $c$ is fixed, and the initial value $z_0$ varies. Similarly, a point $z_0$ belongs to the Julia set for a given $c$ if the sequence remains bounded.

<h3 align="center">Examples of different coordinates/zoom levels on the Mandelbrot set</h3>

<p align="center">
	<img src="images/mandelbrot_set_1x.png"/>
	<br/>
	<img src="images/mandelbrot_set_512x.png"/>
	<br/>
	<img src="images/mandelbrot_set_65536x.png"/>
</p>

<h3 align="center">Examples of Julia sets for different values of $c$</h3>

$c = 0.28 + 0.008\mathbf{i}$

<p align="center">
	<img src="images/julia_set_0.28_0.008i.png"/>
</p>

$c$ interpolated from $-0.8 + 0.16\mathbf{i}$ to $-0.75 + 0.11\mathbf{i}$

<p align="center">
	<img src="images/julia_set_interpolation.webp"/>
</p>
