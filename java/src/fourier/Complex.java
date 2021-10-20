package fourier;

/**
 * Complex number
 * 
 * @author Sam Barba
 */
public class Complex {

	double re;

	double im;

	double freq;

	public Complex(double re, double im) {
		this.re = re;
		this.im = im;
	}

	public Complex add(Complex c) {
		double newRe = re + c.getRe();
		double newIm = im + c.getIm();
		return new Complex(newRe, newIm);
	}

	public Complex mult(Complex c) {
		double newRe = re * c.getRe() - im * c.getIm();
		double newIm = re * c.getIm() + im * c.getRe();
		return new Complex(newRe, newIm);
	}

	public double getRe() {
		return re;
	}

	public void setRe(double re) {
		this.re = re;
	}

	public double getIm() {
		return im;
	}

	public void setIm(double im) {
		this.im = im;
	}

	public double getFreq() {
		return freq;
	}

	public void setFreq(double freq) {
		this.freq = freq;
	}

	public double getAmp() {
		return Math.sqrt(re * re + im * im);
	}

	public double getPhase() {
		return Math.atan2(im, re);
	}
}
