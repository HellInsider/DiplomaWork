#pragma once

class Pixel
{
public:
	unsigned char R, G, B;
	Pixel() : R(0), G(0), B(0) {};
	Pixel(unsigned char R, unsigned char G, unsigned char B) : R(R), G(G), B(B) {};

	void zero()
	{
		R = 0;
		G = 0;
		B = 0;
	}

	void myMax()
	{
		R = 255;
		G = 255;
		B = 255;
	}

	int sum()
	{
		return R + G + B;
	}

	bool operator < (Pixel& b)
	{
		return R + G + B < b.R + b.G + b.B;
	}

	bool operator > (Pixel& b)
	{
		return R + G + B > b.R + b.G + b.B;
	}

	bool operator == (Pixel &b)
	{
		if (R == b.R && G == b.G && B == b.B)
			return true;
		return false;
	}

	bool operator != (Pixel &b)
	{
		if (R == b.R && G == b.G && B == b.B)
			return false;
		return true;
	}

	Pixel operator - (Pixel &c)
	{
		int r, g, b;
		r = R - c.R;
		if (r < 0) r = 0;
		g = G - c.G;
		if (g < 0) g = 0;
		b = B - c.B;
		if (b < 0) b = 0;

		return Pixel(r, g, b);
	}

	Pixel operator + (Pixel &c)
	{
		int r, g, b;
		r = R + c.R;
		if (r > 255) r = 255;
		g = G + c.G;
		if (g > 255) g = 255;
		b = B + c.B;
		if (b > 255) b = 255;

		return Pixel(r, g, b);
	}

	Pixel operator * (float c)
	{
		float r, g, b;
		int _r, _g, _b;

		r = R * c;
		if (r < 0) r = 0;
		if (r > 255) r = 255;
		_r = (int)(r);

		g = G * c;
		if (g < 0) g = 0;
		if (g > 255) g = 255;
		_g = (int)(g);

		b = B * c;
		if (b < 0) b = 0;
		if (b > 255) b = 255;
		_b = (int)(b);

		return Pixel(_r, _g, _b);
	}

	Pixel operator / (float c)
	{
		if(c == 0)
		{
			return Pixel(0, 0, 0);
		}

		float r, g, b;
		int _r, _g, _b;

		r = R / c;
		if (r < 0) r = 0;
		if (r > 255) r = 255;
		_r = (int)(r);

		g = G / c;
		if (g < 0) g = 0;
		if (g > 255) g = 255;
		_g = (int)(g);

		b = B / c;
		if (b < 0) b = 0;
		if (b > 255) b = 255;
		_b = (int)(b);

		return Pixel(_r, _g, _b);
	}

	friend Pixel* operator * (Pixel& a)
	{
		return &a;
	}

	void MulIntense(float a)
	{
		if (a < 0)
		{
			R = G = B = 0;
		}

		float _r = R, _g = G, _b = B;

		_r *= a;
		_g *= a;
		_b *= a;

		if (_r > 255) _r = 255;
		if (_g > 255) _g = 255;
		if (_b > 255) _b = 255;

		R = int(_r);
		G = int(_g);
		B = int(_b);
	}

};
