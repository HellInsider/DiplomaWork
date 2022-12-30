#include "Tool.h"
#define HEIGHEST_POINT 8848.8
#define PI 3.14159265

void Tool::imageCopy(image_data* from, image_data* to)
{
	to->compPerPixel = from->compPerPixel;
	to->h = from->h;
	to->w = from->w;
	to->pixels = new stbi_uc[to->h * to->w * to->compPerPixel];

	for (int i = 0; i < to->h; i++)
		for (int j = 0; j < to->w; j++)
			for (int byte = 0; byte < to->compPerPixel; byte++)
				to->pixels[to->compPerPixel * (to->w * i + j) + byte] = from->pixels[from->compPerPixel * (from->w * i + j) + byte];

}

int Tool::getPixelIntense(image_data Data, int x, int y)
{
	int intense;
	if (Data.compPerPixel == 1)
	{
		intense = Data.pixels[(y * Data.w + x)];
	}
	else if (Data.compPerPixel == 3)
	{
		intense = Data.pixels[(y * Data.w + x) * Data.compPerPixel] +
				Data.pixels[(y * Data.w + x) * Data.compPerPixel + 1] +
				Data.pixels[(y * Data.w + x) * Data.compPerPixel + 2];
		intense /= 3;
	}

	return (intense);
}

void Tool::getPixel(image_data *Data, int x, int y, Pixel* pixel)
{
	if (Data->compPerPixel == 1)
	{
		Data->pixels[(y * Data->w + x)] = pixel->R;
	}
	else if (Data->compPerPixel == 3)
	{
		pixel->R = Data->pixels[(y * Data->w + x) * Data->compPerPixel];
		pixel->G = Data->pixels[(y * Data->w + x) * Data->compPerPixel + 1];
		pixel->B = Data->pixels[(y * Data->w + x) * Data->compPerPixel + 2];
	}
}

void Tool::setPixel(image_data Data, int x, int y, Pixel pixel)
{
	if (Data.compPerPixel == 1)
	{
		Data.pixels[(y * Data.w + x)] = pixel.R;
	}
	else if (Data.compPerPixel == 3)
	{
		Data.pixels[(y * Data.w + x) * Data.compPerPixel] = pixel.R;
		Data.pixels[(y * Data.w + x) * Data.compPerPixel + 1] = pixel.G;
		Data.pixels[(y * Data.w + x) * Data.compPerPixel + 2] = pixel.B;
	}

}

void Tool::prepareDirs(std::string prefolder)
{
	for (std::string _folder : Folders)
	{
		CreateDirectory(std::string(prefolder + "/" + _folder).c_str(), NULL);
	}
}

int Tool::convertToMeters(int pixSum, int compPerPix)
{
	return (int)((float)pixSum / (255.0 * compPerPix) * HEIGHEST_POINT);
}

void Tool::prepareDataset(Tool::MODE mode, std::string prefolder, int h, int w) //h, w - size of out images
{
	heightMapImg = heightMapHandler->getPixelData();
	coloredMapImg = coloredMapHandler->getPixelData();

	prepareDirs(prefolder);

	switch (mode)
	{
		case Tool::HEIGHTMAP: 
		{
			prepareDatasetHeighMaplBased(prefolder, h, w);
			break; 
		}

		case Tool::BOTHMAPS: 
		{
			prepareDatasetBothMapsBased(prefolder, h, w);
			break; 
		}
	}
}

Pixel Tool::getAveragePixel(image_data* data) 
{
	int r, g, b;
	r = g = b = 0;
	Pixel pix;

	for (int _h = 0; _h < data->h; _h++)
	{
		for (int _w = 0; _w < data->w; _w++)
		{
			getPixel(data, _w, _h, &pix);
			r += pix.R;
			g += pix.G;
			b += pix.B;
		}
	}

	pix.R = r / (data->h * data->w);
	pix.G = g / (data->h * data->w);
	pix.B = b / (data->h * data->w);

	return pix;
}


int Tool::checkOnHOG(image_data* origPic, bool needNormalise = true)
{

	image_data pic;
	imageCopy(origPic, &pic);

	if (needNormalise)
	{
		normalizePicture(&pic);
	}
	

	//Input data. Can be changed
	const int binsCount = 9;
	int kernelSize = 3;
	//End of input data.

	float gradX, gradY, angle, angleStep = 360.0f / binsCount;
	int hist[binsCount] = { 0 }, sum = 0, maxPercent = 0;
	Pixel pix1, pix2;

	for (int _h = 1; _h < pic.h - 1; _h++)
	{
		for (int _w = 1; _w < pic.w - 1; _w++)
		{
			getPixel(&pic, _w - 1, _h, &pix1);
			getPixel(&pic, _w + 1, _h, &pix2);
			gradX = pix2.sum() - pix1.sum();

			getPixel(&pic, _w, _h - 1, &pix1);
			getPixel(&pic, _w, _h + 1, &pix2);
			gradY = pix2.sum() - pix1.sum();

			if (gradX != 0)
			{
				//printf("gradx = %f, grady = %f\n", gradX, gradY);

				if (gradX > 0)
				{
					angle = atan(gradY / gradX) * 180.0 / PI;
				}
				else
				{
					angle = 180.0 - fabs(atan(gradY / gradX) * 180.0 / PI);
					if (gradY < 0)
						angle *= -1;
				}

				//printf("angle = %f\n", angle);
			}
			else
			{
				angle = 0;
			}

			int z = 0;
			while (z < binsCount && angle >= (z * angleStep - 180.0))
			{
				//printf("ancle = %f > %f\n", angle, (z * angleStep - 180.0));
				z++;
			}

			if (z - 1 < 0 || z - 1 >= binsCount)
			{
				printf("ERR!!!!!!!!!!!!!!!!!!");
			}

			hist[z - 1]++;
		}
	}

	for (int z = 0; z < binsCount; z++)
	{
		sum += hist[z];
	}

	if (sum == 0)
	{
		return 0;
	}

	for (int z = 0; z < binsCount; z++)
	{
		if (hist[z] * 100 / sum > maxPercent)
		{
			maxPercent = hist[z] * 100 / sum;
		}
		//printf("%i, ", hist[z]);						//For debug
	}

	//printf("\n");

	delete[] pic.pixels;

	return maxPercent;
}

//this method only uses height map
void Tool::prepareDatasetHeighMaplBased(std::string prefolder, int h, int w) //h, w - size of out images
{	
	std::string folder;
	png_toolkit imageWriter;
	image_data cutedImg;

	int skipFirstNPixStrings = 2000, skipLastNPixStrings = 2000; //0 to not skip. Must be multiple to out pic size
	int skipLeftNPix = 0, skipRightNPix = 0; 	//0 to not skip. Must be multiple to out pic size
	int basesNum = 2;		//на сколько частей дроблю

	for (int yOffset = 0; yOffset < basesNum; yOffset++)
	{
		for (int xOffset = 0; xOffset < basesNum; xOffset++)
		{
			
			for (int _h = skipFirstNPixStrings + yOffset * h / basesNum; _h + h <= heightMapImg.h - skipLastNPixStrings; _h += h)
			{
				printf("Progress %i/%i:", (xOffset + 1 + yOffset * basesNum), basesNum * basesNum);
				printf("   %i%%... \n", (int)(100 * ( (float)_h - skipFirstNPixStrings) / (heightMapImg.h - skipFirstNPixStrings - skipLastNPixStrings) ) );

				for (int _w = skipLeftNPix + xOffset * w / basesNum; _w + w <= heightMapImg.w - skipRightNPix; _w += w)
				{
					Pixel maxIntenceDIff, averagePix;
					cutImage(&heightMapImg, &cutedImg, h, w, _h, _w);
					maxIntenceDIff = findMaxIntenceDiff(&cutedImg, h, w, _h, _w);
					averagePix = getAveragePixel(&cutedImg);

					//normalizePicture(&cutedImg);

					int maxMetersDiff = convertToMeters(maxIntenceDIff.sum(), cutedImg.compPerPixel);
					int averagMeters = convertToMeters(maxIntenceDIff.sum(), cutedImg.compPerPixel);

					if (averagMeters <= 10)
						continue;

					if (maxMetersDiff <=5)
					{
						folder = Folders[1];	//ocean
					}
					else if (maxMetersDiff >= 1200)
					{
						//folder = Folders[2];	//mountains
						folder = Folders[2];	
					}
					else
					{
						folder = Folders[0];	//general
					}

					float averageIntense = getAverageIntense(&cutedImg);
					//printf("avInt = %f\n", averageIntense);
					int percent = checkOnHOG(&cutedImg, true);											//true or false - if normalise needed


					//std::cout << "Checking " << _h << "_" << _w << std::endl;			//For debug
 					std::string name = std::string(prefolder + "/" + folder + "/" + std::to_string(_h) + "_" + std::to_string(_w) + "_" + std::to_string(percent) + ".png");  //For debug
					/*std::string name = std::string(prefolder + "/" + folder + "/" + std::to_string(_h) + "_" + std::to_string(_w) + ".png");
					if ( !(folder == Folders[2] && percent > 70) && folder != Folders[1] && percent <= 90)
					{
						imageWriter.save(name);
					}*/
					imageWriter.imgData = RGBToOneChannel(&cutedImg);
					
					if (averageIntense > 25)
					{
						if (folder == Folders[0] && (percent < 60 || maxMetersDiff > 10) )
						{
							imageWriter.save(std::string(prefolder + "/" + Folders[0] + "/" + std::to_string(_h) + "_" + std::to_string(_w) + "_" + std::to_string(percent) + ".png"));
							imageWriter.save(name);
						}
						else if (folder == Folders[2] && percent < 60)
						{
							imageWriter.save(std::string(prefolder + "/" + Folders[2] + "/" + std::to_string(_h) + "_" + std::to_string(_w) + "_" + std::to_string(percent) + ".png"));
						}
					}


					

					delete [] imageWriter.imgData.pixels;
					delete [] cutedImg.pixels;
				}
			}
		}
	}

	printf("Everything ok.\n");
}

float Tool::getAverageIntense(image_data* img)
{
	float avInt = 0;
	Pixel pix;

	for (int _h = 0; _h < img->h; _h++)
	{
		for (int _w = 0; _w < img->w; _w++)
		{
			avInt += getPixelIntense(*img, _w, _h);
		}
	}

	avInt /= (img->h * img->w);

	return avInt;
}


Pixel Tool::findMaxIntenceDiff(image_data* img, int imgH, int imgW, int curH, int curW)
{
	Pixel minPixel, maxPixel, maxDiff, zeroPixel;
	minPixel.myMax();
	maxDiff.zero();
	maxPixel.zero();
	zeroPixel.zero();
	

	for (int pixelY = 0; pixelY < imgH; pixelY++)
	{
		for (int pixelX = 0; pixelX < imgW; pixelX++)
		{
			getPixel(img, pixelX, pixelY, &maxDiff);
			if (maxDiff > maxPixel)
			{
				maxPixel = maxDiff;
			}
			else if (maxDiff < minPixel && maxDiff != zeroPixel)
			{
				minPixel = maxDiff;
			}
		}
	}

	return maxPixel - minPixel;
}

void Tool::cutImage(image_data* heightMap, image_data* cutTo, int imgH, int imgW, int curH, int curW)
{
	cutTo->compPerPixel = heightMap->compPerPixel;
	cutTo->h = imgH;
	cutTo->w = imgW;
	cutTo->pixels = new stbi_uc[cutTo->h * cutTo->w * cutTo->compPerPixel];


	for (int pixelY = 0; pixelY < imgH; pixelY++)
	{
		for (int pixelX = 0; pixelX < imgW; pixelX++)
		{
			for (int byte = 0; byte < cutTo->compPerPixel; byte++)
			{
				cutTo->pixels[(pixelY * imgW + pixelX) * cutTo->compPerPixel + byte] =
					heightMap->pixels[((curH + pixelY) * heightMap->w + curW + pixelX) * cutTo->compPerPixel + byte];
				
			}
			//printf("%i, %i <- %i, %i\n", pixelY, pixelX, (curH + pixelY), curW + pixelX);
		}
	}
}

//this method uses both maps.
void Tool::prepareDatasetBothMapsBased(std::string prefolder, int h, int w)
{
	
}

void Tool::normalizePicture(image_data* pic)
{
	int MAXINTENSE = 255, MININTENSE = 0;
	int minInt = MAXINTENSE, maxInt = MININTENSE;
	float newI, curI, tI;

	for (int pixelY = 0; pixelY < pic->h; pixelY++)
	{
		for (int pixelX = 0; pixelX < pic->w; pixelX++)
		{
			tI = getPixelIntense(*pic, pixelX, pixelY);

			if (tI < minInt && tI >= MININTENSE)
			{
				minInt = tI;
			}
			
			if (tI > maxInt && tI <= MAXINTENSE)
			{
				maxInt = tI;
			}
		}
	}

	Pixel newPixel;

	for (int pixelY = 0; pixelY < pic->h; pixelY++)
	{
		for (int pixelX = 0; pixelX < pic->w; pixelX++)
		{
			newPixel.zero();
			curI = (float)getPixelIntense(*pic, pixelX, pixelY);

			if (maxInt - minInt != 0)
			{
				newI = (float)(curI - minInt) * (float)(MAXINTENSE - MININTENSE) / (float)(maxInt - minInt) + (float)MININTENSE;

				//printf("newI = %f\n", newI);
				getPixel(pic, pixelX, pixelY, &newPixel);

				if (curI == 0)
				{
					newPixel.zero();
				}
				else
				{
					newPixel.MulIntense(newI / curI);
					//printf("newI__ = %f\n", newI/curI);
				}
			}
			setPixel(*pic, pixelX, pixelY, newPixel);	

		}
	}
}

image_data Tool::RGBToOneChannel(image_data* pic)
{
	image_data newImage;
	newImage.compPerPixel = 1;
	newImage.h = pic->h;
	newImage.w = pic->w;
	newImage.pixels = new stbi_uc[newImage.h * newImage.w];

	for (int _h = 0; _h < pic->h; _h++)
	{
		for (int _w = 0; _w < pic->w; _w++)
		{
			newImage.pixels[_h * newImage.w + _w] = getPixelIntense(*pic, _h, _w);
		}
	}

	return newImage;
}