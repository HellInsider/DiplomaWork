#pragma once
#include "PixelRGB.h"
#include "png_toolkit.h"
#include <windows.h>
#include <iostream>
#include <cmath>




class Tool
{
public:
	enum MODE
	{
		HEIGHTMAP,
		BOTHMAPS
	};

	Tool(png_toolkit* h, png_toolkit* c) { heightMapHandler = h; coloredMapHandler = c; };
	~Tool() {};

	void cutImage(image_data* heightMap, image_data* cutTo, int imgH, int imgW, int curH, int curW);

	Pixel getAveragePixel(image_data* data);
	float getAverageIntense(image_data* data);
	Pixel findMaxIntenceDiff(image_data* img, int imgH, int imgW, int curH, int curW);	
	int convertToMeters(int pixSum, int compPerPix);

	void prepareDataset(Tool::MODE mode, std::string prefolder, int h, int w);
	void prepareDatasetHeighMaplBased(std::string prefolder, int h, int w);
	void prepareDatasetBothMapsBased(std::string prefolder, int h, int w);

	void imageCopy(image_data* from, image_data* to);
	void normalizePicture(image_data* pic);
	image_data RGBToOneChannel(image_data* pic);

	int checkOnHOG(image_data* pic, bool needNormalise);					//returns percent of hog gist

	int getPixelIntense(image_data Data, int x, int y);
	void setPixel(image_data Data, int x, int y, Pixel pixel);
	void getPixel(image_data* Data, int x, int y, Pixel* pixel);

	png_toolkit* heightMapHandler;
	png_toolkit* coloredMapHandler;
	image_data heightMapImg;
	image_data coloredMapImg;
	

private:
	void prepareDirs(std::string prefolder);

	std::string Folders[3] = { "general", "ocean", "mountains" };
};
