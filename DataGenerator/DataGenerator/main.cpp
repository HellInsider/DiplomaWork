#include <iostream>
#include <string.h>
#include "Cfg_Reader.h"
#include "Tool.h"
#include "png_toolkit.h"
#include <windows.h>
#pragma warning(disable: 4996)

char* getTimeString(char* buffer)
{
	time_t t = time(0);   // get time now
	struct tm* now = localtime(&t);

	
	strftime(buffer, 80, "%Y_%m_%d_%H-%M", now);
	return buffer;
}

int main( int argc, char *argv[] )
{
	char buffer[80];
	if (argc != 2)
	{
		printf("Err: Not enought args\n");
		return 0;
	}

	Cfg_Reader Reader(argv[1]);
	std::string heighMap = "", coloredMap = "";
	png_toolkit heightMapHandler, coloredMapHandler;
	Tool myTool(&heightMapHandler, &coloredMapHandler);
	std::string path;
	Tool::MODE mode = Tool::HEIGHTMAP;


	Reader.Read(heighMap, coloredMap);

	if (!heightMapHandler.load(heighMap))
	{
		printf("Err: Not found heighMap\n");
		return 0;
	}

	if (!coloredMapHandler.load(coloredMap))
	{
		printf("Err: Not found coloredMap\n");
		return 0;
	}


	path = std::string("./out/") + std::string(getTimeString(buffer));

	if (!CreateDirectory(path.c_str(), NULL))
	{
		std::cout << "ERR!\n Maybe dir already exist. In this case Everything OK.\n";
	}

	std::cout << "Parcing " << path << "...\n";
	myTool.prepareDataset(mode, path, 512, 512); //				Change size of pictures here
	

	printf("Complete!\n\n   Press any key...");
	getchar();

    return 0;
}

