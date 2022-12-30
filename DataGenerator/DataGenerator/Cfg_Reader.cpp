#include <iostream>
#include <fstream>
#include <string>
#include "Cfg_Reader.h"

Cfg_Reader::Cfg_Reader(char* Cfg_path)
{
	stream = new std::ifstream(Cfg_path);
}

Cfg_Reader::~Cfg_Reader()
{
	stream->close();
}

bool Cfg_Reader::Read(std::string &heighMap, std::string& coloredMap)
{   	
	if (!*stream || stream->eof())
	{
		return false;
	}

	*stream >> heighMap;
	*stream >> coloredMap;

	return true;
}