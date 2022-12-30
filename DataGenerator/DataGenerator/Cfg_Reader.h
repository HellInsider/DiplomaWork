#pragma once
#include <string>
#include "Filter.h"

class Cfg_Reader
{  
	std::ifstream *stream;

public:
	bool Read(std::string& heighMap, std::string& coloredMap);
	Cfg_Reader(char* Cfg_path);
	~Cfg_Reader();
};