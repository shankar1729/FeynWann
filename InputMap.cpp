#include <core/Util.h>
#include <fstream>
#include <cmath>
#include "InputMap.h"

InputMap::InputMap(string filename)
{	std::ifstream ifs(filename.c_str());
	if(!ifs.is_open())
		die("Could not open system file '%s' for reading.\n", filename.c_str());
	while(!ifs.eof())
	{	string line; getline(ifs, line); //line-by-line processing (comments can now be inline)
		trim(line);
		istringstream iss(line);
		string name; string val;
		if(iss >> name >> val)
			(*this)[name] = val;
	}
	ifs.close();
}

double InputMap::get(string key, double defaultVal) const
{	auto iter = find(key);
	if(iter == end()) //not found
	{	if(std::isnan(defaultVal)) //no default provided
		{	die("\nCould not find required entry '%s' in input.\n", key.c_str());
		}
		else return defaultVal;
	}
	return atof(iter->second.c_str());
}

string InputMap::getString(string key) const
{	auto iter = find(key);
	if(iter == end()) //not found
		die("\nCould not find required entry '%s' in input.\n", key.c_str())
	return iter->second;
}
