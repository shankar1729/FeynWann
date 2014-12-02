#ifndef WANNIERMETROPOLIS_INPUTMAP_H
#define WANNIERMETROPOLIS_INPUTMAP_H

#include <core/string.h>
#include <map>

//Parse simple input file into a dictionary
class InputMap : std::map<string,double>
{
public:
	InputMap(string filename);
	double get(string key, double defaultVal=NAN) const;
};

#endif //WANNIERMETROPOLIS_INPUTMAP_H