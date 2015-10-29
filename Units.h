#ifndef WANNIERMETROPOLIS_UNITS_H
#define WANNIERMETROPOLIS_UNITS_H

#include <core/Units.h>

const double invSeconds = 2.418884326505e-17;
const double Coulomb = Joule/eV;
const double Volt = Joule/Coulomb;
const double Ampere = Coulomb*invSeconds;
const double Ohm = Volt/Ampere;

#endif //WANNIERMETROPOLIS_UNITS_H