#include "Species_rrll.h"

#include <iostream>

#include "Particles.h"
#include "Interpolator.h"
#include "Projector.h"
#include "Pusher.h"

using namespace std;

// ---------------------------------------------------------------------------------------------------------------------
// Creator for Species_rrLL
// ---------------------------------------------------------------------------------------------------------------------
Species_rrll::Species_rrll( Params& params,  SpeciesStructure& sparams, SmileiMPI* smpi)
    : Species( params, sparams, smpi )
{
    DEBUG("Species " << sparams.species_type << "created as rrLL");
}


// ---------------------------------------------------------------------------------------------------------------------
// Destructor for Species_rrLL
// ---------------------------------------------------------------------------------------------------------------------
Species_rrll::~Species_rrll()
{
    DEBUG("Species rrLL deleted ");
}
