#ifndef IonizationFactory_H
#define IonizationFactory_H

#include "Ionization.h"
#include "IonizationTunnel.h"

#include "PicParams.h"

#include "Tools.h"

//! this class create and associate the right ionization model to species
class IonizationFactory {
public:
    static Ionization* create(PicParams& params, int ispec) {
        Ionization* Ionize = NULL;
        std::string model=params.species_param[ispec].ionization_model;

        if ( model == "tunnel" ) {
            if (params.species_param[ispec].charge > (int)params.species_param[ispec].atomic_number)
                ERROR( "Charge > atomic_number for species " << ispec );

            Ionize = new IonizationTunnel( params, ispec );

        } else if ( model != "none" ) {
            WARNING( "For species #" << ispec << ": unknown ionization model `" << model << "` ... assuming no ionization");
        }
        return Ionize;
    }

};

#endif
