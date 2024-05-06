#include "CollisionalNuclearReaction.h"

#include "Collisions.h"
#include "Species.h"
#include "Particles.h"
#include "Patch.h"

#include <cmath>


using namespace std;

// Constructor
CollisionalNuclearReaction::CollisionalNuclearReaction(
    Params *params,
    vector<Species *> *product_species,
    double rate_multiplier
    )
{
    if( rate_multiplier == 0. ) {
        auto_multiplier_ = true;
        rate_multiplier_ = 1.;
    } else {
        auto_multiplier_ = false;
        rate_multiplier_ = rate_multiplier;
    }
    product_particles_.resize(0);
    product_ispecies_.resize(0);
    product_mass_.resize(0);
    if( params ) {
        for( unsigned int i=0; i<product_species->size(); i++ ) {
            if( product_species->at(i) != NULL ) {
                product_particles_.push_back( new Particles() );
                product_particles_.back()->initialize( 0, *( product_species->at(i)->particles ) );
                product_ispecies_.push_back( product_species->at(i)->species_number_ );
                product_mass_.push_back( product_species->at(i)->mass_ );
            }
        }
    }
}

// Cloning Constructor
CollisionalNuclearReaction::CollisionalNuclearReaction( CollisionalNuclearReaction *CNR )
{
    product_ispecies_ = CNR->product_ispecies_;
    rate_multiplier_ = CNR->rate_multiplier_;
    auto_multiplier_ = CNR->auto_multiplier_;
    product_particles_.resize( CNR->product_particles_.size(), NULL );
    for( unsigned int i=0; i<CNR->product_particles_.size(); i++ ) {
        product_particles_[i] = new Particles();
        product_particles_[i]->initialize( 0, *CNR->product_particles_[i] );
    }
}

// Cloning Constructor
CollisionalNuclearReaction::~CollisionalNuclearReaction()
{
    for( unsigned int i=0; i<product_particles_.size(); i++ ) {
        delete product_particles_[i];
    }
    product_particles_.clear();
}


bool CollisionalNuclearReaction::occurs( double U, double coeff, double m1, double m2, double g1, double g2, double &ekin, double &log_ekin, double &W )
{
    // Interpolate the total cross-section at some value of ekin = m1(g1-1) + m2(g2-1)
    ekin = m1 * (g1-1.) + m2 * (g2-1.);
    log_ekin = log( ekin );
    double cs = crossSection( log_ekin );
    
    // Calculate probability for fusion
    double prob = coeff * cs * rate_multiplier_;
    tot_probability_ += prob;
    if( U > exp( -prob ) ) {
        W /= rate_multiplier_;
        return true;
    } else {
        return false;
    }
}


// Finish the reaction
void CollisionalNuclearReaction::finish(
    Params &params, Patch *patch, std::vector<Diagnostic *> &localDiags,
    bool intra_collisions, vector<unsigned int> sg1, vector<unsigned int> sg2,
    double npairs, int itime
) {
    // Move new particles in place
    for( unsigned int i=0; i<product_particles_.size(); i++ ) {
        patch->vecSpecies[product_ispecies_[i]]->importParticles( params, patch, *product_particles_[i], localDiags );
    }
    
    // Remove reactants that have fully reacted (very rare)
    for( unsigned int is = 0; is < sg1.size(); is++ ) {
        patch->vecSpecies[sg1[is]]->eraseWeightlessParticles();
    }
    if( ! intra_collisions ) {
        for( unsigned int is = 0; is < sg2.size(); is++ ) {
            patch->vecSpecies[sg2[is]]->eraseWeightlessParticles();
        }
    }
    
    if( auto_multiplier_ ) {
        // Update the rate multiplier
        double goal = tot_probability_ * (double) params.n_time / npairs;
        if( goal > 0. ) {
            if( goal > 2. ) {
                rate_multiplier_ *= 0.01;
            } else if( goal > 1.5 ) {
                rate_multiplier_ *= 0.1;
            } else if( goal > 1. ) {
                rate_multiplier_ *= 0.3;
            } else if( rate_multiplier_ < 1.e14 ) {
                if( goal < 0.01 ) {
                    rate_multiplier_ *= 8.;
                } else if( goal < 0.1 ) {
                    rate_multiplier_ *= 2.;
                }
            }
            if( rate_multiplier_ < 1. ) {
                rate_multiplier_ = 1.;
            }
        }
    }
}
