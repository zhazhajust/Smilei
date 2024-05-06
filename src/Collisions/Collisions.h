#ifndef COLLISIONS_H
#define COLLISIONS_H

#include <vector>
#include <cmath>

#include "Tools.h"
#include "H5.h"
#include "CollisionalIonization.h"
#include "CollisionalNuclearReaction.h"
#include "CollisionalFusionDD.h"
#include "Random.h"

class Patch;
class Params;
class Species;
class VectorPatch;

class Collisions
{

public:
    //! Constructor for Collisions between two species
    Collisions(
        Params &params,
        unsigned int n_collisions,
        std::vector<unsigned int>,
        std::vector<unsigned int>,
        double coulomb_log,
        double coulomb_log_factor,
        bool intra_collisions,
        int every,
        int debug_every,
        CollisionalIonization *ionization,
        CollisionalNuclearReaction *nuclear_reaction,
        std::string
    );
    //! Cloning Constructor
    Collisions( Collisions * );
    //! destructor
    virtual ~Collisions();
    
    //! Method to calculate the Debye length in each bin
    static void calculate_debye_length( Params &, Patch * );
    
    //! is true if any of the collisions objects need automatically-computed coulomb log
    static bool debye_length_required;
    
    //! Method called in the main smilei loop to apply collisions at each timestep
    virtual void collide( Params &, Patch *, int, std::vector<Diagnostic *> & );
    
    //! Outputs the debug info if requested
    static void debug( Params &params, int itime, unsigned int icoll, VectorPatch &vecPatches );
    
    //! CollisionalIonization object, created if ionization required
    CollisionalIonization *Ionization;
    
    CollisionalNuclearReaction *NuclearReaction;
    
    H5Write * debug_file_;
    
protected:

    //! Identification number of the Collisions object
    int n_collisions_;
    
    //! Group of the species numbers that are associated for Collisions.
    std::vector<unsigned int> species_group1_, species_group2_;
    
    //! Coulomb logarithm (zero or negative means automatic)
    double coulomb_log_;
    
//! Coulomb logarithm (zero or negative means automatic)
    double coulomb_log_factor_;

    //! True if collisions inside a group of species, False if collisions between different groups of species
    bool intra_collisions_;
    
    //! Number of timesteps between each calculation of collisions
    int every_;
    
    //! Number of timesteps between each dump of collisions debugging
    int debug_every_;
    
    //! Hdf5 file name
    std::string filename_;
    
    //! Temporary variables for the debugging file
    double smean_, logLmean_;//, ncol;//, temperature
    
    const double twoPi = 2. * 3.14159265358979323846;
    double coeff1_, coeff2_;
    
    // Collide one particle with another
    // See equations in http://dx.doi.org/10.1063/1.4742167
    inline double one_collision(
        Particles *p1,
        unsigned int i1,
        double m1,
        Particles *p2,
        unsigned int i2,
        double m2,
        double coeff1,
        double coeff3,
        double coeff4,
        double n123,
        double n223,
        double debye2,
        double &logL,
        Random* random
    )
    {
        double m12 = m1 / m2;
        
        // If one weight is zero, then skip. Can happen after nuclear reaction
        double minW = std::min( p1->weight(i1), p2->weight(i2) );
        if( minW <= 0. ) return 0.;
        
        // Get momenta and calculate gammas
        double gamma1 = sqrt( 1. + p1->momentum( 0, i1 )*p1->momentum( 0, i1 ) + p1->momentum( 1, i1 )*p1->momentum( 1, i1 ) + p1->momentum( 2, i1 )*p1->momentum( 2, i1 ) );
        double gamma2 = sqrt( 1. + p2->momentum( 0, i2 )*p2->momentum( 0, i2 ) + p2->momentum( 1, i2 )*p2->momentum( 1, i2 ) + p2->momentum( 2, i2 )*p2->momentum( 2, i2 ) );
        double gamma12 = m12 * gamma1 + gamma2;
        double gamma12_inv = 1./gamma12;
        
        // Calculate the center-of-mass (COM) frame
        // Quantities starting with "COM" are those of the COM itself, expressed in the lab frame.
        // They are NOT quantities relative to the COM.
        double COM_vx = ( m12 * ( p1->momentum( 0, i1 ) ) + p2->momentum( 0, i2 ) ) * gamma12_inv;
        double COM_vy = ( m12 * ( p1->momentum( 1, i1 ) ) + p2->momentum( 1, i2 ) ) * gamma12_inv;
        double COM_vz = ( m12 * ( p1->momentum( 2, i1 ) ) + p2->momentum( 2, i2 ) ) * gamma12_inv;
        double COM_vsquare = COM_vx*COM_vx + COM_vy*COM_vy + COM_vz*COM_vz;
        
        // Change the momentum to the COM frame (we work only on particle 1)
        // Quantities ending with "COM" are quantities of the particle expressed in the COM frame.
        double COM_gamma, term1, term2, gamma1_COM, gamma2_COM;
        if( COM_vsquare < 1e-6 ) {
            COM_gamma = 1. + 0.5 * COM_vsquare;
            gamma1_COM = gamma1 * COM_gamma;
            gamma2_COM = gamma2 * COM_gamma;
            term1 = 0.5;
            term2 = -gamma1_COM;
        } else {
            COM_gamma = 1./sqrt( 1.-COM_vsquare );
            double vcv1g1  = COM_vx*( p1->momentum( 0, i1 ) ) + COM_vy*( p1->momentum( 1, i1 ) ) + COM_vz*( p1->momentum( 2, i1 ) );
            double vcv2g2  = COM_vx*( p2->momentum( 0, i2 ) ) + COM_vy*( p2->momentum( 1, i2 ) ) + COM_vz*( p2->momentum( 2, i2 ) );
            gamma1_COM = ( gamma1-vcv1g1 )*COM_gamma;
            gamma2_COM = ( gamma2-vcv2g2 )*COM_gamma;
            term1 = ( COM_gamma - 1. ) / COM_vsquare;
            term2 = term1*vcv1g1 - COM_gamma * gamma1;
        }
        double px_COM = p1->momentum( 0, i1 ) + term2*COM_vx;
        double py_COM = p1->momentum( 1, i1 ) + term2*COM_vy;
        double pz_COM = p1->momentum( 2, i1 ) + term2*COM_vz;
        double p2_COM = px_COM*px_COM + py_COM*py_COM + pz_COM*pz_COM;
        double p_COM  = sqrt( p2_COM );
        
        // Calculate some intermediate quantities
        double term3 = COM_gamma * gamma12_inv;
        double term4 = gamma1_COM * gamma2_COM;
        double term5 = term4/p2_COM + m12;
        double vrel = p_COM/term3/term4; // relative velocity
        
        // We first try to do a nuclear reaction
        // If succesful, then no need to do a collision
        double E, logE;
        if( NuclearReaction->occurs( random->uniform(), vrel*coeff3, m1, m2, gamma1_COM, gamma2_COM, E, logE, minW ) ) {
            // Reduce the weight of both reactants
            // If becomes zero, then the particle will be discarded later
            p1->weight(i1) -= minW;
            p2->weight(i2) -= minW;
            
            // Get the magnitude and the angle of the outgoing products in the COM frame
            std::vector<Particles*> particles;
            std::vector<double> new_p_COM, sinX, cosX;
            std::vector<short> q;
            double tot_charge = p1->charge( i1 ) + p2->charge( i2 );
            NuclearReaction->makeProducts( random, E, logE, tot_charge, particles, new_p_COM, q, sinX, cosX );
            
            // Calculate some quantities for rotating vectors
            double phi = random->uniform_2pi();
            double cosPhi = cos( phi );
            double sinPhi = sin( phi );
            double p_perp = sqrt( px_COM*px_COM + py_COM*py_COM );
            // Prepare the deflection in the COM frame
            double newpx_COM_0, newpy_COM_0, newpz_COM_0;
            if( p_perp > 1.e-10*p_COM ) { // make sure p_perp is not too small
                double inv_p_perp = 1./p_perp;
                newpx_COM_0 = ( px_COM * pz_COM * cosPhi - py_COM * p_COM * sinPhi ) * inv_p_perp;
                newpy_COM_0 = ( py_COM * pz_COM * cosPhi + px_COM * p_COM * sinPhi ) * inv_p_perp;
                newpz_COM_0 = -p_perp * cosPhi;
            } else { // if p_perp is too small, we use the limit px->0, py=0
                newpx_COM_0 = p_COM * cosPhi;
                newpy_COM_0 = p_COM * sinPhi;
                newpz_COM_0 = 0.;
            }
            // Calculate new weights
            double newW1, newW2;
            if( p1->charge(i1) != 0. || p2->charge(i2) != 0. ) {
                double weight_factor = minW / tot_charge;
                newW1 = p1->charge( i1 ) * weight_factor;
                newW2 = p2->charge( i2 ) * weight_factor;
            } else {
                newW1 = minW;
                newW2 = 0.;
            }
            
            // For each product
            for( unsigned int iproduct=0; iproduct<particles.size(); iproduct++ ){
                // Calculate the deflection in the COM frame
                double newpx_COM = newpx_COM_0 * sinX[iproduct] + px_COM *cosX[iproduct];
                double newpy_COM = newpy_COM_0 * sinX[iproduct] + py_COM *cosX[iproduct];
                double newpz_COM = newpz_COM_0 * sinX[iproduct] + pz_COM *cosX[iproduct];
                
                // Go back to the lab frame and store the results in the particle array
                double vcp = COM_vx * newpx_COM + COM_vy * newpy_COM + COM_vz * newpz_COM;
                double momentum_ratio = new_p_COM[iproduct] / p_COM;
                double term6 = momentum_ratio*term1*vcp + sqrt( new_p_COM[iproduct]*new_p_COM[iproduct] + 1. ) * COM_gamma;
                double newpx = momentum_ratio * newpx_COM + COM_vx * term6;
                double newpy = momentum_ratio * newpy_COM + COM_vy * term6;
                double newpz = momentum_ratio * newpz_COM + COM_vz * term6;
                // Make new particle at position of particle 1
                if( newW1 > 0. ) {
                    particles[iproduct]->makeParticleAt( *p1, i1, newW1, q[iproduct], newpx, newpy, newpz );
                }
                // Make new particle at position of particle 2
                if( newW2 > 0. ) {
                    particles[iproduct]->makeParticleAt( *p2, i2, newW2, q[iproduct], newpx, newpy, newpz );
                }
            }
            
            if( p1->weight(i1) == 0. || p2->weight(i2) == 0. ) {
                return 0.; // no collision
            }
            
        } // end nuclear reaction
        
        // Calculate stuff
        double qqm  = p1->charge( i1 ) * p2->charge( i2 ) / m1;
        double qqm2 = qqm * qqm;
        
        // Calculate coulomb log if necessary
        if( logL <= 0. ) { // if auto-calculation requested
            // Note : 0.00232282 is coeff2 / coeff1
            double bmin = coeff1 * std::max( 1./m1/p_COM, std::abs( 0.00232282*qqm*term3*term5 ) ); // min impact parameter
            logL = 0.5*log( 1.+debye2/( bmin*bmin ) );
            if( logL < 2. ) {
                logL = 2.;
            }
        }
        
        // Calculate the collision parameter s12 (similar to number of real collisions)
        double s = coeff3 * logL * qqm2 * term3 * p_COM * term5*term5 / ( gamma1*gamma2 );
        
        // Low-temperature correction
        double smax = coeff4 * ( m12+1. ) * vrel / std::max( m12*n123, n223 );
        if( s>smax ) {
            s = smax;
        }
        
        // Pick the deflection angles in the center-of-mass frame.
        // Instead of Nanbu http://dx.doi.org/10.1103/PhysRevE.55.4642
        // and Perez http://dx.doi.org/10.1063/1.4742167
        // we made a new fit (faster and more accurate)
        double cosX, sinX;
        double U1 = random->uniform();
        if( s < 4. ) {
            double s2 = s*s;
            double alpha = 0.37*s - 0.005*s2 - 0.0064*s2*s;
            double sin2X2 = alpha * U1 / sqrt( (1.-U1) + alpha*alpha*U1 );
            cosX = 1. - 2.*sin2X2;
            sinX = 2.*sqrt( sin2X2 *(1.-sin2X2) );
        } else {
            cosX = 2.*U1 - 1.;
            sinX = sqrt( 1. - cosX*cosX );
        }
        
        // Calculate combination of angles
        double phi = random->uniform_2pi();
        double sinXcosPhi = sinX*cos( phi );
        double sinXsinPhi = sinX*sin( phi );
        
        // Apply the deflection
        double p_perp = sqrt( px_COM*px_COM + py_COM*py_COM );
        double newpx_COM, newpy_COM, newpz_COM;
        if( p_perp > 1.e-10*p_COM ) { // make sure p_perp is not too small
            double inv_p_perp = 1./p_perp;
            newpx_COM = ( px_COM * pz_COM * sinXcosPhi - py_COM * p_COM * sinXsinPhi ) * inv_p_perp + px_COM * cosX;
            newpy_COM = ( py_COM * pz_COM * sinXcosPhi + px_COM * p_COM * sinXsinPhi ) * inv_p_perp + py_COM * cosX;
            newpz_COM = -p_perp * sinXcosPhi  +  pz_COM * cosX;
        } else { // if p_perp is too small, we use the limit px->0, py=0
            newpx_COM = p_COM * sinXcosPhi;
            newpy_COM = p_COM * sinXsinPhi;
            newpz_COM = p_COM * cosX;
        }
        
        // Go back to the lab frame and store the results in the particle array
        double vcp = COM_vx * newpx_COM + COM_vy * newpy_COM + COM_vz * newpz_COM;
        double U2 = random->uniform();
        if( U2 < p2->weight( i2 )/p1->weight( i1 ) ) { // deflect particle 1 only with some probability
            double term6 = term1*vcp + gamma1_COM * COM_gamma;
            p1->momentum( 0, i1 ) = newpx_COM + COM_vx * term6;
            p1->momentum( 1, i1 ) = newpy_COM + COM_vy * term6;
            p1->momentum( 2, i1 ) = newpz_COM + COM_vz * term6;
        }
        if( U2 < p1->weight( i1 )/p2->weight( i2 ) ) { // deflect particle 2 only with some probability
            double term6 = -m12 * term1*vcp + gamma2_COM * COM_gamma;
            p2->momentum( 0, i2 ) = -m12 * newpx_COM + COM_vx * term6;
            p2->momentum( 1, i2 ) = -m12 * newpy_COM + COM_vy * term6;
            p2->momentum( 2, i2 ) = -m12 * newpz_COM + COM_vz * term6;
        }
        
        return s;
        
    };
};


#endif
