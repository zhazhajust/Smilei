#include "Interpolator3D.h"

#include <cmath>
#include <iostream>

#include "Patch.h"
#include "ElectroMagn.h"

using namespace std;

Interpolator3D::Interpolator3D( Patch *patch )
    : Interpolator()
{

    i_domain_begin = patch->getCellStartingGlobalIndex( 0 );
    j_domain_begin = patch->getCellStartingGlobalIndex( 1 );
    k_domain_begin = patch->getCellStartingGlobalIndex( 2 );
    
}

void Interpolator3D::externalMagneticField( ElectroMagn *EMfields, Particles &particles, SmileiMPI *smpi, int ibin, int ithread ){
    // Interpolate the external field at the particle position
    double *const __restrict__ ExtBLoc  = smpi->dynamics_external_Bpart[ithread].data();
    const double *const __restrict__ position_x = particles.getPtrPosition( 0 );
    const double *const __restrict__ position_y = particles.getPtrPosition( 1 );
    const double *const __restrict__ position_z = particles.getPtrPosition( 2 );
    const int nparts = particles.numberOfParticles();
    for(auto pt = EMfields->partExtFields.begin(); pt < EMfields->partExtFields.end(); pt++){
        int idx = pt->index - 3;
        #ifndef SMILEI_ACCELERATOR_GPU_OACC
            #pragma omp simd
        #endif
        for( int ipart = particles.first_index[ibin]; ipart < particles.last_index[ibin]; ipart++ ){
            ExtBLoc[idx*nparts+ipart] = pt->profile->valueAt({position_x[ipart], position_y[ipart], position_z[ipart]});
        }
    }
};