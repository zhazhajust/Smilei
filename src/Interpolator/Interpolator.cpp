#include "Interpolator.h"

#include <cmath>
#include <iostream>

#include "Params.h"
#include "Patch.h"
#include "ElectroMagn.h"

using namespace std;

void Interpolator::externalMagneticField( ElectroMagn *EMfields, Particles &particles, SmileiMPI *smpi, int ibin, int ithread, int nDim_particle ){
    // Interpolate the external field at the particle position
    double *const __restrict__ ExtBLoc  = smpi->dynamics_external_Bpart[ithread].data();
    // std::vector<double> *extBpart = &( smpi->dynamics_external_Bpart[ithread] );

    // You can't initialize it, which will leed to data error. And i don't know why.
    // std::fill(ExtBLoc, ExtBLoc + smpi->dynamics_external_Bpart[ithread].size(), 0.0);
    // #ifndef SMILEI_ACCELERATOR_GPU_OACC
    //     #pragma omp simd
    // #else
    //     #pragma acc parallel loop present(ExtBLoc) vector_length(256)
    // #endif
    // for (int i = 0; i < smpi->dynamics_external_Bpart[ithread].size(); ++i) {
    //     ExtBLoc[i] = 0.0;
    // }

    const double *const __restrict__ position_x = particles.getPtrPosition( 0 );
    const double *const __restrict__ position_y = nDim_particle > 1 ? particles.getPtrPosition( 1 ) : nullptr;
    const double *const __restrict__ position_z = nDim_particle > 2 ? particles.getPtrPosition( 2 ) : nullptr;
    const int nparts = particles.numberOfParticles();
    for(auto pt = EMfields->partExtFields.begin(); pt < EMfields->partExtFields.end(); pt++){
        // string field_name = LowerCase(pt->field);
        // int idx = ext_idx_enum[field_name];
        int idx = pt->index - 3;
        // double *const __restrict__ extBLoc = &( ( *extBpart )[idx * nparts] );
        #ifndef SMILEI_ACCELERATOR_GPU_OACC
            #pragma omp simd
        // #else
        //     #pragma acc parallel loop present(position_x, position_y, position_z, ExtBLoc) vector_length(256)
        #endif
        for( int ipart = particles.first_index[ibin]; ipart < particles.last_index[ibin]; ipart++ ){
            std::vector<double> pos(nDim_particle);
            pos[0] = position_x[ipart];
            if( nDim_particle > 1 ) {
                pos[1] = position_y[ipart];
                if( nDim_particle > 2 ) {
                    pos[2] = position_z[ipart];
                }
            }
            ExtBLoc[idx*nparts+ipart] = pt->profile->valueAt(pos);
            // extBLoc[ipart] = pt->profile->valueAt({position_x[ipart], position_y[ipart], position_z[ipart]});
        }
    }
};