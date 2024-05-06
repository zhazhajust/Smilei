// -----------------------------------------------------------------------------
//! \file PusherHigueraCary.cpp
//
//!  \brief Class fot the photon pusher
//
//  @date 2017-07-24
// -----------------------------------------------------------------------------

#include "PusherPhoton.h"

#include <iostream>
#include <cmath>

#include "Species.h"
#include "Particles.h"

using namespace std;

PusherPhoton::PusherPhoton( Params &params, Species *species )
    : Pusher( params, species )
{
}

PusherPhoton::~PusherPhoton()
{
}

/***********************************************************************
    Rectilinear propagation of the photons
***********************************************************************/

void PusherPhoton::operator()( Particles &particles, SmileiMPI *smpi,
                               int istart, int iend, int ithread, int ipart_ref )
{
    // Inverse normalized energy
    double * __restrict__ invgf = &( smpi->dynamics_invgf[ithread][0] );

    double* __restrict__ position_x = particles.getPtrPosition(0);
    double* __restrict__ position_y = NULL;
    double* __restrict__ position_z = NULL;
    if (nDim_>1) {
        position_y = particles.getPtrPosition(1);
        if (nDim_>2) {
            position_z = particles.getPtrPosition(2);
        }
    }
    double* __restrict__ momentum_x = particles.getPtrMomentum(0);
    double* __restrict__ momentum_y = particles.getPtrMomentum(1);
    double* __restrict__ momentum_z = particles.getPtrMomentum(2);

    #pragma omp simd
    for( int ipart=istart ; ipart<iend; ipart++ ) {

        invgf[ipart] = 1. / sqrt( momentum_x[ipart]*momentum_x[ipart] +
                                       momentum_y[ipart]*momentum_y[ipart] +
                                       momentum_z[ipart]*momentum_z[ipart] );

        // Move the photons
        position_x[ipart] += dt*momentum_x[ipart]*invgf[ipart];
        if (nDim_>1) {
            position_y[ipart] += dt*momentum_y[ipart]*invgf[ipart];
            if (nDim_>2) {
                position_z[ipart] += dt*momentum_z[ipart]*invgf[ipart];
            }
        }

    }

    //if( vecto ) {
        // int *cell_keys;
        // particles.cell_keys.resize( iend-istart );
        // cell_keys = &( particles.cell_keys[0] );

        // #pragma omp simd
        // for( int ipart=istart ; ipart<iend; ipart++ ) {
        //
        //     for( int i = 0 ; i<nDim_ ; i++ ) {
        //         cell_keys[ipart] *= nspace[i];
        //         cell_keys[ipart] += round( ( position[i][ipart]-min_loc_vec[i] ) * dx_inv_[i] );
        //     }
        //
        // }
    //}

}
