#include "SpeciesV.h"

#include <cmath>
#include <ctime>
#include <cstdlib>

#include <iostream>

#include <omp.h>

// IDRIS
#include <cstring>
// IDRIS
#include "PusherFactory.h"
#include "IonizationFactory.h"
#include "PartBoundCond.h"
#include "PartWall.h"
#include "BoundaryConditionType.h"

#include "ElectroMagn.h"
#include "Interpolator.h"
#include "InterpolatorFactory.h"
#include "Profile.h"

#include "Projector.h"
#include "ProjectorFactory.h"

#include "SimWindow.h"
#include "Patch.h"

// #include "Field.h"
#include "Field1D.h"
#include "Field2D.h"
#include "Field3D.h"
#include "Tools.h"

#include "DiagnosticTrack.h"

using namespace std;


// ---------------------------------------------------------------------------------------------------------------------
// Constructor for Species
// input: simulation parameters & Species index
// ---------------------------------------------------------------------------------------------------------------------
SpeciesV::SpeciesV( Params &params, Patch *patch ) :
    Species( params, patch )
{
    initCluster( params );
    npack_ = 0 ;
    packsize_ = 0;

    for (unsigned int idim=0; idim < params.nDim_field; idim++){
        distance[idim] = &Species::cartesian_distance;
    }
    if (params.geometry == "AMcylindrical"){
        distance[1] = &Species::radial_distance;
    }


}//END SpeciesV creator

// ---------------------------------------------------------------------------------------------------------------------
// Destructor for Species
// ---------------------------------------------------------------------------------------------------------------------
SpeciesV::~SpeciesV()
{
}


void SpeciesV::initCluster( Params &params )
{
    int ncells = 1;
    for( unsigned int iDim=0 ; iDim<nDim_field ; iDim++ ) {
        ncells *= ( params.n_space[iDim]+1 );
    }
    particles->last_index.resize( ncells, 0 );
    particles->first_index.resize( ncells, 0 );
    count.resize( ncells, 0 );

    //Size in each dimension of the buffers on which each bin are projected
    //In 1D the particles of a given bin can be projected on 6 different nodes at the second order (oversize = 2)

    //Primal dimension of fields.
    f_dim0 =  params.n_space[0] + 2 * oversize[0] +1;
    f_dim1 =  params.n_space[1] + 2 * oversize[1] +1;
    f_dim2 =  params.n_space[2] + 2 * oversize[2] +1;

    b_dim.resize( params.nDim_field, 1 );
    if( nDim_field == 1 ) {
        b_dim[0] = ( 1 + cluster_width_ ) + 2 * oversize[0];
        f_dim1 = 1;
        f_dim2 = 1;
    }
    if( nDim_field == 2 ) {
        b_dim[0] = ( 1 + cluster_width_ ) + 2 * oversize[0]; // There is a primal number of bins.
        b_dim[1] =  f_dim1;
        f_dim2 = 1;
    }
    if( nDim_field == 3 ) {
        b_dim[0] = ( 1 + cluster_width_ ) + 2 * oversize[0]; // There is a primal number of bins.
        b_dim[1] = f_dim1;
        b_dim[2] = f_dim2;
    }

    //Initialize specMPI
    MPI_buffer_.allocate( nDim_field );

    //ener_tot = 0.;
    nrj_bc_lost = 0.;
    nrj_mw_out = 0.;
    nrj_mw_inj = 0.;
    nrj_new_part_ = 0.;
    nrj_radiated_ = 0.;

}//END initCluster


void SpeciesV::dynamics( double time_dual, unsigned int ispec,
                         ElectroMagn *EMfields, Params &params, bool diag_flag,
                         PartWalls *partWalls,
                         Patch *patch, SmileiMPI *smpi,
                         RadiationTables &RadiationTables,
                         MultiphotonBreitWheelerTables &MultiphotonBreitWheelerTables,
                         vector<Diagnostic *> &localDiags )
{
    int ithread;
#ifdef _OPENMP
    ithread = omp_get_thread_num();
#else
    ithread = 0;
#endif

#ifdef  __DETAILED_TIMERS
    double timer;
#endif

    if( npack_==0 ) {
        npack_    = 1;
        packsize_ = ( f_dim1-2*oversize[1] );

        //if ( ( (long int)particles->last_index.back() < (long int)60000 ) || (Radiate) || (Ionize) || (Multiphoton_Breit_Wheeler_process) )
        packsize_ *= ( f_dim0-2*oversize[0] );
        //else
        //    npack_ *= (f_dim0-2*oversize[0]);

        if( nDim_field == 3 ) {
            packsize_ *= ( f_dim2-2*oversize[2] );
        }
    }

    unsigned int iPart;

    int tid( 0 );
    std::vector<double> nrj_lost_per_thd( 1, 0. );

    // -------------------------------
    // calculate the particle dynamics
    // -------------------------------
    if( time_dual>time_frozen_ || Ionize ) { // moving particle

        //Point to local thread dedicated buffers
        //Still needed for ionization
        vector<double> *Epart = &( smpi->dynamics_Epart[ithread] );


        for( unsigned int ipack = 0 ; ipack < npack_ ; ipack++ ) {

            int nparts_in_pack = particles->last_index[( ipack+1 ) * packsize_-1 ];
            smpi->dynamics_resize( ithread, nDim_field, nparts_in_pack, params.geometry=="AMcylindrical" );

#ifdef  __DETAILED_TIMERS
            timer = MPI_Wtime();
#endif

            // Interpolate the fields at the particle position
            for( unsigned int scell = 0 ; scell < packsize_ ; scell++ )
                Interp->fieldsWrapper( EMfields, *particles, smpi, &( particles->first_index[ipack*packsize_+scell] ),
                                       &( particles->last_index[ipack*packsize_+scell] ),
                                       ithread, particles->first_index[ipack*packsize_] );

#ifdef  __DETAILED_TIMERS
            patch->patch_timers[0] += MPI_Wtime() - timer;
#endif

            // Ionization
            if( Ionize ) {
#ifdef  __DETAILED_TIMERS
                timer = MPI_Wtime();
#endif
                for( unsigned int scell = 0 ; scell < particles->first_index.size() ; scell++ ) {
                    ( *Ionize )( particles, particles->first_index[scell], particles->last_index[scell], Epart, patch, Proj );
                }
#ifdef  __DETAILED_TIMERS
                patch->patch_timers[4] += MPI_Wtime() - timer;
#endif
            }

            if ( time_dual <= time_frozen_ ) continue;

            //Prepare for sorting
            for( unsigned int i=0; i<count.size(); i++ ) {
                count[i] = 0;
            }

            // Radiation losses
            if( Radiate ) {
#ifdef  __DETAILED_TIMERS
                timer = MPI_Wtime();
#endif

                for( unsigned int scell = 0 ; scell < particles->first_index.size() ; scell++ ) {

                    ( *Radiate )( *particles, this->photon_species_, smpi,
                                  RadiationTables, nrj_radiated_,
                                  particles->first_index[scell], particles->last_index[scell], ithread );

                    // // Update scalar variable for diagnostics
                    // nrj_radiated_ += Radiate->getRadiatedEnergy();
                    //
                    // // Update the quantum parameter chi
                    // Radiate->computeParticlesChi( *particles,
                    //                               smpi,
                    //                               first_index[scell],
                    //                               last_index[scell],
                    //                               ithread );
                }
#ifdef  __DETAILED_TIMERS
                patch->patch_timers[5] += MPI_Wtime() - timer;
#endif
            }

            // Multiphoton Breit-Wheeler
            if( Multiphoton_Breit_Wheeler_process ) {
#ifdef  __DETAILED_TIMERS
                timer = MPI_Wtime();
#endif
                for( unsigned int scell = 0 ; scell < particles->first_index.size() ; scell++ ) {

                    // Pair generation process
                    // We reuse nrj_radiated_ for the pairs
                    ( *Multiphoton_Breit_Wheeler_process )( *particles,
                                                            smpi,
                                                            MultiphotonBreitWheelerTables,
                                                            nrj_radiated_,
                                                            particles->first_index[scell], particles->last_index[scell], ithread );

                    // Update the photon quantum parameter chi of all photons
                    Multiphoton_Breit_Wheeler_process->compute_thread_chiph( *particles,
                            smpi,
                            particles->first_index[scell],
                            particles->last_index[scell],
                            ithread );

                    // Suppression of the decayed photons into pairs
                    Multiphoton_Breit_Wheeler_process->decayed_photon_cleaning(
                        *particles, smpi, scell, particles->first_index.size(), &particles->first_index[0], &particles->last_index[0], ithread );

                }
#ifdef  __DETAILED_TIMERS
                patch->patch_timers[6] += MPI_Wtime() - timer;
#endif
            }

#ifdef  __DETAILED_TIMERS
            timer = MPI_Wtime();
#endif

            // Push the particles and the photons
            ( *Push )( *particles, smpi, particles->first_index[ipack*packsize_],
                       particles->last_index[ipack*packsize_+packsize_-1],
                       ithread, particles->first_index[ipack*packsize_] );

#ifdef  __DETAILED_TIMERS
            patch->patch_timers[1] += MPI_Wtime() - timer;
            timer = MPI_Wtime();
#endif

            // Boundary conditions and energy lost

            double energy_lost( 0. );

            if( mass_>0 ) {

                for( unsigned int scell = 0 ; scell < packsize_ ; scell++ ) {

                    energy_lost = 0;

                    // Apply wall and boundary conditions
                    for( unsigned int iwall=0; iwall<partWalls->size(); iwall++ ) {
                        ( *partWalls )[iwall]->apply( this, particles->first_index[ipack*packsize_+scell], particles->last_index[ipack*packsize_+scell], smpi->dynamics_invgf[ithread], patch->rand_, energy_lost );
                        nrj_lost_per_thd[tid] += mass_ * energy_lost;
                    }

                    // Boundary Condition may be physical or due to domain decomposition
                    partBoundCond->apply( this, particles->first_index[ipack*packsize_+scell], particles->last_index[ipack*packsize_+scell], smpi->dynamics_invgf[ithread], patch->rand_, energy_lost );
                    nrj_lost_per_thd[tid] += mass_ * energy_lost;

                    // for( iPart=particles->first_index[ipack*packsize_+scell] ; iPart<particles->last_index[ipack*packsize_+scell]; iPart++ ) {
                    //     if ( particles->cell_keys[iPart] != -1 ) {
                    //         //Compute cell_keys of remaining particles
                    //         for( unsigned int i = 0 ; i<nDim_field; i++ ) {
                    //             particles->cell_keys[iPart] *= length_[i];
                    //             particles->cell_keys[iPart] += round( ((this)->*(distance[i]))(particles, i, iPart) * dx_inv_[i] );
                    //         }
                    //         count[particles->cell_keys[iPart]] ++;
                    //     }
                    // }

                }

            } else if( mass_==0 ) {

                for( unsigned int scell = 0 ; scell < packsize_ ; scell++ ) {

                    energy_lost = 0;

                    for( unsigned int iwall=0; iwall<partWalls->size(); iwall++ ) {
                        ( *partWalls )[iwall]->apply( this, particles->first_index[ipack*packsize_+scell], particles->last_index[ipack*packsize_+scell], smpi->dynamics_invgf[ithread], patch->rand_, energy_lost );
                        nrj_lost_per_thd[tid] += energy_lost;
                    }

                    // Boundary Condition may be physical or due to domain decomposition
                    partBoundCond->apply( this, particles->first_index[ipack*packsize_+scell], particles->last_index[ipack*packsize_+scell], smpi->dynamics_invgf[ithread], patch->rand_, energy_lost );
                    nrj_lost_per_thd[tid] += energy_lost;

                    // for( iPart=particles->first_index[ipack*packsize_+scell] ; iPart<particles->last_index[ipack*packsize_+scell]; iPart++ ) {
                    //     if ( particles->cell_keys[iPart] != -1 ) {
                    //         //Compute cell_keys of remaining particles
                    //         for( unsigned int i = 0 ; i<nDim_field; i++ ) {
                    //             particles->cell_keys[iPart] *= length_[i];
                    //             particles->cell_keys[iPart] += round( ((this)->*(distance[i]))(particles, i, iPart) * dx_inv_[i] );
                    //         }
                    //         count[particles->cell_keys[iPart]] ++;
                    //     }
                    // }

                }
            }

            // Cell keys
            computeParticleCellKeys( params,
                                     particles,
                                     &particles->cell_keys[0],
                                     &count[0],
                                     particles->first_index[ipack*packsize_],
                                     particles->last_index[ipack*packsize_+packsize_-1] );

            // if (params.geometry == "AMcylindrical"){
            //
            //     for( iPart=particles->first_index[ipack*packsize_] ; iPart<particles->last_index[ipack*packsize_+packsize_-1]; iPart++ ) {
            //         if ( particles->cell_keys[iPart] != -1 ) {
            //             //Compute cell_keys of remaining particles
            //             for( unsigned int i = 0 ; i<nDim_field; i++ ) {
            //                 particles->cell_keys[iPart] *= length_[i];
            //                 particles->cell_keys[iPart] += round( ((this)->*(distance[i]))(particles, i, iPart) * dx_inv_[i] );
            //             }
            //             count[particles->cell_keys[iPart]] ++;
            //         }
            //     }
            //
            // } else if (nDim_field == 3) {
            //
            //     #pragma omp simd
            //     for( iPart=particles->first_index[ipack*packsize_] ; iPart<particles->last_index[ipack*packsize_+packsize_-1]; iPart++ ) {
            //         if ( particles->cell_keys[iPart] != -1 ) {
            //             //Compute cell_keys of remaining particles
            //             particles->cell_keys[iPart] *= length_[0];
            //             particles->cell_keys[iPart] += round( (particles->position(0, iPart) - min_loc_vec[0]) * dx_inv_[0] );
            //             particles->cell_keys[iPart] *= length_[1];
            //             particles->cell_keys[iPart] += round( (particles->position(1, iPart) - min_loc_vec[1]) * dx_inv_[1] );
            //             particles->cell_keys[iPart] *= length_[2];
            //             particles->cell_keys[iPart] += round( (particles->position(2, iPart) - min_loc_vec[2]) * dx_inv_[2] );
            //         }
            //     }
            //
            // } else if (nDim_field == 2) {
            //
            //     #pragma omp simd
            //     for( iPart=particles->first_index[ipack*packsize_] ; iPart<particles->last_index[ipack*packsize_+packsize_-1]; iPart++ ) {
            //         if ( particles->cell_keys[iPart] != -1 ) {
            //             //Compute cell_keys of remaining particles
            //             particles->cell_keys[iPart] *= length_[0];
            //             particles->cell_keys[iPart] += round( (particles->position(0, iPart) - min_loc_vec[0]) * dx_inv_[0] );
            //             particles->cell_keys[iPart] *= length_[1];
            //             particles->cell_keys[iPart] += round( (particles->position(1, iPart) - min_loc_vec[1]) * dx_inv_[1] );
            //
            //         }
            //     }
            // } else if (nDim_field == 1) {
            //
            //     #pragma omp simd
            //     for( iPart=particles->first_index[ipack*packsize_] ; iPart<particles->last_index[ipack*packsize_+packsize_-1]; iPart++ ) {
            //         if ( particles->cell_keys[iPart] != -1 ) {
            //             //Compute cell_keys of remaining particles
            //             particles->cell_keys[iPart] *= length_[0];
            //             particles->cell_keys[iPart] += round( (particles->position(0, iPart) - min_loc_vec[0]) * dx_inv_[0] );
            //         }
            //     }
            //
            // }
            //
            // for( iPart=particles->first_index[ipack*packsize_] ; iPart<particles->last_index[ipack*packsize_+packsize_-1]; iPart++ ) {
            //     if ( particles->cell_keys[iPart] != -1 ) {
            //         count[particles->cell_keys[iPart]] ++;
            //     }
            // }



#ifdef  __DETAILED_TIMERS
            patch->patch_timers[3] += MPI_Wtime() - timer;
#endif

            // Project currents if not a Test species and charges as well if a diag is needed.
            // Do not project if a photon
            if( ( !particles->is_test ) && ( mass_ > 0 ) )
#ifdef  __DETAILED_TIMERS
                timer = MPI_Wtime();
#endif

            for( unsigned int scell = 0 ; scell < packsize_ ; scell++ )
                Proj->currentsAndDensityWrapper(
                    EMfields, *particles, smpi, particles->first_index[ipack*packsize_+scell],
                    particles->last_index[ipack*packsize_+scell],
                    ithread,
                    diag_flag, params.is_spectral,
                    ispec, ipack*packsize_+scell, particles->first_index[ipack*packsize_]
                );

#ifdef  __DETAILED_TIMERS
            patch->patch_timers[2] += MPI_Wtime() - timer;
#endif

            for( unsigned int ithd=0 ; ithd<nrj_lost_per_thd.size() ; ithd++ ) {
                nrj_bc_lost += nrj_lost_per_thd[tid];
            }
        } // End loop on packs
    } //End if moving or ionized particles

    if(time_dual <= time_frozen_ && diag_flag &&( !particles->is_test ) ) { //immobile particle (at the moment only project density)

        if( params.geometry != "AMcylindrical" ) {
            double *b_rho = EMfields->rho_s[ispec] ? &( *EMfields->rho_s[ispec] )( 0 ) : &( *EMfields->rho_ )( 0 ) ;
            for( unsigned int scell = 0 ; scell < particles->first_index.size() ; scell ++ ) { //Loop for projection on buffer_proj
                for( iPart=particles->first_index[scell] ; ( int )iPart<particles->last_index[scell]; iPart++ ) {
                    Proj->basic( b_rho, ( *particles ), iPart, 0 );
                } //End loop on particles
            }//End loop on scells

        } else { // AM case
            ElectroMagnAM *emAM = static_cast<ElectroMagnAM *>( EMfields );
            int n_species = patch->vecSpecies.size();
            for( unsigned int imode = 0; imode<params.nmodes; imode++ ) {
                int ifield = imode*n_species+ispec;
                complex<double> *b_rho = emAM->rho_AM_s[ifield] ? &( *emAM->rho_AM_s[ifield] )( 0 ) : &( *emAM->rho_AM_[imode] )( 0 ) ;
                for( unsigned int scell = 0 ; scell < particles->first_index.size() ; scell ++ ) { //Loop for projection on buffer_proj
                    for( int iPart=particles->first_index[scell] ; iPart<particles->last_index[scell]; iPart++ ) {
                        Proj->basicForComplex( b_rho, ( *particles ), iPart, 0, imode );
                    }
                }
            }
        }


    } // End projection for frozen particles

}//END dynamics


// ---------------------------------------------------------------------------------------------------------------------
// For all particles of the species
//   - increment the charge (projection)
//   - used at initialisation for Poisson (and diags if required, not for now dynamics )
// ---------------------------------------------------------------------------------------------------------------------
void SpeciesV::computeCharge( unsigned int ispec, ElectroMagn *EMfields, bool old /*=false*/ )
{
    // -------------------------------
    // calculate the particle charge
    // -------------------------------
    if( ( !particles->is_test ) ) {
        if( !dynamic_cast<ElectroMagnAM *>( EMfields ) ) {
            double *b_rho=&( *EMfields->rho_ )( 0 );
            for( unsigned int iPart=particles->first_index[0] ; ( int )iPart<particles->last_index[particles->last_index.size()-1]; iPart++ ) {
                Proj->basic( b_rho, ( *particles ), iPart, 0 );
            }
        } else {
            ElectroMagnAM *emAM = static_cast<ElectroMagnAM *>( EMfields );
            unsigned int Nmode = emAM->rho_AM_.size();
            for( unsigned int imode=0; imode<Nmode; imode++ ) {
                complex<double> *b_rho = old ? &( *emAM->rho_old_AM_[imode] )( 0 ) : &( *emAM->rho_AM_[imode] )( 0 );
                for( unsigned int iPart=particles->first_index[0] ; ( int )iPart<particles->last_index[particles->last_index.size()-1]; iPart++ ) {
                    Proj->basicForComplex( b_rho, ( *particles ), iPart, 0, imode );
                }
             }
       }
   }

}//END computeCharge


// ---------------------------------------------------------------------------------------------------------------------
// Sort particles
// ---------------------------------------------------------------------------------------------------------------------
void SpeciesV::sortParticles( Params &params, Patch *patch )
{
    unsigned int npart, ncell;
    int ip_dest, cell_target;
    vector<int> buf_cell_keys[3][2];
    std::vector<unsigned int> cycle;
    unsigned int ip_src;

    //Number of dual cells
    ncell = ( params.n_space[0]+1 );
    for( unsigned int i=1; i < nDim_field; i++ ) {
        ncell *= length_[i];
    }

    //Number of particles before exchange
    npart = particles->size();

    //Loop over just arrived particles to compute their cell keys and contribution to count
    for( unsigned int idim=0; idim < nDim_field ; idim++ ) {
        for( unsigned int ineighbor=0 ; ineighbor < 2 ; ineighbor++ ) {
            buf_cell_keys[idim][ineighbor].resize( MPI_buffer_.part_index_recv_sz[idim][ineighbor] );

            // #pragma omp simd
            // for( unsigned int ip=0; ip < MPI_buffer_.part_index_recv_sz[idim][ineighbor]; ip++ ) {
            //     for( unsigned int ipos=0; ipos < nDim_field ; ipos++ ) {
            //         double X = ((this)->*(distance[ipos]))(&MPI_buffer_.partRecv[idim][ineighbor], ipos, ip);
            //         int IX = round( X * dx_inv_[ipos] );
            //         buf_cell_keys[idim][ineighbor][ip] = buf_cell_keys[idim][ineighbor][ip] * length_[ipos] + IX;
            //     }
            // }
            // // not vectorizable because random access to count
            // for( unsigned int ip=0; ip < MPI_buffer_.part_index_recv_sz[idim][ineighbor]; ip++ ) {
            //     count[buf_cell_keys[idim][ineighbor][ip]] ++;
            // }

            computeParticleCellKeys( params,
                                     &MPI_buffer_.partRecv[idim][ineighbor],
                                     &buf_cell_keys[idim][ineighbor][0],
                                     &count[0],
                                     0,
                                     MPI_buffer_.part_index_recv_sz[idim][ineighbor] );

        }
    }

    // second loop convert the count array in cumulative sum
    particles->first_index[0]=0;
    for( unsigned int ic=1; ic < ncell; ic++ ) {
        particles->first_index[ic] = particles->first_index[ic-1] + count[ic-1];
        particles->last_index[ic-1]= particles->first_index[ic];
    }


    //New total number of particles is stored as last element of particles->last_index
    particles->last_index[ncell-1] = particles->last_index[ncell-2] + count.back() ;

    //Now proceed to the cycle sort

    if( MPI_buffer_.partRecv[0][0].size() == 0 ) {
        MPI_buffer_.partRecv[0][0].initialize( 0, *particles );    //Is this correct ?
    }

    // Resize the particle vector
    if( ( unsigned int )particles->last_index.back() > npart ) {
        particles->resize( particles->last_index.back(), nDim_particle, params.keep_position_old );
        for (int ip = npart ; ip < particles->last_index.back() ; ip ++) {
            particles->cell_keys[ip] = -1;
        }
    }

    //Copy all particles from MPI buffers back to the writable particles via cycle sort pass.
    for( unsigned int idim=0; idim < nDim_field ; idim++ ) {
        for( unsigned int ineighbor=0 ; ineighbor < 2 ; ineighbor++ ) {
            for( unsigned int ip=0; ip < MPI_buffer_.part_index_recv_sz[idim][ineighbor]; ip++ ) {
                cycle.resize( 1 );
                cell_target = buf_cell_keys[idim][ineighbor][ip];
                ip_dest = particles->first_index[cell_target];
                while( particles->cell_keys[ip_dest] == cell_target ) {
                    ip_dest++;
                }
                particles->first_index[cell_target] = ip_dest + 1 ;
                cycle[0] = ip_dest;
                cell_target = particles->cell_keys[ip_dest];
                //As long as the particle is not erased, we can build up the cycle.
                while( cell_target != -1 ) {
                    ip_dest = particles->first_index[cell_target];
                    while( particles->cell_keys[ip_dest] == cell_target ) {
                        ip_dest++;
                    }
                    particles->first_index[cell_target] = ip_dest + 1 ;
                    cycle.push_back( ip_dest );
                    cell_target = particles->cell_keys[ip_dest];
                }
                //Last target_cell is -1, the particle must be erased:
                particles->translateParticles( cycle );
                //Eventually copy particle from the MPI buffer into the particle vector.
                MPI_buffer_.partRecv[idim][ineighbor].overwriteParticle( ip, *particles, cycle[0] );
            }
        }
    }

    //Copy valid particles siting over particles->last_index.back() back into the real particles array (happens when more particles are lost than received)
    for( unsigned int ip=( unsigned int )particles->last_index.back(); ip < npart; ip++ ) {
        cell_target = particles->cell_keys[ip];

        if( cell_target == -1 ) {
            continue;
        }
        cycle.resize( 0 );
        cycle.push_back( ip );

        //As long as the particle is not erased, we can build up the cycle.
        while( cell_target != -1 ) {

            ip_dest = particles->first_index[cell_target];

            while( particles->cell_keys[ip_dest] == cell_target ) {
                ip_dest++;
            }
            particles->first_index[cell_target] = ip_dest + 1 ;
            cycle.push_back( ip_dest );
            cell_target = particles->cell_keys[ip_dest];
        }
        //Last target_cell is -1, the particle must be erased:
        particles->translateParticles( cycle );
    }

    // Resize the particle vector
    if( ( unsigned int )particles->last_index.back() < npart ) {
        particles->resize( particles->last_index.back(), nDim_particle, params.keep_position_old );
        //particles->cell_keys.resize( particles->last_index.back() ); // Merge this in particles.resize(..) ?
    }

    //Loop over all cells
    for( int icell = 0 ; icell < ( int )ncell; icell++ ) {
        for( unsigned int ip=( unsigned int )particles->first_index[icell]; ip < ( unsigned int )particles->last_index[icell] ; ip++ ) {
            //update value of current cell 'icell' if necessary
            //if particle changes cell, build a cycle of exchange as long as possible. Treats all particles
            if( particles->cell_keys[ip] != icell ) {
                cycle.resize( 1 );
                cycle[0] = ip;
                ip_src = ip;
                //While the destination particle is not going out of the patch or back to the initial cell, keep building the cycle.
                while( particles->cell_keys[ip_src] != icell ) {
                    //Scan the next cell destination
                    ip_dest = particles->first_index[particles->cell_keys[ip_src]];
                    while( particles->cell_keys[ip_dest] == particles->cell_keys[ip_src] ) {
                        ip_dest++;
                    }
                    //In the destination cell, if a particle is going out of this cell, add it to the cycle.
                    particles->first_index[particles->cell_keys[ip_src]] = ip_dest + 1 ;
                    cycle.push_back( ip_dest );
                    ip_src = ip_dest; //Destination becomes source for the next iteration
                }
                //swap parts
                particles->swapParticles( cycle );
            }
        }
    } //end loop on cells
    // Restore particles->first_index initial value
    particles->first_index[0]=0;
    for( unsigned int ic=1; ic < ncell; ic++ ) {
        particles->first_index[ic] = particles->last_index[ic-1];
    }
}

// Compute particle cell_keys from istart to iend
// This function vectorizes well on Intel and ARM architectures
void SpeciesV::computeParticleCellKeys( Params    & params,
                                        Particles * particles,
                                        int       * __restrict__ cell_keys,
                                        int       * __restrict__ count,
                                        unsigned int istart,
                                        unsigned int iend ) {

    unsigned int iPart;

    // int    * __restrict__ cell_keys  = particles->getPtrCellKeys();
    double * __restrict__ position_x = particles->getPtrPosition(0);
    double * __restrict__ position_y = particles->getPtrPosition(1);
    double * __restrict__ position_z = particles->getPtrPosition(2);

    if (params.geometry == "AMcylindrical"){

        for( iPart=istart; iPart < iend ; iPart++ ) {
            if ( cell_keys[iPart] != -1 ) {
                //Compute cell_keys of remaining particles
                for( unsigned int i = 0 ; i<nDim_field; i++ ) {
                    cell_keys[iPart] *= length_[i];
                    cell_keys[iPart] += round( ((this)->*(distance[i]))(particles, i, iPart) * dx_inv_[i] );
                }
                count[cell_keys[iPart]] ++;
            }
        }

    } else if (nDim_field == 3) {

        double min_loc_x = min_loc_vec[0];
        double min_loc_y = min_loc_vec[1];
        double min_loc_z = min_loc_vec[2];

        #pragma omp simd
        for( iPart=istart; iPart < iend ; iPart++  ) {
            if ( cell_keys[iPart] != -1 ) {
                //Compute cell_keys of remaining particles
                cell_keys[iPart]  = round( (position_x[iPart] - min_loc_x) * dx_inv_[0] );
                cell_keys[iPart] *= length_[1];
                cell_keys[iPart] += round( (position_y[iPart] - min_loc_y) * dx_inv_[1] );
                cell_keys[iPart] *= length_[2];
                cell_keys[iPart] += round( (position_z[iPart] - min_loc_z) * dx_inv_[2] );
            }
        }

    } else if (nDim_field == 2) {

        double min_loc_x = min_loc_vec[0];
        double min_loc_y = min_loc_vec[1];

        #pragma omp simd
        for( iPart=istart; iPart < iend ; iPart++  ) {
            if ( cell_keys[iPart] != -1 ) {
                //Compute cell_keys of remaining particles
                cell_keys[iPart]  = round( (position_x[iPart] - min_loc_x) * dx_inv_[0] );
                cell_keys[iPart] *= length_[1];
                cell_keys[iPart] += round( (position_y[iPart] - min_loc_y) * dx_inv_[1] );

            }
        }
    } else if (nDim_field == 1) {

        double min_loc_x = min_loc_vec[0];

        #pragma omp simd
        for( iPart=istart; iPart < iend ; iPart++  ) {
            if ( cell_keys[iPart] != -1 ) {
                //Compute cell_keys of remaining particles
                cell_keys[iPart] = round( (position_x[iPart] - min_loc_x) * dx_inv_[0] );
            }
        }

    }

    for( iPart=istart; iPart < iend ; iPart++  ) {
        if ( cell_keys[iPart] != -1 ) {
            count[cell_keys[iPart]] ++;
        }
    }
}

//Compute part_cell_keys at patch creation.
// This operation is normally done in the pusher to avoid additional particles pass.
void SpeciesV::computeParticleCellKeys( Params &params )
{

    //unsigned int ip
    unsigned int npart;
    // int IX;
    // double X;

    npart = particles->size(); //Number of particles

    int * __restrict__ cell_keys  = particles->getPtrCellKeys();

    // #pragma omp simd
    // for( ip=0; ip < npart ; ip++ ) {
    //     // Counts the # of particles in each cell (or sub_cell) and store it in sparticles->last_index.
    //     for( unsigned int ipos=0; ipos < nDim_field ; ipos++ ) {
    //         X = ((this)->*(distance[ipos]))(particles, ipos, ip);
    //         IX = round( X * dx_inv_[ipos] );
    //         particles->cell_keys[ip] = particles->cell_keys[ip] * this->length_[ipos] + IX;
    //     }
    // }
    // for( ip=0; ip < npart ; ip++ ) {
    //     count[particles->cell_keys[ip]] ++ ;
    // }

    computeParticleCellKeys( params, particles, cell_keys, &count[0], 0, npart );

}

void SpeciesV::importParticles( Params &params, Patch *patch, Particles &source_particles, vector<Diagnostic *> &localDiags )
{

    unsigned int npart = source_particles.size(), ncells=particles->first_index.size();

    // If this species is tracked, set the particle IDs
    if( particles->tracked ) {
        dynamic_cast<DiagnosticTrack *>( localDiags[tracking_diagnostic] )->setIDs( source_particles );
    }

    // compute cell keys of new parts
    vector<int> src_cell_keys( npart, 0 );
    vector<int> src_count( ncells, 0 );

    // for ( unsigned int ip = 0 ; ip < npart ; ip++ ) {
    //     for( unsigned int ipos=0; ipos < nDim_field ; ipos++ ) {
    //         double X = ((this)->*(distance[ipos]))(&source_particles, ipos, ip);
    //         int IX = round( X * dx_inv_[ipos] );
    //         src_cell_keys[ip] = src_cell_keys[ip] * length_[ipos] + IX;
    //     }
    // }
    // for( unsigned int ip=0; ip < npart ; ip++ ) {
    //     src_count[src_cell_keys[ip]] ++;
    // }

    computeParticleCellKeys( params,
                             &source_particles,
                             &src_cell_keys[0],
                             &src_count[0],
                             0,
                             npart );

    // sort new parts per cells
    int istart = 0;
    int istop  = src_count[0];

    for ( int icell = 0 ; icell < (int)ncells ; icell++ ) {
        if (src_count[icell]!=0) {
            for( int ip=istart; ip < istop ; ip++ ) {
                if ( src_cell_keys[ip] == icell )
                    continue;
                else { // rearrange particles
                    int ip_swap = istop;
                    while (( src_cell_keys[ip_swap] != icell ) && (ip_swap<(int)npart))
                        ip_swap++;
                    source_particles.swapParticle(ip, ip_swap);
                    int tmp = src_cell_keys[ip];
                    src_cell_keys[ip] = src_cell_keys[ip_swap];
                    src_cell_keys[ip_swap] = tmp;
                } // rearrange particles
            } // end loop on particles of a cell

            // inject in main data structure per cell
            source_particles.copyParticles( istart, src_count[icell],
                                        *particles,
                                        particles->first_index[icell] );
            particles->last_index[icell] += src_count[icell];
            for ( unsigned int idx=icell+1 ; idx<particles->last_index.size() ; idx++ ) {
                particles->first_index[idx] += src_count[icell];
                particles->last_index[idx]  += src_count[icell];
            }
            count[icell] += src_count[icell];

        }
        // update istart/istop fot the next cell
        istart += src_count[icell];
        if ( icell != (int)ncells-1  )
            istop  += src_count[icell+1];
        else
            istop = npart;

    } // End cell loop
    //source_particles.clear();

    // Set place for new particles in species->particles->cell_keys
    for (unsigned int ip=0;ip<npart ; ip++ )
        addSpaceForOneParticle();

    source_particles.clear();

}

// ---------------------------------------------------------------------------------------------------------------------
//! Particle merging cell by cell
// ---------------------------------------------------------------------------------------------------------------------
void SpeciesV::mergeParticles( double time_dual, unsigned int ispec,
                               Params &params,
                               Patch *patch, SmileiMPI *smpi,
                               std::vector<Diagnostic *> &localDiags )
{
//     int ithread;
// #ifdef _OPENMP
//     ithread = omp_get_thread_num();
// #else
//     ithread = 0;
// #endif


    // Only for moving particles
    if( time_dual>time_frozen_ ) {

        unsigned int scell ;
        // double weight_before = 0;
        // double weight_after = 0;
        // double energy_before = 0;
        // double energy_after = 0;
        std::vector <int> mask(particles->last_index.back(), 1);

        // Resize the cell_keys
        // particles->cell_keys.resize( particles->last_index.back(), 1 );
        // #pragma omp simd
        // for (unsigned int ip = 0; ip < (unsigned int)(particles->last_index.back()) ; ip++) {
        //         particles->cell_keys[ip] = 1;
        // }

        // for (unsigned int ip = 0; ip < (unsigned int)(particles->last_index.back()) ; ip++) {
        //         weight_before += particles->weight(ip);
        //         energy_before += sqrt(1 + pow(particles->momentum(0,ip),2) + pow(particles->momentum(1,ip),2) + pow(particles->momentum(2,ip),2));
        // }

        // For each cell, we apply independently the merging process
        for( scell = 0 ; scell < particles->first_index.size() ; scell++ ) {

            ( *Merge )( mass_, *particles, mask, smpi, particles->first_index[scell],
                        particles->last_index[scell], count[scell]);

        }

        // We remove empty space in an optimized manner
        particles->eraseParticlesWithMask(0, particles->last_index.back(), mask);

        // Update of first and last cell indexes
        particles->first_index[0] = 0;
        particles->last_index[0] = count[0];
        for( scell = 1 ; scell < particles->first_index.size(); scell++ ) {
            particles->first_index[scell] = particles->last_index[scell-1];
            particles->last_index[scell] = particles->first_index[scell] + count[scell];
        }

        //particles->cell_keys.resize(particles->last_index.back());

        // -------------------------------------------------------------------------------------
        // Checkpoint for debugging

        // for (unsigned int ip = 0; ip < (unsigned int)(particles->last_index.back()) ; ip++) {
        //         weight_after += particles->weight(ip);
        //         energy_after += sqrt(1 + pow(particles->momentum(0,ip),2)
        //                      + pow(particles->momentum(1,ip),2)
        //                      + pow(particles->momentum(2,ip),2));
        // }
        //
        // if (weight_before != weight_after) {
        //     std::cerr
        //     << " Weight before: " << weight_before
        //     << " Weight after: " << weight_after
        //     << " Energy before: " << energy_before
        //     << " Energy after: " << energy_after
        //     << std::endl;
        // }
        // -------------------------------------------------------------------------------------

        // -------------------------------------------------------------------------------------
        // Checkpoint for debugging

        // for( scell = 0 ; scell < particles->first_index.size(); scell++ ) {
        //     for (unsigned int ip = particles->first_index[scell] ; ip  < particles->last_index[scell] ; ip ++) {
        //
        //         double xmin = patch->getDomainLocalMin(0);
        //         double xmax = patch->getDomainLocalMax(0);
        //         double ymin = patch->getDomainLocalMin(1);
        //         double ymax = patch->getDomainLocalMax(1);
        //
        //         double x = particles->position(0,ip);
        //         double y = particles->position(1,ip);
        //         double mx = particles->momentum(0,ip);
        //         double my = particles->momentum(1,ip);
        //         double mz = particles->momentum(2,ip);
        //         double v = sqrt(mx*mx+my*my+mz*mz)/sqrt(1+mx*mx+my*my+mz*mz);
        //         if (particles->cell_keys[ip] < 0 || particles->cell_keys[ip] > particles->last_index.back()) {
        //         std::cerr
        //         << " Npc: " << particles->last_index[scell] - particles->first_index[scell]
        //         << " Cell keys size: " << particles->cell_keys.size()
        //         << " ip: "<< ip
        //         << " cell_keys: " << particles->cell_keys[ip]
        //         << ", x: " << xmin
        //         << " < " << x
        //         << " < " << xmax
        //         << ", y: " << ymin
        //         << " < " << y
        //         << " < " << ymax
        //         << ", mx: " << mx
        //         << ", my: " << my
        //         << ", mz: " << mz
        //         << setprecision(10)
        //         << ", v: " << v
        //         << std::endl;
        //         ERROR("")
        //
        //         if (x <= xmin
        //             || x >= xmax
        //             || y <= ymin
        //             || y >= ymax
        //             || v >= 1) {
        //             ERROR("")
        //         }
        //         }
        //     }
        // }

        // -------------------------------------------------------------------------------------

    }
}


// ---------------------------------------------------------------------------------------------------------------------
// For all particles of the species reacting to laser envelope
//   - interpolate the fields at the particle position
//   - deposit susceptibility
//   - calculate the new momentum
// ---------------------------------------------------------------------------------------------------------------------
void SpeciesV::ponderomotiveUpdateSusceptibilityAndMomentum( double time_dual, unsigned int ispec,
        ElectroMagn *EMfields,
        Params &params, bool diag_flag,
        Patch *patch, SmileiMPI *smpi,
        std::vector<Diagnostic *> &localDiags )
{

////////////////////////////// new vectorized
    int ithread;
#ifdef _OPENMP
    ithread = omp_get_thread_num();
#else
    ithread = 0;
#endif

#ifdef  __DETAILED_TIMERS
    double timer;
#endif

    if( npack_==0 ) {
        npack_    = 1;
        packsize_ = ( f_dim1-2*oversize[1] );

        //if ( (long int)particles->last_index.back() < (long int)60000 || (Radiate) || (Ionize) || (Multiphoton_Breit_Wheeler_process) )
        packsize_ *= ( f_dim0-2*oversize[0] );
        //else
        //    npack_ *= (f_dim0-2*oversize[0]);

        if( nDim_particle == 3 ) {
            packsize_ *= ( f_dim2-2*oversize[2] );
        }
    }


    // -------------------------------
    // calculate the particle dynamics
    // -------------------------------
    if( time_dual>time_frozen_ || Ionize ) { // advance particle momentum

        for( unsigned int ipack = 0 ; ipack < npack_ ; ipack++ ) {

            // ipack start @ particles->first_index [ ipack * packsize_ ]
            // ipack end   @ particles->last_index [ ipack * packsize_ + packsize_ - 1 ]
            //int nparts_in_pack = particles->last_index[ (ipack+1) * packsize_-1 ] - particles->first_index [ ipack * packsize_ ];
            int nparts_in_pack = particles->last_index[( ipack+1 ) * packsize_-1 ];
            smpi->dynamics_resize( ithread, nDim_field, nparts_in_pack, params.geometry=="AMcylindrical" );

#ifdef  __DETAILED_TIMERS
            timer = MPI_Wtime();
#endif
            // Interpolate the fields at the particle position
            for( unsigned int scell = 0 ; scell < packsize_ ; scell++ ) {
                Interp->fieldsAndEnvelope( EMfields, *particles, smpi, &( particles->first_index[ipack*packsize_+scell] ), &( particles->last_index[ipack*packsize_+scell] ), ithread, particles->first_index[ipack*packsize_] );
            }
#ifdef  __DETAILED_TIMERS
            patch->patch_timers[7] += MPI_Wtime() - timer;
#endif

            // Ionization
            if( Ionize ) {

#ifdef  __DETAILED_TIMERS
                timer = MPI_Wtime();
#endif
                vector<double> *Epart = &( smpi->dynamics_Epart[ithread] );
                vector<double> *EnvEabs_part  = &( smpi->dynamics_EnvEabs_part[ithread] );
                vector<double> *EnvExabs_part = &( smpi->dynamics_EnvExabs_part[ithread] );
                vector<double> *Phipart = &( smpi->dynamics_PHIpart[ithread] );
                for( unsigned int scell = 0 ; scell < packsize_ ; scell++ ) {
                    Interp->envelopeFieldForIonization( EMfields, *particles, smpi, &( particles->first_index[ipack*packsize_+scell] ), &( particles->last_index[ipack*packsize_+scell] ), ithread );
                    Ionize->envelopeIonization( particles, particles->first_index[ipack*packsize_+scell], particles->last_index[ipack*packsize_+scell], Epart, EnvEabs_part, EnvExabs_part, Phipart, patch, Proj );
                }
#ifdef  __DETAILED_TIMERS
                patch->patch_timers[4] += MPI_Wtime() - timer;
#endif
            }

            if( time_dual<=time_frozen_ ) continue; // Do not push nor project frozen particles

            // Project susceptibility, the source term of envelope equation
#ifdef  __DETAILED_TIMERS
            timer = MPI_Wtime();
#endif
            for( unsigned int scell = 0 ; scell < packsize_ ; scell++ ) {
                Proj->susceptibility( EMfields, *particles, mass_, smpi, particles->first_index[ipack*packsize_+scell], particles->last_index[ipack*packsize_+scell], ithread, ipack*packsize_+scell, particles->first_index[ipack*packsize_] );
            }

#ifdef  __DETAILED_TIMERS
            patch->patch_timers[8] += MPI_Wtime() - timer;
#endif

            // Push the particles
#ifdef  __DETAILED_TIMERS
            timer = MPI_Wtime();
#endif
            ( *Push )( *particles, smpi, particles->first_index[ipack*packsize_], particles->last_index[ipack*packsize_+packsize_-1], ithread, particles->first_index[ipack*packsize_] );
#ifdef  __DETAILED_TIMERS
            patch->patch_timers[9] += MPI_Wtime() - timer;
#endif
        }

    } else { // immobile particle (at the moment only project density)

    }//END if time vs. time_frozen_

} // end ponderomotiveUpdateSusceptibilityAndMomentum

// ---------------------------------------------------------------------------------------------------------------------
// For all particles of the species reacting to laser envelope
//   - interpolate the fields at the particle position
//   - deposit susceptibility
// ---------------------------------------------------------------------------------------------------------------------
void SpeciesV::ponderomotiveProjectSusceptibility( double time_dual, unsigned int ispec,
        ElectroMagn *EMfields,
        Params &params, bool diag_flag,
        Patch *patch, SmileiMPI *smpi,
        std::vector<Diagnostic *> &localDiags )
{

////////////////////////////// new vectorized
    int ithread;
#ifdef _OPENMP
    ithread = omp_get_thread_num();
#else
    ithread = 0;
#endif

#ifdef  __DETAILED_TIMERS
    double timer;
#endif

    if( npack_==0 ) {
        npack_    = 1;
        packsize_ = ( f_dim1-2*oversize[1] );

        //if ( (long int)particles->last_index.back() < (long int)60000 || (Radiate) || (Ionize) || (Multiphoton_Breit_Wheeler_process) )
        packsize_ *= ( f_dim0-2*oversize[0] );
        //else
        //    npack_ *= (f_dim0-2*oversize[0]);

        if( nDim_particle == 3 ) {
            packsize_ *= ( f_dim2-2*oversize[2] );
        }
    }


    // -------------------------------
    // calculate the particle dynamics
    // -------------------------------
    if( time_dual>time_frozen_ ) { // advance particle momentum

        for( unsigned int ipack = 0 ; ipack < npack_ ; ipack++ ) {

            // ipack start @ particles->first_index [ ipack * packsize_ ]
            // ipack end   @ particles->last_index [ ipack * packsize_ + packsize_ - 1 ]
            //int nparts_in_pack = particles->last_index[ (ipack+1) * packsize_-1 ] - particles->first_index [ ipack * packsize_ ];
            int nparts_in_pack = particles->last_index[( ipack+1 ) * packsize_-1 ];
            smpi->dynamics_resize( ithread, nDim_field, nparts_in_pack, params.geometry=="AMcylindrical" );

#ifdef  __DETAILED_TIMERS
            timer = MPI_Wtime();
#endif
            // Interpolate the fields at the particle position
            for( unsigned int scell = 0 ; scell < packsize_ ; scell++ ) {
                Interp->fieldsAndEnvelope( EMfields, *particles, smpi, &( particles->first_index[ipack*packsize_+scell] ), &( particles->last_index[ipack*packsize_+scell] ), ithread, particles->first_index[ipack*packsize_] );
            }
#ifdef  __DETAILED_TIMERS
            patch->patch_timers[4] += MPI_Wtime() - timer;
#endif

            // Project susceptibility, the source term of envelope equation
#ifdef  __DETAILED_TIMERS
            timer = MPI_Wtime();
#endif
            for( unsigned int scell = 0 ; scell < packsize_ ; scell++ ) {
                Proj->susceptibility( EMfields, *particles, mass_, smpi, particles->first_index[ipack*packsize_+scell], particles->last_index[ipack*packsize_+scell], ithread, ipack*packsize_+scell, particles->first_index[ipack*packsize_] );
            }

#ifdef  __DETAILED_TIMERS
            patch->patch_timers[8] += MPI_Wtime() - timer;
#endif

        }

    } else { // immobile particle (at the moment only project density)

    }//END if time vs. time_frozen_

} // end ponderomotiveProjectSusceptibility

// ---------------------------------------------------------------------------------------------------------------------
// For all particles of the species reacting to laser envelope
//   - interpolate the ponderomotive potential and its gradient at the particle position, for present and previous timestep
//   - calculate the new particle position
//   - particles BC
//   - project charge and current density
// ---------------------------------------------------------------------------------------------------------------------
void SpeciesV::ponderomotiveUpdatePositionAndCurrents( double time_dual, unsigned int ispec,
        ElectroMagn *EMfields,
        Params &params, bool diag_flag, PartWalls *partWalls,
        Patch *patch, SmileiMPI *smpi,
        std::vector<Diagnostic *> &localDiags )
{


    int ithread;
#ifdef _OPENMP
    ithread = omp_get_thread_num();
#else
    ithread = 0;
#endif

#ifdef  __DETAILED_TIMERS
    double timer;
#endif

    unsigned int iPart;

    int tid( 0 );
    std::vector<double> nrj_lost_per_thd( 1, 0. );

    // -------------------------------
    // calculate the particle dynamics
    // -------------------------------
    if( time_dual>time_frozen_ ) { // moving particle

        //Prepare for sorting
        for( unsigned int i=0; i<count.size(); i++ ) {
            count[i] = 0;
        }

        for( unsigned int ipack = 0 ; ipack < npack_ ; ipack++ ) {

            //int nparts_in_pack = particles->last_index[ (ipack+1) * packsize_-1 ] - particles->first_index [ ipack * packsize_ ];
            int nparts_in_pack = particles->last_index[( ipack+1 ) * packsize_-1 ];
            smpi->dynamics_resize( ithread, nDim_field, nparts_in_pack, params.geometry=="AMcylindrical" );

#ifdef  __DETAILED_TIMERS
            timer = MPI_Wtime();
#endif
            // Interpolate the fields at the particle position
            for( unsigned int scell = 0 ; scell < packsize_ ; scell++ ) {
                Interp->timeCenteredEnvelope( EMfields, *particles, smpi, &( particles->first_index[ipack*packsize_+scell] ), &( particles->last_index[ipack*packsize_+scell] ), ithread, particles->first_index[ipack*packsize_] );
            }
#ifdef  __DETAILED_TIMERS
            patch->patch_timers[10] += MPI_Wtime() - timer;
#endif

#ifdef  __DETAILED_TIMERS
            timer = MPI_Wtime();
#endif
            // Push only the particle position
            ( *Push_ponderomotive_position )( *particles, smpi, particles->first_index[ipack*packsize_], particles->last_index[ipack*packsize_+packsize_-1], ithread, particles->first_index[ipack*packsize_] );
#ifdef  __DETAILED_TIMERS
            patch->patch_timers[11] += MPI_Wtime() - timer;
            timer = MPI_Wtime();
#endif

            for( unsigned int scell = 0 ; scell < packsize_ ; scell++ ) {
                double energy_lost( 0. );
                // Apply wall and boundary conditions
                if( mass_>0 ) { // condition mass_>0
                    for( unsigned int iwall=0; iwall<partWalls->size(); iwall++ ) {
                        ( *partWalls )[iwall]->apply( this, particles->first_index[ipack*packsize_+scell], particles->last_index[ipack*packsize_+scell], smpi->dynamics_invgf[ithread], patch->rand_, energy_lost );
                        nrj_lost_per_thd[tid] += mass_ * energy_lost;
                    }

                    // Boundary Condition may be physical or due to domain decomposition
                    // apply returns 0 if iPart is not in the local domain anymore

                    partBoundCond->apply( this, particles->first_index[ipack*packsize_+scell], particles->last_index[ipack*packsize_+scell], smpi->dynamics_invgf[ithread], patch->rand_, energy_lost );
                    nrj_lost_per_thd[tid] += mass_ * energy_lost;

                    for( iPart=particles->first_index[ipack*packsize_+scell] ; ( int )iPart<particles->last_index[ipack*packsize_+scell]; iPart++ ) {
                        if ( particles->cell_keys[iPart] != -1 ) {
                            //First reduction of the count sort algorithm. Lost particles are not included.
                            for( int i = 0 ; i<( int )nDim_field; i++ ) {
                                particles->cell_keys[iPart] *= length_[i];
                                particles->cell_keys[iPart] += round( ((this)->*(distance[i]))(particles, i, iPart) * dx_inv_[i] );
                            }
                            count[particles->cell_keys[iPart]] ++; //First reduction of the count sort algorithm. Lost particles are not included.
                        }
                    }
                } else if( mass_==0 ) { // condition mass_=0
                    ERROR_NAMELIST( "Particles with zero mass cannot interact with envelope",
                    LINK_NAMELIST + std::string("#laser-envelope-model"));
                }
            }
            //START EXCHANGE PARTICLES OF THE CURRENT BIN ?
#ifdef  __DETAILED_TIMERS
            patch->patch_timers[3] += MPI_Wtime() - timer;
#endif

            // Project currents if not a Test species and charges as well if a diag is needed.
            // Do not project if a photon
#ifdef  __DETAILED_TIMERS
            timer = MPI_Wtime();
#endif
            if( ( !particles->is_test ) && ( mass_ > 0 ) )
                for( unsigned int scell = 0 ; scell < packsize_ ; scell++ ) {
                    Proj->currentsAndDensityWrapper( EMfields, *particles, smpi, particles->first_index[ipack*packsize_+scell], particles->last_index[ipack*packsize_+scell], ithread, diag_flag, params.is_spectral, ispec, ipack*packsize_+scell, particles->first_index[ipack*packsize_] );
                }

#ifdef  __DETAILED_TIMERS
            patch->patch_timers[12] += MPI_Wtime() - timer;
#endif
        }

        for( unsigned int ithd=0 ; ithd<nrj_lost_per_thd.size() ; ithd++ ) {
            nrj_bc_lost += nrj_lost_per_thd[tid];
        }

    } else { // immobile particle (at the moment only project density)
        if( diag_flag &&( !particles->is_test ) ) {
            if( params.geometry != "AMcylindrical" ) {
                double *b_rho = EMfields->rho_s[ispec] ? &( *EMfields->rho_s[ispec] )( 0 ) : &( *EMfields->rho_ )( 0 ) ;
                for( unsigned int scell = 0 ; scell < particles->first_index.size() ; scell ++ ) { //Loop for projection on buffer_proj
                    for( iPart=particles->first_index[scell] ; ( int )iPart<particles->last_index[scell]; iPart++ ) {
                        Proj->basic( b_rho, ( *particles ), iPart, 0 );
                    } //End loop on particles
                }//End loop on scells

            } else { // AM case
                ElectroMagnAM *emAM = static_cast<ElectroMagnAM *>( EMfields );
                int n_species = patch->vecSpecies.size();
                for( unsigned int imode = 0; imode<params.nmodes; imode++ ) {
                    int ifield = imode*n_species+ispec;
                    complex<double> *b_rho = emAM->rho_AM_s[ifield] ? &( *emAM->rho_AM_s[ifield] )( 0 ) : &( *emAM->rho_AM_[imode] )( 0 ) ;
                    for( unsigned int scell = 0 ; scell < particles->first_index.size() ; scell ++ ) { //Loop for projection on buffer_proj
                        for( int iPart=particles->first_index[scell] ; iPart<particles->last_index[scell]; iPart++ ) {
                            Proj->basicForComplex( b_rho, ( *particles ), iPart, 0, imode );
                        }
                    }
                }
            }
        }

    }//END if time vs. time_frozen_

} // end ponderomotiveUpdatePositionAndCurrents
