#ifndef INTERPOLATOR3D2ORDERV_H
#define INTERPOLATOR3D2ORDERV_H


#include "Interpolator3D2Order.h"
#include "Field3D.h"
#include "Pragma.h"

//  --------------------------------------------------------------------------------------------------------------------
//! Class for vectorized 2nd order interpolator for 3d3v simulations
//  --------------------------------------------------------------------------------------------------------------------
class Interpolator3D2OrderV final : public Interpolator3D2Order
{

public:

    //! Creator for Interpolator3D2OrderV
    Interpolator3D2OrderV( Params &, Patch * );

    //! Destructor for Interpolator3D2OrderV
    ~Interpolator3D2OrderV() override final {};

    // inline void __attribute__((always_inline)) fields( ElectroMagn *EMfields, Particles &particles, int ipart, double *ELoc, double *BLoc );

    //! Interpolation of all fields and currents for a single particles located at istart.
    void fieldsAndCurrents( ElectroMagn *EMfields, Particles &particles, SmileiMPI *smpi, int *istart, int *iend, int ithread, LocalFields *JLoc, double *RhoLoc ) override final;

    //! Wrapper called by the particle dynamics section
    void fieldsWrapper( ElectroMagn *EMfields, Particles &particles, SmileiMPI *smpi, int *istart, int *iend, int ithread, int ipart_ref = 0 ) override final;

    // void fieldsSelection( ElectroMagn *EMfields, Particles &particles, double *buffer, int offset, std::vector<unsigned int> *selection ) override final {};

    //! Interpolator on another field than the basic ones
    void oneField( Field **field, Particles &particles, int *istart, int *iend, double *FieldLoc, double *l1=NULL, double *l2=NULL, double *l3=NULL ) override final;

    // //! Computation of a field from provided coefficients
    // inline double __attribute__((always_inline)) compute( double *coeffx, double *coeffy, double *coeffz, Field3D *f, int idx, int idy, int idz )
    // {
    //     double interp_res( 0. );
    //     //unroll ?
    //     for( int iloc=-1 ; iloc<2 ; iloc++ ) {
    //         for( int jloc=-1 ; jloc<2 ; jloc++ ) {
    //             for( int kloc=-1 ; kloc<2 ; kloc++ ) {
    //                 interp_res += *( coeffx+iloc ) * *( coeffy+jloc ) * *( coeffz+kloc ) * ( *f )( idx+iloc, idy+jloc, idz+kloc );
    //             }
    //         }
    //     }
    //     return interp_res;
    // };

    //! Interpolator specific to the envelope model
    void fieldsAndEnvelope( ElectroMagn *EMfields, Particles &particles, SmileiMPI *smpi, int *istart, int *iend, int ithread, int ipart_ref = 0 ) override final;

    //! Interpolator specific to the envelope model
    void timeCenteredEnvelope( ElectroMagn *EMfields, Particles &particles, SmileiMPI *smpi, int *istart, int *iend, int ithread, int ipart_ref = 0 ) override final;

    //! Interpolator specific to the envelope model
    void envelopeAndSusceptibility( ElectroMagn *EMfields, Particles &particles, int ipart, double *Env_A_abs_Loc, double *Env_Chi_Loc, double *Env_E_abs_Loc, double *Env_Ex_abs_Loc ) override final;

    //! Computation of a field from provided coefficients
    inline double __attribute__((always_inline)) computeV( int ipart, double *coeffx, double *coeffy, double *coeffz, Field3D *f, int *idx, int *dual, int idual )
    {
        double interp_res = 0.;
        int idx2[3];
        for( int i=0 ; i<3 ; i++ ) {
            idx2[i] = idx[i];
        }
        idx2[idual]++;
        for( int iloc=-1 ; iloc<2 ; iloc++ ) {
            for( int jloc=-1 ; jloc<2 ; jloc++ ) {
                for( int kloc=-1 ; kloc<2 ; kloc++ ) {
                    interp_res += *( coeffx+iloc*32 ) * *( coeffy+jloc*32 ) * *( coeffz+kloc*32 ) *
                                  ( ( 1-*( dual+( idual )*32 ) )*( *f )( idx[0]+iloc, idx[1]+jloc, idx[2]+kloc ) + *( dual+( idual )*32 )*( *f )( idx2[0]+iloc, idx2[1]+jloc, idx2[2]+kloc ) );
                }
            }
        }
        return interp_res;
    }

    //! Computation of a field from provided coefficients
    inline double __attribute__((always_inline)) computeV( int ipart, double *coeffx, double *coeffy, double *coeffz, Field3D *f, int *idx, int *dual, int idual0, int idual1 )
    {
        double interp_res = 0.;
        int idx2[3], idx3[3], idx4[3];
        for( int i=0 ; i<3 ; i++ ) {
            idx2[i] = idx[i];
            idx3[i] = idx[i];
            idx4[i] = idx[i];
        }
        idx2[idual0]++;
        idx3[idual1]++;
        idx4[idual0]++;
        idx4[idual1]++;
        for( int iloc=-1 ; iloc<2 ; iloc++ ) {
            for( int jloc=-1 ; jloc<2 ; jloc++ ) {
                for( int kloc=-1 ; kloc<2 ; kloc++ ) {
                    interp_res += *( coeffx+iloc*32 ) * *( coeffy+jloc*32 ) * *( coeffz+kloc*32 ) *
                                  ( ( 1-*( dual+( idual1 )*32 ) ) * ( ( 1-*( dual+( idual0 )*32 ) )*( *f )( idx [0]+iloc, idx [1]+jloc, idx [2]+kloc ) + *( dual+( idual0 )*32 )*( *f )( idx2[0]+iloc, idx2[1]+jloc, idx2[2]+kloc ) )
                                    + ( *( dual+( idual1 )*32 ) ) * ( ( 1-*( dual+( idual0 )*32 ) )*( *f )( idx3[0]+iloc, idx3[1]+jloc, idx3[2]+kloc ) + *( dual+( idual0 )*32 )*( *f )( idx4[0]+iloc, idx4[1]+jloc, idx4[2]+kloc ) ) );
                }
            }
        }
        return interp_res;
    }

private:


};//END class

#endif
