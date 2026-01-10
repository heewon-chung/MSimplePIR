#ifndef __PARAMETER__
#define __PARAMETER__

#include <cstdint>
#include "util.h"
#include "database.h"

class parameter
{
private:
    int degree;
    int rank;
    int numInstance;

    int root;

    uint64_t ptxt_modulus;
    uint64_t ctxt_modulus;
    uint64_t scale;

    ringMatrix mlwe_crs;
    matrix lwe_crs;

public:
    parameter();
    parameter(const database& db, const int degree, const int rank, const uint64_t prime = 991);

    int getDegree() const {return degree;}
    int getRank() const {return rank;}
    int getNumInstance() const {return numInstance;}
    int getRoot() const {return root;}

    void setCRSforMLWE (ringMatrix crs) {mlwe_crs = crs;}
    ringMatrix getCRSforMLWE() const {return mlwe_crs;}

    void setCRSforLWE(matrix crs) {lwe_crs = crs;}
    matrix getCRSforLWE() const {return lwe_crs;}

    uint64_t getPtxtModulus() const {return ptxt_modulus;}
    uint64_t getCtxtModulus() const {return ctxt_modulus;}
    uint64_t getScale() const {return scale;}

    void print() const;
};


// class pir_parameter
// {
// private:
//     // uint64_t ctxt_modulus;
//     matrix crs;

// public:
//     pir_parameter()
//     {
//         // ctxt_modulus = 0;
//         crs.clear();
//     }
//     pir_parameter(database& db);  

//     // int getCtxtModulus() const {return ctxt_modulus;}
//     void setCRS(matrix crs) {this -> crs = crs;}
//     matrix getCRS() const {return crs;}

//     void print();
// };


#endif