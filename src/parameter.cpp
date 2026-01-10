#include "parameter.h"

parameter::parameter()
{
    // security = 128;
    degree = 0;
    rank = 0;
    
    ptxt_modulus = 0;
    ctxt_modulus = 0;

    ringMatrix mlwe_crs;    mlwe_crs.clear();
    matrix lwe_crs;         lwe_crs.clear();

    this -> mlwe_crs = mlwe_crs;
    this -> lwe_crs = lwe_crs;
}

parameter::parameter(const database& db, const int degree, const int rank, const uint64_t prime)
{
    int numCol = db.getNumCol();
    int numRow = db.getNumRow();

    // p = k 2^m + 1
    // 2m-th root of unity
    // ctxt_modulus = 17;
    ctxt_modulus = 2013265921;
    root = 31;

    ptxt_modulus = prime;
    scale = ctxt_modulus / ptxt_modulus;

    numInstance = numRow / degree;

    // initiate CRS for MLWE and LWE
    ringMatrix mlwe_crs(numInstance, vector<poly>(rank));
    matrix lwe_crs(numInstance * degree, vector<int64_t>(rank * degree));

    this -> degree = degree;
    this -> rank = rank;
    this -> mlwe_crs = mlwe_crs;
    this -> lwe_crs = lwe_crs;

    printf("\nParameter Set\n");
    db.print();
    print();
}


void parameter::print() const
{
    cout << "   - degree: " << degree << endl;
    cout << "   - rank: " << rank << endl;
    cout << "   - number of instances: " << numInstance << endl;
    cout << "   - MLWE CRS size: " << mlwe_crs.size() << " * " << mlwe_crs[0].size() << endl;
    cout << "   - PIR CRS size: " << lwe_crs.size() << " * " << lwe_crs[0].size() << endl;
    cout << "   - Plaintext space: " << ptxt_modulus << endl;
}