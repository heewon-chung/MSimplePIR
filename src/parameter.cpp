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

parameter::parameter(const database& db, const int degree, const int rank)
{
    int numCol = db.getNumCol();
    int numRow = db.getNumRow();

    // p = k 2^m + 1
    // 2m-th root of unity
    // ctxt_modulus = 17;
    ctxt_modulus = 998244353;
    ptxt_modulus = 1000;
    scale = ctxt_modulus / ptxt_modulus;

    root = 3;
    numInstance = numRow / degree;

    // initiate CRS for MLWE and LWE
    ringMatrix mlwe_crs(numInstance, vector<poly>(rank));
    matrix lwe_crs(numInstance * degree, vector<int64_t>(rank * degree));

    this -> degree = degree;
    this -> rank = rank;
    this -> mlwe_crs = mlwe_crs;
    this -> lwe_crs = lwe_crs;

    print();
}


void parameter::print()
{
    cout << "   - degree: " << degree << endl;
    cout << "   - rank: " << rank << endl;
    cout << "   - number of instances: " << numInstance << endl;
    cout << "   - MLWE CRS size: " << mlwe_crs.size() << " * " << mlwe_crs[0].size() << endl;
    cout << "   - PIR CRS size: " << lwe_crs.size() << " * " << lwe_crs[0].size() << endl;
}