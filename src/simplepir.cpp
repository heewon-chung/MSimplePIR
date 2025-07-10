#include "simplepir.h"

void MLWEtoLWE(parameter& param)
{
    int rank = param.getRank();
    int degree = param.getDegree();
    int numInstance = param.getNumInstance();
    uint64_t ctxt_modulus = param.getCtxtModulus();
    // ringMatrix mlwe_crs = param.getCRSforMLWE();
    ringMatrix mlwe_crs;
    randMatrix(mlwe_crs, numInstance, rank, degree, ctxt_modulus); // move to setup

    assert(mlwe_crs.size() == numInstance);

    int numRow = numInstance * degree;
    int numCol = rank * degree;
    int rowIdx = 0;

    matrix lwe_crs(numRow, vector<int64_t>(numCol));

    #pragma omp parallel for
    for(int i = 0; i < numInstance; ++i)
    {
        int rowIdx = i * degree;
        for(int j = 0; j < rank; ++j)
        {
            poly tmp = mlwe_crs[i][j];
            for(int k = 0; k < degree; ++k)
            {
                int colIdx = j * degree + k;
                for(int deg = 0; deg < degree; ++deg)
                {
                    lwe_crs[rowIdx + deg][colIdx] = tmp[deg];
                }
                rotate(tmp.rbegin(), tmp.rbegin() + 1, tmp.rend());
                tmp[0] = ctxt_modulus - tmp[0];
            }
        }
    }

    param.setCRSforMLWE(mlwe_crs);
    param.setCRSforLWE(lwe_crs);

}


void setup(parameter& param, const database& db, matrix& hint_client)
{
    int db_col = db.getNumCol();
    int degree = param.getDegree();
    int rank = param.getRank();
    int numInstance = param.getNumInstance();
    uint64_t ctxtModulus = param.getCtxtModulus();

    MLWEtoLWE(param);
    // hint_client = DB * CRS
    matrixMultiply(db.getDB(), param.getCRSforLWE(), ctxtModulus, hint_client);

}

// void query(const parameter& param, const int col, vector<poly>& qry, vector<poly>& sk)
// {
//     int rank = param.getRank();
//     int degree = param.getDegree();
//     int numInstance = param.getNumInstance();
//     uint64_t ctxt_modulus = param.getCtxtModulus();
//     int root = param.getRoot();
//     const ringMatrix& mlwe_crs = param.getCRSforMLWE();

//     assert(mlwe_crs.size() == numInstance);
//     assert(mlwe_crs[0].size() == rank);
    
//     int blk = col / degree;
//     int pos = col % degree;
    
//     sk.resize(rank);
//     vector<poly> sk_ntt(rank);

//     // Generate random vectors and compute their NTT in parallel
//     #pragma omp parallel for
//     for(int j = 0; j < rank; j++)
//     {
//         randVector(sk[j], degree, ctxt_modulus); // TODO sample ternary for the efficient
//         sk_ntt[j] = poly(sk[j].begin(), sk[j].end());
//         sk_ntt[j].resize(2 * degree);
//         ntt(sk_ntt[j], ctxt_modulus, root);
//     }

//     // Initialize qry
//     qry.resize(numInstance, poly(degree, 0));

//     // Perform matrix-vector multiplication in parallel
//     // #pragma omp parallel for
//     for(int i = 0; i < numInstance; ++i)
//     {
//         vector<poly> tmp(rank, poly(2 * degree));
//         // Compute NTT of mlwe_crs rows
//         for(int j = 0; j < rank; ++j)
//         {
//             tmp[j] = poly(mlwe_crs[i][j].begin(), mlwe_crs[i][j].end());
//             tmp[j].resize(2 * degree);
//             ntt(tmp[j], ctxt_modulus, root);
//             multiply_ntt(tmp[j], sk_ntt[j], tmp[j], ctxt_modulus, root, true);
//             // Add tmp to qry[i] and ensure results are within modulus
//             for(int k = 0; k < degree; ++k)
//             {
//                 #pragma omp critical
//                 qry[i][k] = (qry[i][k] + tmp[j][k]) % ctxt_modulus;
//             }
//         }

//         for(int k = 0; k < degree; ++k)
//         {
//             qry[i][k] = (qry[i][k] + generateDiscreteGaussian(0, 6.4)) % ctxt_modulus;
//         }
//     }
//     qry[blk][pos] = (qry[blk][pos] + param.getScale()) % ctxt_modulus;
// }

void query(const parameter& param, const int col, vector<poly>& qry, vector<poly>& sk)
{
    int rank = param.getRank();
    int degree = param.getDegree();
    int numInstance = param.getNumInstance();
    uint64_t ctxt_modulus = param.getCtxtModulus();
    int root = param.getRoot();
    const ringMatrix& mlwe_crs = param.getCRSforMLWE();

    assert(mlwe_crs.size() == numInstance);
    assert(mlwe_crs[0].size() == rank);

    int blk = col / degree;
    int pos = col % degree;

    sk.resize(rank);
    vector<poly> sk_ntt(rank);

    // Generate random vectors and compute their NTT in parallel
    #pragma omp parallel for
    for (int j = 0; j < rank; j++)
    {
        randVector(sk[j], degree, ctxt_modulus); // TODO sample ternary for the efficient
        sk_ntt[j] = poly(sk[j].begin(), sk[j].end());
        sk_ntt[j].resize(2 * degree);
        ntt(sk_ntt[j], ctxt_modulus, root);
    }

    // Initialize qry
    qry.resize(numInstance, poly(degree, 0));

    // Precompute NTT of mlwe_crs rows and store in a temporary structure
    vector<vector<poly>> tmp_ntt(numInstance, vector<poly>(rank, poly(2 * degree)));
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < numInstance; ++i)
    {
        for (int j = 0; j < rank; ++j)
        {
            tmp_ntt[i][j] = poly(mlwe_crs[i][j].begin(), mlwe_crs[i][j].end());
            tmp_ntt[i][j].resize(2 * degree);
            ntt(tmp_ntt[i][j], ctxt_modulus, root);
        }
    }

    // Perform matrix-vector multiplication in parallel without critical sections
    #pragma omp parallel for
    for (int i = 0; i < numInstance; ++i)
    {
        // Use thread-local accumulator to avoid critical sections
        poly local_qry(degree, 0);
        
        for (int j = 0; j < rank; ++j)
        {
            poly tmp(2 * degree);
            multiply_ntt(tmp_ntt[i][j], sk_ntt[j], tmp, ctxt_modulus, root, true);
            for (int k = 0; k < degree; ++k)
            {
                local_qry[k] = (local_qry[k] + tmp[k]) % ctxt_modulus;
            }
        }

        // Add noise and store result (no race condition since each thread works on different i)
        for (int k = 0; k < degree; ++k)
        {
            qry[i][k] = (local_qry[k] + generateDiscreteGaussian(0, 6.4)) % ctxt_modulus;
        }
    }

    qry[blk][pos] = (qry[blk][pos] + param.getScale()) % ctxt_modulus;
}


void query(const parameter& param, const int qryCol, vector<int64_t>& qry, int64_t ctxt_modulus)
{
    const matrix& crs = param.getCRSforLWE();
    int scale = param.getScale();
    int numRow = crs.size();
    int numCol = crs[0].size();
    qry.resize(numRow, 0);

    vector<int64_t> sk_lwe(numCol);
    randVector(sk_lwe, numCol, ctxt_modulus);

    #pragma omp parallel for
    for (int i = 0; i < numRow; i++)
    {
        int64_t sum = 0;
        for (int j = 0; j < numCol; j++)
        {
            sum += crs[i][j] * sk_lwe[j];
        }
        // mean 0 & stddev 6.4
        qry[i] = (sum % ctxt_modulus + ctxt_modulus + generateDiscreteGaussian(0, 6.4)) % ctxt_modulus;
    }
    qry[qryCol] = (qry[qryCol] + scale) % ctxt_modulus;
}


void answer(const parameter& param, const database& db, const vector<poly>& qry, vector<int64_t>& ans)
{
    assert(qry.size() * qry[0].size() == db.getNumCol());
    vector<int64_t> qryLWE;
    qryLWE.reserve(qry.size() * qry[0].size());

    // Flatten the qry vector into qryLWE
    for(const auto& poly : qry)
    {
        qryLWE.insert(qryLWE.end(), poly.begin(), poly.end());
    }

    matrixMultiply(db.getDB(), qryLWE, param.getCtxtModulus(), ans);
}


void recover(const parameter& param, const vector<int64_t>& ans, const matrix& hint_client, const vector<poly>& sk, const int qryRow, int64_t& res)
{
    uint64_t ctxt_modulus = param.getCtxtModulus();
    int scale = param.getScale();
    res = ans[qryRow];

    int idx = 0;
    int64_t tmp = 0;

    for(const auto& poly : sk)
    {
        assert(poly.size() == param.getDegree());
        for(const auto& element : poly)
        {
            tmp += hint_client[qryRow][idx] * element;
            tmp = (tmp % ctxt_modulus + ctxt_modulus) % ctxt_modulus;
            idx++;
        }
    }
    res = (res - tmp) % ctxt_modulus;
    res = (res + ctxt_modulus) % ctxt_modulus;
    res = (2 * res + scale) / (scale << 1);

    assert(idx == static_cast<int>(sk.size() * sk[0].size()));
}