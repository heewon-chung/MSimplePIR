#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <sys/resource.h>
#include <omp.h>

#include "database.h"
#include "simplepir.h"
#include "json.hpp"

using json = nlohmann::json;
using namespace std;

struct BenchmarkConfig {
    string output_file;
    int repetitions;
    int warmup_runs;
    double memory_limit_gb;
    vector<tuple<int, int, uint64_t>> db_sizes;
    vector<pair<int, int>> crypto_params;
    vector<int> threads;
};

struct BenchmarkResult {
    int db_rows, db_cols;
    int crypto_degree, crypto_rank;
    int thread_count;
    int repetition;
    int query_row, query_col;  // Random index queried
    double setup_ms, query_ms, answer_ms, recovery_ms;
    double mlwe_query_ms, lwe_query_ms;
    double query_speedup_ratio;
    double peak_mem_mb;
    double total_ms;  // Excludes recovery time
    bool success;
};

struct PreprocessedData {
    database db;
    parameter param;
    matrix hint_client;
    double setup_ms;

    PreprocessedData(int rows, int cols, int degree, int rank, uint64_t prime)
        : db(rows, cols), param(db, degree, rank, prime) {}
};

long double get_time() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0L + t.tv_nsec / 1000000.0L;
}

double get_memory_usage_mb() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss / 1024.0; // Convert KB to MB on Linux, already MB on macOS
}

BenchmarkConfig load_config(const string& config_file) {
    ifstream file(config_file);
    if (!file.is_open()) {
        throw runtime_error("Cannot open config file: " + config_file);
    }

    json j;
    file >> j;
    
    BenchmarkConfig config;
    config.output_file = j["output_file"];
    config.repetitions = j["repetitions"];
    config.warmup_runs = j.value("warmup_runs", 2);
    config.memory_limit_gb = j.value("memory_limit_gb", 16.0);
    
    for (const auto& db_size : j["db_sizes"]) {
        config.db_sizes.push_back(make_tuple(
            db_size["rows"], 
            db_size["cols"],
            db_size.value("prime", 991)  // 기본값 991
        ));
    }
    
    for (const auto& crypto_param : j["crypto_params"]) {
        config.crypto_params.push_back({crypto_param["degree"], crypto_param["rank"]});
    }
     
    for (const auto& thread_count : j["threads"]) {
        config.threads.push_back(thread_count);
    }
    
    return config;
}

bool estimate_memory_usage(int rows, int cols, int degree, int rank, double limit_gb) {
    // Rough estimation of memory usage
    double db_size_gb = (double(rows) * cols * sizeof(int64_t)) / (1024.0 * 1024.0 * 1024.0);
    double param_size_gb = (double(degree) * rank * sizeof(int64_t)) / (1024.0 * 1024.0 * 1024.0);
    double total_estimate_gb = db_size_gb + param_size_gb * 2; // Factor of 2 for safety
    
    return total_estimate_gb <= limit_gb;
}

// Run preprocessing once and return preprocessed data
PreprocessedData* run_preprocessing(int db_rows, int db_cols, uint64_t prime,
                                      int degree, int rank, int thread_count, int warmup_runs) {
    try {
        // Set number of threads
        omp_set_num_threads(thread_count);

        // Create preprocessed data
        PreprocessedData* data = new PreprocessedData(db_rows, db_cols, degree, rank, prime);

        // Warmup runs
        for (int i = 0; i < warmup_runs; i++) {
            matrix hint_warmup;
            setup(data->param, data->db, hint_warmup);
            vector<poly> qry_warmup, sk_warmup;
            query(data->param, 1, qry_warmup, sk_warmup);
            vector<int64_t> ans_warmup;
            answer(data->param, data->db, qry_warmup, ans_warmup);
            int64_t res_warmup = 0;
            recover(data->param, ans_warmup, hint_warmup, sk_warmup, 1, res_warmup);
        }

        // Measure setup time
        long double start = get_time();
        setup(data->param, data->db, data->hint_client);
        long double end = get_time();
        data->setup_ms = end - start;

        return data;

    } catch (const exception& e) {
        cerr << "Error in preprocessing: " << e.what() << endl;
        return nullptr;
    }
}

// Run query benchmark using preprocessed data with random index
BenchmarkResult run_query_benchmark(PreprocessedData* data, int thread_count, int repetition) {
    BenchmarkResult result;
    result.db_rows = data->db.getNumRow();
    result.db_cols = data->db.getNumCol();
    result.crypto_degree = data->param.getDegree();
    result.crypto_rank = data->param.getRank();
    result.thread_count = thread_count;
    result.repetition = repetition;
    result.setup_ms = data->setup_ms;  // Shared setup time
    result.success = false;

    try {
        // Set number of threads
        omp_set_num_threads(thread_count);

        // Generate random query index
        result.query_row = rand() % data->db.getNumRow();
        result.query_col = rand() % data->db.getNumCol();

        long double start, end;

        // Measure MLWE-based query
        vector<poly> qry, sk;
        start = get_time();
        query(data->param, result.query_col, qry, sk);
        end = get_time();
        result.query_ms = end - start;
        result.mlwe_query_ms = end - start;

        // Measure answer (server-side computation using MLWE query)
        vector<int64_t> ans;
        start = get_time();
        answer(data->param, data->db, qry, ans);
        end = get_time();
        result.answer_ms = end - start;

        // Measure recovery (excluded from total time)
        int64_t res = 0;
        start = get_time();
        recover(data->param, ans, data->hint_client, sk, result.query_row, res);
        end = get_time();
        result.recovery_ms = end - start;

        // Measure LWE-based query for comparison (SimplePIR baseline)
        vector<int64_t> qry_lwe;
        start = get_time();
        query(data->param, result.query_col, qry_lwe, data->param.getCtxtModulus());
        end = get_time();
        result.lwe_query_ms = end - start;

        // Calculate query speedup ratio (MLWE vs LWE)
        result.query_speedup_ratio = (result.lwe_query_ms > 0) ? (result.lwe_query_ms / result.mlwe_query_ms) : 0.0;

        // Total time excludes recovery (as requested: setup + query + answer only)
        result.total_ms = result.setup_ms + result.query_ms + result.answer_ms;
        result.peak_mem_mb = get_memory_usage_mb();

        // Verify correctness
        if (res == data->db.getDB()[result.query_row][result.query_col]) {
            result.success = true;
        } else {
            cerr << "Verification failed for index [" << result.query_row << "][" << result.query_col << "]" << endl;
        }

    } catch (const exception& e) {
        cerr << "Error in query benchmark: " << e.what() << endl;
        result.success = false;
    }

    return result;
}

void write_csv_header(ofstream& file) {
    file << "db_rows,db_cols,crypto_degree,crypto_rank,threads,repetition,query_row,query_col,"
         << "setup_ms,query_ms,answer_ms,recovery_ms,"
         << "mlwe_query_ms,lwe_query_ms,query_speedup_ratio,"
         << "total_ms,peak_mem_mb,success\n";
}

void write_csv_row(ofstream& file, const BenchmarkResult& result) {
    file << result.db_rows << "," << result.db_cols << ","
         << result.crypto_degree << "," << result.crypto_rank << ","
         << result.thread_count << "," << result.repetition << ","
         << result.query_row << "," << result.query_col << ","
         << fixed << setprecision(3)
         << result.setup_ms << "," << result.query_ms << ","
         << result.answer_ms << "," << result.recovery_ms << ","
         << result.mlwe_query_ms << "," << result.lwe_query_ms << ","
         << result.query_speedup_ratio << ","
         << result.total_ms << "," << result.peak_mem_mb << ","
         << (result.success ? "true" : "false") << "\n";
}

void print_progress(int current, int total, const string& description) {
    double progress = (double)current / total * 100.0;
    cout << "\r[" << setw(3) << (int)progress << "%] " << description << flush;
}

int main(int argc, char* argv[]) {
    string config_file = (argc > 1) ? argv[1] : "benchmark.json";
    
    try {
        BenchmarkConfig config = load_config(config_file);
        
        cout << "SimplePIR Benchmark Suite\n";
        cout << "========================\n";
        cout << "Configuration: " << config_file << "\n";
        cout << "Output file: " << config.output_file << "\n";
        cout << "Repetitions: " << config.repetitions << "\n";
        cout << "Warmup runs: " << config.warmup_runs << "\n";
        cout << "Memory limit: " << config.memory_limit_gb << " GB\n";
        cout << "Database sizes: " << config.db_sizes.size() << "\n";
        cout << "Crypto params: " << config.crypto_params.size() << "\n";
        cout << "Thread counts: " << config.threads.size() << "\n";
        
        // Calculate total number of benchmarks
        int total_benchmarks = 0;
        for (const auto& db_size : config.db_sizes) {
            for (const auto& crypto_param : config.crypto_params) {
                if (estimate_memory_usage(std::get<0>(db_size), std::get<1>(db_size), 
                                        crypto_param.first, crypto_param.second, 
                                        config.memory_limit_gb)) {
                    total_benchmarks += config.threads.size() * config.repetitions;
                }
            }
        }
        
        cout << "Total benchmarks to run: " << total_benchmarks << "\n\n";
        
        // Open output file
        ofstream output_file(config.output_file);
        if (!output_file.is_open()) {
            throw runtime_error("Cannot open output file: " + config.output_file);
        }
        
        write_csv_header(output_file);
        
        int current_benchmark = 0;

        // Seed random number generator for random query indices
        srand(time(nullptr));

        // Run benchmarks with optimized preprocessing
        for (const auto& db_size : config.db_sizes) {
            for (const auto& crypto_param : config.crypto_params) {
                // Check memory constraints
                if (!estimate_memory_usage(std::get<0>(db_size), std::get<1>(db_size),
                                         crypto_param.first, crypto_param.second,
                                         config.memory_limit_gb)) {
                    cout << "\nSkipping " << std::get<0>(db_size) << "x" << std::get<1>(db_size)
                         << " with degree=" << crypto_param.first
                         << " rank=" << crypto_param.second
                         << " (exceeds memory limit)\n";
                    continue;
                }

                for (int thread_count : config.threads) {
                    // Run preprocessing ONCE per parameter set
                    stringstream preproc_desc;
                    preproc_desc << "Preprocessing DB:" << std::get<0>(db_size) << "x" << std::get<1>(db_size)
                                 << " D:" << crypto_param.first << " R:" << crypto_param.second
                                 << " T:" << thread_count;
                    cout << "\n[PREP] " << preproc_desc.str() << flush;

                    PreprocessedData* preprocessed = run_preprocessing(
                        std::get<0>(db_size), std::get<1>(db_size), std::get<2>(db_size),
                        crypto_param.first, crypto_param.second,
                        thread_count, config.warmup_runs
                    );

                    if (!preprocessed) {
                        cerr << "\nPreprocessing failed, skipping this configuration\n";
                        continue;
                    }

                    cout << " (Setup: " << fixed << setprecision(2) << preprocessed->setup_ms << " ms)\n";

                    // Now run multiple query benchmarks reusing the same preprocessed data
                    for (int rep = 1; rep <= config.repetitions; rep++) {
                        current_benchmark++;

                        stringstream desc;
                        desc << "DB:" << std::get<0>(db_size) << "x" << std::get<1>(db_size)
                             << " D:" << crypto_param.first
                             << " R:" << crypto_param.second
                             << " T:" << thread_count
                             << " Rep:" << rep;

                        print_progress(current_benchmark, total_benchmarks, desc.str());

                        BenchmarkResult result = run_query_benchmark(preprocessed, thread_count, rep);

                        write_csv_row(output_file, result);
                        output_file.flush();
                    }

                    // Clean up preprocessed data
                    delete preprocessed;
                }
            }
        }
        
        cout << "\n\nBenchmark completed successfully!\n";
        cout << "Results saved to: " << config.output_file << "\n";
        
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}