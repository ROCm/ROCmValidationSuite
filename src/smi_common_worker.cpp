#include <unistd.h>
#include <string>
#include <iostream>
#include <chrono>
#include <memory>
#include <exception>

#include "rocm_smi/rocm_smi.h"
#include "include/rvs_module.h"
#include "include/rvsloglp.h"

constexpr std::string RESULT_PASS_MESSAGE{"TRUE"};
constexpr std::string RESULT_FAIL_MESSAGE{"FALSE"};
constexpr std::string BLAS_FAILURE{"BLAS setup failed!"};
constexpr std::string SGEMM_FAILURE{"GPU failed to run the SGEMMs!"};
constexpr std::string TARGET_MESSAGE{"target"};
constexpr std::string DTYPE_MESSAGE{"dtype"};
constexpr std::string PWR_TARGET_ACHIEVED_MSG{"target achieved"};
constexpr std::string PWR_RAMP_EXCEEDED_MSG{"ramp time exceeded"};
constexpr std::string PASS_KEY{"pass"};

constexpr int MEM_ALLOC_ERROR=1;
constexpr int BLAS_ERROR=2;
constexpr int BLAS_MEMCPY_ERROR=3;
constexpr int MAX_MS_TRAIN_GPU=1000;
constexpr int MAX_MS_WAIT_BLAS_THREAD=10000;
constexpr int SGEMM_DELAY_FREQ_DEV=10;

constexpr std::string JSON_LOG_GPU_ID_KEY{"gpu_id"};
constexpr int TST_BLAS_ITERATIONS=25;
constexpr std::string LOG_GFLOPS_INTERVAL_KEY{"GFLOPS"};
constexpr std::string AVERAGE_EDGE_TEMP_KEY{"average edge temperature"};
constexpr std::string AVERAGE_JUNCTION_TEMP_KEY{"average junction temperature"};

static uint64_t time_diff_ms(
                std::chrono::time_point<std::chrono::system_clock> t_end,
                std::chrono::time_point<std::chrono::system_clock> t_start) {
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
                            t_end - t_start);
    return milliseconds.count();
}

void smi_worker::log_to_json(const std::string &key, const std::string &value,
                     int log_level) {
        if(!smi_worker::bjson)
                return;
        void *json_node = json_node_create(std::string(module_name),
                            action_name.c_str(), log_level);
        if (json_node) {
            rvs::lp::AddString(json_node, JSON_LOG_GPU_ID_KEY,
                            std::to_string(gpu_id));
            rvs::lp::AddString(json_node, key, value);
            rvs::lp::LogRecordFlush(json_node);
        }
}

smi_worker::smi_worker():endtest(false) {
}

smi_worker::~smi_worker() {
}

void smi_worker::log_interval_gflops(double gflops_interval) {
    string msg;
    msg = " GPU flops :" + std::to_string(gflops_interval);
    rvs::lp::Log(msg, rvs::logtrace);
    log_to_json(LOG_GFLOPS_INTERVAL_KEY, std::to_string(gflops_interval),
                rvs::loginfo);

}

void smi_worker::blasThread(int gpuIdx, uint64_t matrix_size, std::string tst_ops_type,
    bool start, uint64_t run_duration_ms, int transa, int transb, float alpha, float beta,
    int tst_lda_offset, int tst_ldb_offset, int tst_ldc_offset){

    std::chrono::time_point<std::chrono::system_clock> tst_start_time, tst_end_time;
    double timetakenforoneiteration;
    double gflops_interval;
    double duration;
    uint64_t gem_ops;
    std::unique_ptr<rvs_blas> gpu_blas;
    rvs_blas *free_gpublas;
    string msg;

    duration = 0;
    gem_ops = 0;
   // setup rvsBlas
    gpu_blas = std::unique_ptr<rvs_blas>(new rvs_blas(gpuIdx,  matrix_size,  matrix_size,  matrix_size, transa, transb, alpha, beta,
          tst_lda_offset, tst_ldb_offset, tst_ldc_offset, tst_ops_type));

    //Genreate random matrix data
    gpu_blas->generate_random_matrix_data();

    //Copy data to GPU
    gpu_blas->copy_data_to_gpu(tst_ops_type);

    tst_start_time = std::chrono::system_clock::now();

    //Hit the GPU with load to increase temperature
    while ( (duration < run_duration_ms) && (endtest == false) ){
        //call the gemm blas
        gpu_blas->run_blass_gemm(tst_ops_type);

        /* Set callback to be called upon completion of blas gemm operations */
        gpu_blas->set_callback(blas_callback, (void *)this);

        std::unique_lock<std::mutex> lk(mutex);
        cv.wait(lk);

        if(!blas_status) {
          msg = "[" + action_name + "] " + module_name + " " +
            std::to_string(gpu_id) + " " + " BLAS gemm operations failed !!! ";
          rvs::lp::Log(msg, rvs::logtrace);
        }

        //get the end time
        tst_end_time = std::chrono::system_clock::now();
        //Duration in the call
        duration = time_diff(tst_end_time, tst_start_time);
        gem_ops++;

        //Converting microseconds to seconds
        timetakenforoneiteration = duration/1e6;
        //calculating Gemm count
        gflops_interval = gpu_blas->gemm_gflop_count()/timetakenforoneiteration;
        //Print the gflops interval
        log_interval_gflops(gflops_interval);
        // check end test to avoid unnecessary sleep
        if (endtest)
            break;
        //if gemm ops greater than 10000, lets yield
        //if this is not happening we are ending up in
        //out of memmory state
        if(gem_ops > 10000) {
            sleep(1);
            gem_ops = 0;
        }
    }
}

void smi_worker::blasThread(int gpuIdx, uint64_t matrix_size, std::string ops_type,
    bool start, uint64_t run_duration_ms, int transa, int transb, float alpha, float beta,
    int lda_offset, int ldb_offset, int ldc_offset){

    std::chrono::time_point<std::chrono::system_clock> test_start_time, test_end_time;
    double timetakenforoneiteration;
    double gflops_interval;
    double duration;
    uint64_t gem_ops;
    std::unique_ptr<rvs_blas> gpu_blas;
    rvs_blas *free_gpublas;
    string msg;

    duration = 0;
    gem_ops = 0;
   // setup rvsBlas
    gpu_blas = std::unique_ptr<rvs_blas>(new rvs_blas(gpuIdx,  matrix_size,  matrix_size,  matrix_size, transa, transb, alpha, beta,
          lda_offset, ldb_offset, ldc_offset, ops_type));

    //Genreate random matrix data
    gpu_blas->generate_random_matrix_data();

    //Copy data to GPU
    gpu_blas->copy_data_to_gpu(ops_type);

    test_start_time = std::chrono::system_clock::now();

    //Hit the GPU with load to increase temperature
    while ( (duration < run_duration_ms) && (endtest == false) ){
        //call the gemm blas
        gpu_blas->run_blass_gemm(ops_type);

        /* Set callback to be called upon completion of blas gemm operations */
        gpu_blas->set_callback(blas_callback, (void *)this);

        std::unique_lock<std::mutex> lk(mutex);
        cv.wait(lk);

        if(!blas_status) {
          msg = "[" + action_name + "] " + module_name + " " +
            std::to_string(gpu_id) + " " + " BLAS gemm operations failed !!! ";
          rvs::lp::Log(msg, rvs::logtrace);
        }

        //get the end time
        test_end_time = std::chrono::system_clock::now();
        //Duration in the call
        duration = time_diff(test_end_time, test_start_time);
        gem_ops++;

        //Converting microseconds to seconds
        timetakenforoneiteration = duration/1e6;
        //calculating Gemm count
        gflops_interval = gpu_blas->gemm_gflop_count()/timetakenforoneiteration;
        //Print the gflops interval
        log_interval_gflops(gflops_interval);
        // check end test to avoid unnecessary sleep
        if (endtest)
            break;
        //if gemm ops greater than 10000, lets yield
        //if this is not happening we are ending up in
        //out of memmory state
        if(gem_ops > 10000) {
            sleep(1);
            gem_ops = 0;
        }
    }
}



void smi_worker::run() {
    string msg, err_description;

    msg = "[" + action_name + "] " + module_name + " " +
            std::to_string(gpu_id) + " start " + std::to_string(target_metric);

    rvs::lp::Log(msg, rvs::loginfo);

    if (run_duration_ms < MAX_MS_TRAIN_GPU)
        run_duration_ms += MAX_MS_TRAIN_GPU;

    bool pass = do_stress();

    // check if stop signal was received
    if (rvs::lp::Stopping())
         return;

    msg = "[" + action_name + "] " + module_name + " " +
               std::to_string(gpu_id) + " " + PASS_KEY + ": " +
               (pass ? RESULT_PASS_MESSAGE : RESULT_FAIL_MESSAGE);
    rvs::lp::Log(msg, rvs::logresults);

    sleep(5);
}


void IETWorker::blas_callback (bool status, void *user_data) {

  if(!user_data) {
    return;
  }
  IETWorker* worker = (IETWorker*)user_data;

  /* Notify gst worker thread gemm operation completion */
  std::lock_guard<std::mutex> lk(worker->mutex);
  worker->blas_status = status;
  worker->cv.notify_one();
}
