
#ifndef _GST_WORKER_H_
#define _GST_WORKER_H_

#include <string>
#include "rvsthreadbase.h"

class GSTWorker : public rvs::ThreadBase {

public:
    GSTWorker();
    ~GSTWorker();

    void set_name(const std::string& name) { action_name = name; } 
    const std::string& get_name(void) { return action_name; }

    void set_gpu_id(uint16_t _gpu_id) { gpu_id = _gpu_id; }
    uint16_t get_gpu_id(void) { return gpu_id; }

    void set_gpu_device_index(int _gpu_device_index) { gpu_device_index = _gpu_device_index; }
    int get_gpu_device_index(void) { return gpu_device_index; }
    
    void set_run_wait_ms(unsigned long _run_wait_ms) { run_wait_ms = _run_wait_ms; }
    unsigned long get_run_wait_ms(void) { return run_wait_ms; }

    void set_run_duration_ms(unsigned long _run_duration_ms) { run_duration_ms = _run_duration_ms; }
    unsigned long get_run_duration_ms(void) { return run_duration_ms; }

    void set_ramp_interval(unsigned long _ramp_interval) { ramp_interval = _ramp_interval; }
    unsigned long get_ramp_interval(void) { return ramp_interval; }
    
    void set_log_interval(unsigned long _log_interval) { log_interval = _log_interval; }
    unsigned long get_log_interval(void) { return log_interval; }

    void set_max_violations(int _max_violations) { max_violations = _max_violations; }
    int get_max_violations(void) { return max_violations; }

    void set_copy_matrix(bool _copy_matrix) { copy_matrix = _copy_matrix; }
    bool get_copy_matrix(void) { return copy_matrix; }

    void set_target_stress(float _target_stress) { target_stress = _target_stress; }
    float get_target_stress(void) { return target_stress; }

    void set_tolerance(float _tolerance) { tolerance = _tolerance; }
    float get_tolerance(void) { return tolerance; }
    
protected:
    virtual void run(void);

protected:
    std::string	action_name;
    int gpu_device_index;
    uint16_t gpu_id;    
    unsigned long run_wait_ms;
    unsigned long run_duration_ms;
    unsigned long ramp_interval;
    unsigned long log_interval;
    int max_violations;
    bool copy_matrix;
    float target_stress;
    float tolerance;
    
};

#endif // _GST_WORKER_H_