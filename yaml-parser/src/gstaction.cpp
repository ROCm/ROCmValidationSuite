#include <cstdlib>
#include "gstaction.h"

void add_action(std::vector<gst_action> &actions, std::string name,
                  std::string module_name,  int count,
                  std::string ops, float target_stress, int duration,
                  int size_a, int size_b, int size_c, int log_interval,
                  bool parallel, bool copy_matrix){
    gst_action f; 
    f.m_name = name;
    f.m_module_name = module_name;
		f.m_count = count;
		f.m_ops = ops;
		f.m_target_stress = target_stress;
		f.m_duration = duration;
    f.m_size_a = size_a;
    f.m_size_b = size_b;
    f.m_size_c = size_c;
    f.m_log_interval = log_interval;
    f.m_parallel = parallel;
		f.m_copy_matrix = copy_matrix;
		actions.push_back(f);
}

void destroy_actions(std::vector<gst_action> &actions){
}
