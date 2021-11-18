/*
 * Example application data structures.
 */
#ifndef GST_ACTION_H
#define GST_ACTION_H
#include <string>
#include <vector>

struct gst_action{
    gst_action *next;
    std::string m_name{};
    std::string m_module_name{};
    std::string m_devices{};
    std::string m_ops{};
    float m_target_stress;
    int m_count;
    int m_duration;
    int m_size_a;
    int m_size_b;
    int m_size_c;
    int m_log_interval;
    bool m_parallel;
    bool m_copy_matrix;
};


void add_action(std::vector<gst_action> &actions, std::string name, 
  std::string module_name,  int count,
  std::string ops, float target_stress, int duration,
  int size_a, int size_b, int size_c, int log_interval,
  bool parallel, bool copy_matrix);

void destroy_actions(std::vector<gst_action> &actions);
#endif 
