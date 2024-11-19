#include <iostream>
#include <algorithm>
#include <vector>
#include <iomanip>
#include <string>
#include <random>
#include <future>


#include <nlohmann/json.hpp>

#include "indexes/VPPLUS_OL.hpp"
using json = nlohmann::json;

#include "utils/DEFINE.h"


#include "utils/dataset.hpp"
#include "utils/WorkLoad.hpp"
#include "utils/TimerClock.hpp"
#include "indexes/RPTree.hpp"
#include "indexes/VPPLUS.hpp"
#include "indexes/MTree.hpp"
#include "indexes/VPTree.hpp"
#include "indexes/KDTree.hpp"
#include "indexes/MVPTree.hpp"

#include <array>
#include <utility> // for std::pair

TimerClock tc;




template<typename T>
json test(const WorkLoad<T> &workload, const Index<T> &index, const bool do_point = false, const bool do_range = false,
          const bool do_similar_range = false, const bool do_kNN = false, const bool do_AkNN = false,idx_t k = 100) {
    json exp_result;
    std::vector<idx_t> result;
    idx_t result_count = 0;
    idx_t total_time = 0;
    if (do_point) {
        for (idx_t i = 0; i < workload.point_query_size(); ++i) {
            tc.tick();
            result.clear();
            auto point_query = workload.get_point_query_ptr(i);
            index.point(point_query, result, workload);
            total_time += tc.nanoSec();
            result_count += static_cast<idx_t>(result.size());
#ifdef CHECK_POINT_RESULT
            std::vector<idx_t> another_result;
            workload.point(point_query, another_result);
            if (const auto equal_num = check_result(result, another_result); another_result.size() != equal_num) {
                std::cout << equal_num << "" << result.size() << ":" << another_result.size() << std::endl;
            }
#endif
        }
        exp_result["point"] = static_cast<num_t>(total_time) / workload.point_query_size();
        std::cout << "point:" << std::scientific << std::setprecision(4)
                << static_cast<num_t>(total_time) / workload.point_query_size() << "ns " << std::defaultfloat
                << static_cast<num_t>(result_count) / workload.point_query_size() << " per query" << std::endl;
#ifdef COUNT_SCAN_NUM
        std::cout << "point:" << "scan_node_count:" << static_cast<num_t>(scan_node_count) / workload.point_query_size()
                << " scan_count:" << static_cast<num_t>(scan_count) / workload.point_query_size() << std::endl;
        scan_node_count = 0;
        scan_count = 0;
#endif
    }

    if (do_range) {
        result_count = 0;
        total_time = 0;
        for (idx_t i = 0; i < workload.range_query_size(); ++i) {
            tc.tick();
            result.clear();
            auto range_query = workload.get_range_query_ptr(i);
            index.range(range_query.first, range_query.second, result, workload);
            total_time += tc.nanoSec();
            result_count += static_cast<idx_t>(result.size());
#ifdef CHECK_RANGE_RESULT
            std::vector<idx_t > another_result;
            workload.range(range_query.first,range_query.second,another_result);
            if(auto equal_num = check_result(result,another_result);another_result.size() != equal_num){
                std::cout <<equal_num<<":"<<result.size()<<":"<<another_result.size()<< std::endl;
            }
#endif
        }
        exp_result["range"] = static_cast<num_t>(total_time) / workload.range_query_size();
        std::cout << "range:" << std::scientific << std::setprecision(4)
                << static_cast<num_t>(total_time) / workload.range_query_size() << std::defaultfloat << "ns "
                << static_cast<num_t>(result_count) / workload.range_query_size() << " per query" << std::endl;

#ifdef COUNT_SCAN_NUM
        std::cout << "range:" << "scan_node_count:" << static_cast<num_t>(scan_node_count) / workload.range_query_size()
                << " scan_count:" << static_cast<num_t>(scan_count) / workload.range_query_size() << std::endl;
        scan_node_count = 0;
        scan_count = 0;
#endif
    }

    if (do_similar_range) {
        result_count = 0;
        total_time = 0;
        for (idx_t i = 0; i < workload.similar_range_query_size(); ++i) {
            tc.tick();
            result.clear();
            auto similar_range_query = workload.get_similar_range_query_ptr(i);
            index.similar_range(similar_range_query.first, similar_range_query.second,result, workload);
            total_time += tc.nanoSec();
            result_count += static_cast<idx_t>(result.size());
#ifdef CHECK_SIMILAR_RANGE_RESULT
            std::vector<idx_t> another_result;
            workload.similar_range(similar_range_query.first, similar_range_query.second, another_result);
            if (const auto equal_num = check_result(result, another_result); another_result.size() != equal_num) {
                std::cout << equal_num << ":" << result.size() << ":" << another_result.size() << std::endl;
            }
#endif
        }
        exp_result["similar_range"] = static_cast<double>(total_time) / workload.similar_range_query_size();
        std::cout << "similar_range:" << std::scientific << std::setprecision(4)
                << static_cast<double>(total_time) / workload.similar_range_query_size() << std::defaultfloat << "ns "
                << static_cast<num_t>(result_count) / workload.similar_range_query_size() << " per query" << std::endl;

#ifdef COUNT_SCAN_NUM
        std::cout << "similar_range:" << "scan_node_count:"
                << static_cast<num_t>(scan_node_count) / workload.similar_range_query_size()
                << " scan_count:" << static_cast<num_t>(scan_count) / workload.similar_range_query_size() << std::endl;
        scan_node_count = 0;
        scan_count = 0;
#endif
    }

    if (do_kNN) {
        result_count = 0;
        total_time = 0;
        progress_display display(workload.kNN_query_size());
        for (idx_t i = 0; i < workload.kNN_query_size(); ++i) {
            tc.tick();
            result.clear();
            auto kNN_query = workload.get_kNN_query_ptr(i);
            index.kNN(kNN_query.first, k, result, workload);
            auto time = tc.nanoSec();
            total_time += time;
            result_count += static_cast<idx_t>(result.size());
#ifdef CHECK_KNN_RESULT
            std::vector<idx_t> another_result;
            workload.kNN(kNN_query.first, k, another_result);
            if (const auto equal_num = check_result(result, another_result); another_result.size() != equal_num) {
                std::cout << equal_num << ":" << result.size() << ":" << another_result.size() << std::endl;
            }
            display.increment_with_data("query time:"+std::to_string(time/1e6)+" ms");
#endif
        }
        exp_result["kNN"] = static_cast<double>(total_time) / workload.kNN_query_size();
        std::cout << "kNN:" << std::scientific << std::setprecision(4)
                << static_cast<double>(total_time) / workload.kNN_query_size() << std::defaultfloat << "ns "
                << static_cast<num_t>(result_count) / workload.kNN_query_size() << " per query" << std::endl;
        exp_result["kNN-node"] = static_cast<num_t>(scan_node_count) / workload.kNN_query_size()/ workload.size;
        exp_result["cost"] = static_cast<num_t>(scan_count) / workload.kNN_query_size()/ workload.size;
#ifdef COUNT_SCAN_NUM
        std::cout << "kNN:" << "scan_node_count:" << static_cast<num_t>(scan_node_count) / workload.kNN_query_size()
                << " scan_count:" << static_cast<num_t>(scan_count) / workload.kNN_query_size() << std::endl;
        scan_node_count = 0;
        scan_count = 0;
#endif
    }

    if (do_AkNN) {
#ifdef CHECK_AKNN_RESULT
        idx_t is_nn_num = 0;
        idx_t all_nn_num = 0;
#endif

        result_count = 0;
        total_time = 0;
        for (idx_t i = 0; i < workload.kNN_query_size(); ++i) {
            tc.tick();
            result.clear();
            auto kNN_query = workload.get_kNN_query_ptr(i);
            index.AkNN(kNN_query.first, kNN_query.second, 2.0,result, workload);
            total_time += tc.nanoSec();
            result_count += static_cast<idx_t>(result.size());
#ifdef CHECK_AKNN_RESULT
            std::vector<idx_t> another_result;
            workload.kNN(kNN_query.first, kNN_query.second, another_result);
            const auto equal_num = check_result(result, another_result);
            is_nn_num += equal_num;
            all_nn_num += static_cast<idx_t>(another_result.size());
            // std::cout <<"result:"<<result.size()<<" another_result:"<<another_result.size()<<"  "<<static_cast<num_t>(equal_num)/static_cast<num_t>(another_result.size())<< std::endl;
#endif
        }
        exp_result["AkNN"] = static_cast<double>(total_time) / workload.kNN_query_size();
        std::cout << "AkNN:" << std::scientific << std::setprecision(4)
                << static_cast<double>(total_time) / workload.kNN_query_size() << std::defaultfloat << "ns  "
                << static_cast<num_t>(result_count) / workload.kNN_query_size() << " per query" << std::endl;
#ifdef CHECK_AKNN_RESULT
        std::cout << "reacall:" << static_cast<num_t>(is_nn_num) / static_cast<num_t>(all_nn_num) << std::endl;
#endif

#ifdef COUNT_SCAN_NUM
        std::cout << "AkNN:" << "scan_node_count:" << static_cast<num_t>(scan_node_count) / workload.kNN_query_size()
                << " scan_count:" << static_cast<num_t>(scan_count) / workload.kNN_query_size() << std::endl;
        scan_node_count = 0;
        scan_count = 0;
#endif
    }

    return exp_result;
}

// color::ColorManager color_manager;


template<class T>
auto run_workload(WorkLoad<T> &workload) {
    constexpr bool point = false;
    constexpr bool range = false;
    constexpr bool similar_range = false;
    constexpr bool kNN = true;
    constexpr bool AkNN = false;
    // color_manager.getNextColor();
    const idx_t before_memory = MemoryInfo();
    TimerClock construct_tc;
    const Index<T> *index;
    std::cout <<"Index:"<<indexTypeToString(index_type)<< std::endl;
    switch (index_type) {
        case LS: {
            index = new Index<T> (workload);
            break;
        }
        case RP: {
            // SP<T>::plane_selection_scheme = SP<T>::PCA_SGD;
            index = new RPTree<T> (workload);
            break;
        }
        case VPP_OL: {
            index = new VPPLUS_OL<T> (workload);
            break;
        }
        case VPP: {
            index = new VPPLUS<T> (workload);
            break;
        }
        case VP: {
            index = new VPTree<T> (workload);
            break;
        }
        case MVP: {
            index = new MVPTree<T> (workload);
            break;
        }
        case M: {
            index = new MTree<T>(workload);
            break;
        }
        case KD: {
            index = new KDTree<T>(workload);
            break;
        }
    }
    const num_t constructing_time = construct_tc.second();
    const idx_t after_memory = MemoryInfo();
    const idx_t used_memory = after_memory - before_memory;
    std::cout << "constructed:" << constructing_time << "s" << " memory:" << static_cast<num_t>(used_memory)/(1024*1024) << "MB" << std::endl;
    json result;
    result["construct"] = constructing_time;
    result["memory"] = used_memory;
    for(idx_t k : k_list) {
        result[std::to_string(k)] = test<T>(workload, *index, point, range, similar_range, kNN, AkNN,k);
    }
    delete index;
    return result;
}


int main(int argc, char* argv[]) {
    std::string index = "RP";
    std::string dataset = "MNIST";
#ifndef UBUNTU_CPU_SERVER
    dataset_size = 1e5;
#endif

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
            k_list = parse_k_values(argv[++i]);
        }
        if (strcmp(argv[i], "--datasize") == 0 && i + 1 < argc) {
            dataset_size = std::atoll(argv[++i]);
        }
        if (strcmp(argv[i], "--index") == 0 && i + 1 < argc) {
            index = std::string(argv[++i]);
        }
        if (strcmp(argv[i], "--dataset") == 0 && i + 1 < argc) {
            dataset = std::string(argv[++i]);
        }
    }
    index_type = stringToIndexType(index);
    std::cout << "dataset: " << dataset <<" index:"<<index<<" datasize:"<<dataset_size<<" k:"<<k_list<< std::endl;
    auto workload = WorkLoad<float>(dataset,dataset_size, point_query_num,
                                          static_cast<idx_t>(range_query_num), static_cast<idx_t>(range_query_num),
                                          static_cast<idx_t>(kNN_query_num),2.5);
    const auto result = run_workload<float>(workload);

    json all_result;
    all_result[dataset][std::to_string(workload.size)] = result;
    ERRLN(all_result);
    return 0;
}
