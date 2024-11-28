//
// Created by redamancyguy on 24-8-28.
//
#include <future>
#include <torch/torch.h>

#include "utils/dataset.hpp"
#include "utils/WorkLoad.hpp"
#include "utils/TimerClock.hpp"


#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "FLAREFast.hpp"

template<class T>
auto run_affix_fast(WorkLoad<T> &workload, FLAREFast<T> &index, num_t k_, num_t c, json &result) {
    // auto ground_truth = workload.groud_truth(*std::ranges::max_element(k_list));
    for (idx_t k: k_list) {
        double all_cost = 0;
        double avg_recall = 0;
        double over_ratio = 0;
        long long all_time = 0;
        long long all_net_work_time = 0;
        long long all_index_time = 0;
        long long all_scan_time = 0;
        long long all_sort_time = 0;
        TimerClock tc;

        // using namespace ;

        for (int _ = 0, kNN_query_size = workload.kNN_query_size(); _ < kNN_query_size; ++_) {
            torch::NoGradGuard no_grad;
            auto original_center = workload.get_kNN_query(_).first.view({1, -1});
            std::vector<idx_t> projected_ids;
            tc.tick();
            auto [network_time,index_time,scan_time,sort_time,cost] = index.AkNN(original_center, k, c, k_, projected_ids, workload);
            auto this_time = tc.nanoSec();
            all_time += this_time;
            all_net_work_time += network_time;
            all_index_time += index_time;
            all_scan_time += scan_time;
            all_sort_time += sort_time;
            all_cost += cost;
            auto cmp_func = [](const std::pair<num_t, idx_t> &a, const std::pair<num_t, idx_t> &b) {
                return a.first < b.first;
            };
            std::vector<idx_t> original_ids;
            workload.kNN(workload.get_kNN_query_ptr(_).first, k, original_ids);
            std::vector<num_t> original_dis_list;
            std::vector<num_t> projected_dis_list;
            for (int i = 0; i < k; ++i) {
                auto original_dis_ = calculate::dis_l2(workload.dim, original_center.template data_ptr<float>(), workload[original_ids[i]]);
                original_dis_list.push_back(original_dis_);
                auto projected_dis_ = calculate::dis_l2(workload.dim, original_center.template data_ptr<float>(), workload[projected_ids[i]]);
                projected_dis_list.push_back(projected_dis_);
            }
            std::ranges::sort(original_dis_list);
            std::ranges::sort(projected_dis_list);
            double weight_sum = 0;
            double over_ratios_sum = 0;
            for (int i = 0; i < k; ++i) {
                double over = 0;
                if (original_dis_list[i] == 0) {
                    over = 1;
                } else {
                    over = projected_dis_list[i] / original_dis_list[i];
                }
                auto weight = 100 - i;
                weight = 1;
                over_ratios_sum += over * weight;
                weight_sum += weight;
            }

            over_ratio += over_ratios_sum / weight_sum;
            const auto equal_num = check_result(original_ids, projected_ids);
            auto dis_func = [&](idx_t id) {
                return calculate::dis_l2(workload.dim, original_center.template data_ptr<float>(), workload[id]);
            };
            avg_recall += static_cast<double>(equal_num) / k;
        }
        avg_recall /= workload.kNN_query_size();
        over_ratio /= workload.kNN_query_size();
        result[std::to_string(k)]["kNN"] = all_time / static_cast<double>(workload.kNN_query_size());
        result[std::to_string(k)]["net-work-time"] = all_net_work_time / static_cast<double>(workload.kNN_query_size());
        result[std::to_string(k)]["index-time"] = all_index_time / static_cast<double>(workload.kNN_query_size());
        result[std::to_string(k)]["scan-time"] = all_scan_time / static_cast<double>(workload.kNN_query_size());
        result[std::to_string(k)]["sort-time"] = all_sort_time / static_cast<double>(workload.kNN_query_size());
        result[std::to_string(k)]["cost"] = all_cost / static_cast<double>(workload.kNN_query_size());
        result[std::to_string(k)]["recall"] = avg_recall;
        result[std::to_string(k)]["over-ratio"] = over_ratio;
    }
    return result;
}


int main(int argc, char *argv[]) {
    std::string dataset = "TinyImages";
    dataset = "Word2Vec";
    dataset = "MIRFLICKR";
    dataset = "Sift1B";
    dataset = "Deep1M";
    dataset = "MNIST";
    dataset = "Gist";
    dataset = "Sift1M";
    dataset = "Audio";
    dataset = "GoogleEarth";
    int adjust_with_skewness = false;
    // dataset_size = 2e7;
    dataset_size = 1e6;
    dataset_size = 1e7;
    init_epochs = 10;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
            k_list = parse_k_values(argv[++i]);
        }
        if (strcmp(argv[i], "--datasize") == 0 && i + 1 < argc) {
            dataset_size = std::atoll(argv[++i]);
        }
        if (strcmp(argv[i], "--dataset") == 0 && i + 1 < argc) {
            dataset = std::string(argv[++i]);
        }
        if (strcmp(argv[i], "--adjust_with_skewness") == 0 && i + 1 < argc) {
            adjust_with_skewness = std::atoi(argv[++i]);
        }
    }
    init_epochs = static_cast<float>(init_epochs) * std::pow(std::max<float>(1, 1e6 / dataset_size), 0.5);
    std::cout << "AFFIX dataset: " << dataset << " datasize:" << dataset_size << " k:" << k_list << "adjust_with_skewness:" << static_cast<bool>(adjust_with_skewness) << std::endl;
    auto workload = WorkLoad<float>(dataset, dataset_size, point_query_num,
                                    static_cast<idx_t>(range_query_num), static_cast<idx_t>(range_query_num),
                                    static_cast<idx_t>(kNN_query_num), 2.5);
    int64_t output_dim = 12;
    std::map<std::string, float> skewness_dict = {
            {"Word2Vec", 0.1181032983885169},
            {"Deep1M", 0.13595005477806105},
            {"Sift1B", 0.10160469578076116},
            {"GoogleEarth", 0.017509608867848317},
            {"MIRFLICKR", 0.015671104072814575},
            {"TinyImages", 0.05490293569658989},
            {"MNIST", 0.07302957646880387},
            {"Gist", 0.10082281605404843},
            {"Sift1M", 0.0948954985171799},
            {"CIFAR", 0.03569944349565443},
            {"Audio", 0.04413142573644204}
    };
    auto skewness = skewness_dict[dataset];
    if(skewness < 0.1f) {
        output_dim = 12;
    } else if (skewness < 0.3f) {
        output_dim = 13;
    } else if (skewness >= 0.3f) {
        output_dim = 14;
    }
    num_t c = 2;
    num_t k_ = 5 * std::pow(static_cast<num_t>(workload.size), 0.7);
    // k_ *= 2;
    std::cout << c << ":" << k_ << std::endl;
    json result;
    json all_result;
    FLAREFast<float> index(workload, output_dim, result, 256, 2);
    all_result[dataset][std::to_string(workload.size)] = run_affix_fast(workload, index, k_, c, result);
    ERRLN(all_result.dump());
    return 0;
}
