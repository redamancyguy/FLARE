//
// Created by redamancyguy on 24-10-27.
//

#ifndef AFFIX_FAST_H
#define AFFIX_FAST_H
#include "DEFINE.h"
#include "utils/Model.hpp"
#include "utils/Network.hpp"
#include "indexes/VPPLUSS.hpp"
inline float init_epochs = 5;
inline num_t training_scale = 1.075;

template<class T>
class FLAREFast {
    using SpatialIndex = VPPLUSS<T>;
    using NN = FC_NET;
    NN *net = nullptr;
    SpatialIndex *projected_index = nullptr;
    WorkLoad<T> *projected_workload = nullptr;
    torch::Tensor mean{};
    torch::Tensor std_var{};

public :
    auto &get_projected_workload() {
        return *projected_workload;
    }

    explicit FLAREFast(WorkLoad<T> &workload, idx_t output_dim, json &result, idx_t hidden_dim = 256, idx_t layers = 2) {
        MemoryLog memory_log;
        long long memory_occupy = 0;
        TimerClock tc;
        TimerClock tc2;
    const int64_t input_dim = workload.dim;
    float max_lr = 5e-3;
    const int64_t d_size = workload.size;
        int batch_size = 0.1 * 1024. * 1024.;
        std::cout << "output_dim:" << output_dim << " hidden_dim:" << hidden_dim << " layers:" << layers << std::endl;
        memory_log.tick();
        net = new NN(input_dim, output_dim, layers, hidden_dim);
        memory_occupy += memory_log.get_memory();
        torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(max_lr).weight_decay(1e-6));
        torch::nn::MSELoss loss_f;
        net->to(GPU_DEVICE);
        net->eval();
        tc.tick();
        const torch::Tensor &X = workload.dataset;
        mean = X.mean().to(GPU_DEVICE);
        std_var = torch::clamp(X.std().to(GPU_DEVICE), 1e-7);
        net->train();
        float epochs = init_epochs;
        int64_t backward_times = epochs * static_cast<double>(d_size) / batch_size;
        tc2.tick();
        double min_lr = 1e-5;
        double gamma = 0.9;
        double scheduler_steps = std::log(max_lr / min_lr) / std::log(1 / gamma);
        int step_size = std::max<int>(1, backward_times / scheduler_steps / 4);
        torch::optim::StepLR scheduler(optimizer, step_size, gamma);
        progress_display display(backward_times);
        for (int _ = 0, times = backward_times; _ < times; ++_) {
            // auto [batch_id0, batch_id1] = generate_two_indices(d_size, batch_size);
            auto batch_id0 = torch::randint(0, d_size, {batch_size}, torch::kInt64);
            auto batch_id1 = torch::randint(0, d_size, {batch_size}, torch::kInt64);
            torch::Tensor X0_batch = X.index_select(0, batch_id0).to(GPU_DEVICE);
            torch::Tensor X1_batch = X.index_select(0, batch_id1).to(GPU_DEVICE);
            X0_batch = scalar(X0_batch, mean, std_var);
            X1_batch = scalar(X1_batch, mean, std_var);
            auto y0_batch = net->forward(X0_batch.detach());
            auto y1_batch = net->forward(X1_batch.detach());

            auto original_dis = L2_mean_dis(X0_batch.detach(), X1_batch.detach());
            auto projected_dis = L2_mean_dis(y0_batch, y1_batch);
            auto loss_data = loss_f(original_dis, projected_dis);
            auto &loss = loss_data;
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
            double current_lr = optimizer.param_groups()[0].options().get_lr();
            std::ostringstream stream;
            stream << std::fixed << std::setprecision(10) << loss.item<float>() << " lr:" << current_lr;
            std::string formatted_loss = stream.str();
            display.increment_with_data("loss:" + formatted_loss);
            if (_ > 0.75 * times) {
                scheduler.step();
            }
        }
        double training_time = tc2.second();
        std::cout << "training-time:" << training_time << std::endl;
        std::cout << std::resetiosflags(std::ios_base::basefield)
                << std::resetiosflags(std::ios_base::showbase);

        net->eval();
        tc2.tick();
        memory_log.tick();
        torch::Tensor y = torch::zeros({d_size, output_dim});
        // memory_occupy += memory_log.get_memory();//do not need store data in the index
        for (int i = 0; i < d_size; i += batch_size) {
            torch::NoGradGuard no_grad;
            int end = std::min<int>(i + batch_size, d_size);
            torch::Tensor batch_X = X.narrow(0, i, end - i).to(GPU_DEVICE).detach();
            batch_X = scalar(batch_X, mean, std_var);
            torch::Tensor batch_result = net->forward(batch_X).to(CPU_DEVICE).detach();
            y.narrow(0, i, end - i).copy_(batch_result);
        }
        projected_workload = new WorkLoad<float>(std::move(y));
        auto dimension_reduction_time = tc2.second();
        std::cout << "dimension_reduction_time:" << dimension_reduction_time << std::endl;
#ifndef using_gpu
        mean = mean.to(CPU_DEVICE);
        std_var = std_var.to(CPU_DEVICE);
        net->to(CPU_DEVICE);
#endif
        memory_log.tick();
        tc2.tick();
        projected_index = new SpatialIndex(*projected_workload);
        projected_index->reorganize_tow_workload(*projected_workload, workload);
        auto index_construction_time = tc2.second();
        memory_occupy += memory_log.get_memory();
        std::cout << "index construct time:" << index_construction_time << std::endl;
        result["training-time"] = training_time;
        result["construct"] = tc.second();
        // result["construct"] = training_time + dimension_reduction_time + index_construction_time;
        result["memory"] = memory_occupy;
    }


    TimerClock tc;

    auto AkNN(const torch::Tensor &original_center, const idx_t k, num_t c, num_t projected_k, std::vector<idx_t> &projected_ids,
              const WorkLoad<T> &workload) {
        torch::NoGradGuard no_grad;
        tc.tick();
#ifdef using_gpu
            auto gpu_origin_query = original_center.to(GPU_DEVICE);
            gpu_origin_query = scalar(gpu_origin_query, mean, std_var);
            auto projected_center = net->forward(gpu_origin_query).to(CPU_DEVICE);
#   else
        const auto origin_query = scalar(original_center, mean, std_var);
        auto projected_center = net->forward(origin_query);
#endif
        long long network_time = tc.nanoSec();
        tc.tick();
        projected_index->AkNN(projected_center.data_ptr<float>(), projected_k, c, projected_ids, *projected_workload);
        long long index_time = tc.nanoSec();
        const idx_t cost_count = projected_ids.size();
        auto cmp_func = [](const std::pair<num_t, idx_t> &a, const std::pair<num_t, idx_t> &b) {
            return a.first < b.first;
        };
        tc.tick();
        std::vector<std::pair<num_t, idx_t> > temp_ids;
        for (auto i: projected_ids) {
            temp_ids.push_back({calculate::dis_l2(workload.dim, workload[i], original_center.template data_ptr<float>()), i});
        }
        long long scan_time = tc.nanoSec();
        tc.tick();
        std::nth_element(temp_ids.begin(), temp_ids.begin() + k, temp_ids.end(), cmp_func);
        temp_ids.resize(k);
        // auto time = tc.nanoSec();
        projected_ids.clear();
        for (auto snd: temp_ids | std::views::values) {
            projected_ids.push_back(snd);
        }
        long long sort_time = tc.nanoSec();
        return std::make_tuple(network_time, index_time, scan_time, sort_time, static_cast<double>(cost_count) / workload.size);
    }
};

#endif //AFFIX_FAST_H
