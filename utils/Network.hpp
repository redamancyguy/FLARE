//
// Created by redamancyguy on 24-8-21.
//

#ifndef NETWORK_H
#define NETWORK_H


#include <utility>
#include <torch/torch.h>
#include "DEFINE.h"
#include "Model.hpp"

class FC_NET final : public torch::nn::Module {
    std::shared_ptr<FC> input_layer;
    std::vector<std::shared_ptr<FC> > hidden_layers;
    torch::nn::Linear output{nullptr};

public:
    FC_NET(const int64_t input_dim, const int64_t output_dim, const int64_t hidden_layers_num = 2, const int64_t hidden_dim = 256) {
        input_layer = register_module("input_layer", std::make_shared<FC>(FC(input_dim, hidden_dim)));
        for (int64_t i = 0; i < hidden_layers_num; ++i) {
            hidden_layers.push_back(register_module("hidden_layer_" + std::to_string(i), std::make_shared<FC>(FC(hidden_dim, hidden_dim))));
        }
        output = register_module("output", torch::nn::Linear(hidden_dim, output_dim));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = input_layer->forward(x);
        for (const auto &hl: hidden_layers) {
            x = hl->forward(x);
        }
        x = output->forward(x);
        return x;
    }
};

inline torch::Tensor scalar(const torch::Tensor &X, const torch::Tensor &mean, const torch::Tensor &std_var) {
    return (X - mean) / std_var;
}

#include "calculate.hpp"

inline std::pair<torch::Tensor, torch::Tensor> generate_two_indices_pair(const std::int64_t n, const std::int64_t m) {
    std::vector<idx_t> ids = generate_random_indices(n);

    return std::make_pair(torch::tensor(std::vector(ids.begin(), ids.begin() + m), torch::kInt64), torch::tensor(std::vector(ids.begin() + m, ids.begin() + m + m), torch::kInt64));
}


inline std::pair<torch::Tensor, torch::Tensor> generate_two_indices(const std::int64_t n, const std::int64_t m) {
    if (2 * m < n) {
        return generate_two_indices_pair(n, m);
    }
    std::uniform_int_distribution<std::int64_t> dis(0, n - 1);
    std::vector<std::int64_t> ids0;
    std::vector<std::int64_t> ids1;
    for (std::int64_t i = 0; i < m; ++i) {
        ids0.push_back(dis(gen));
    }
    for (std::int64_t i = 0; i < m; ++i) {
        auto id = dis(gen);
        while (id == ids0[i]) {
            id = dis(gen);
        }
        ids1.push_back(id);
    }
    return std::make_pair(torch::tensor(ids0, torch::kInt64), torch::tensor(ids1, torch::kInt64));
}

inline auto L2_mean_dis(const torch::Tensor data1, const torch::Tensor data2) -> torch::Tensor {
    auto dis = data1 - data2;
    dis = dis * dis;
    dis = dis.mean(1);
    dis = torch::sqrt(dis);
    return dis;
}

inline auto L2_mean_dis_full(torch::Tensor A, torch::Tensor B) -> torch::Tensor {
    float dim = A.size(1);
    auto A_B = 2 * A.matmul(B.transpose(0, 1));
    A = A * A;
    A = A.sum(1, true);
    B = B * B;
    B = B.sum(1, true);
    B = B.transpose(0, 1);
    return ((A + B) - A_B) / dim;
}

#endif //NETWORK_H
