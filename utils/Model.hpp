#ifndef MODEL_HPP
#define MODEL_HPP

#include <torch/torch.h>
#include <torch/nn.h>
#include <torch/nn/module.h>

inline auto CPU_DEVICE = torch::kCPU;
#ifdef UBUNTU_DESKTOP
inline auto GPU_DEVICE = torch::Device(torch::DeviceType::CUDA, 1);
// inline auto GPU_DEVICE = CPU_DEVICE;
#elif defined(UBUNTU_DESKTOP_WSL)
inline auto GPU_DEVICE = torch::Device(torch::DeviceType::CUDA, 1);
#elif defined(UBUNTU_NOTEBOOK)
inline auto GPU_DEVICE = torch::Device(torch::DeviceType::CUDA, 0);
#elif defined(UBUNTU_CPU_SERVER)
inline auto GPU_DEVICE = CPU_DEVICE;
#else
torch::Device GPU_DEVICE = torch::kCPU;
#endif

#include <ostream>
// #define usingDropOut
class FC final : public torch::nn::Module {
public:
    torch::nn::Linear ln = nullptr;
    torch::nn::BatchNorm1d bn = nullptr;
    // torch::nn::ReLU ac;
    torch::nn::ELU ac;
#ifdef usingDropOut
    torch::nn::Dropout dropout;
#endif

    FC(std::int_fast32_t input_size, std::int_fast32_t output_size, double dropout_prob = 0.2)
        : ln(register_module("fc", torch::nn::Linear(input_size, output_size)))
          ,bn(register_module("bn", torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(output_size))))
          ,ac(register_module("ac", torch::nn::ELU()))
          // ,ac(register_module("ac", torch::nn::ReLU()))
#ifdef usingDropOut
          ,dropout(register_module("dropout", torch::nn::Dropout(dropout_prob)))
#endif
    {
    }

    torch::Tensor forward(torch::Tensor x) {
        x = ln->forward(x);
        x = bn->forward(x);
        x = ac(x);
#ifdef usingDropOut
        x = dropout(x);
#endif
        return x;
    }
};
#endif //MODEL_HPP
