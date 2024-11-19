//
// Created by redam on 7/17/2024.
//

#ifndef WORKLOAD_HPP
#define WORKLOAD_HPP
#include <functional>
#include <queue>
#include <ctime>
#include <ranges>
#include <algorithm>
#include <torch/torch.h>

#include "DEFINE.h"
#include "calculate.hpp"
#include "dataset.hpp"

class TensorStorage {
public:
    static void writeTensor(const std::string& filename, const torch::Tensor& tensor) {
        if (tensor.dim() != 2) {
            throw StringException("Error: Only 2D tensors are supported.");
        }

        std::ofstream outfile(filename, std::ios::binary);
        if (!outfile.is_open()) {
            throw StringException("Cannot open file " + filename + " for writing.");
        }

        int64_t rows = tensor.size(0);
        int64_t cols = tensor.size(1);

        outfile.write(reinterpret_cast<char*>(&rows), sizeof(int64_t));
        outfile.write(reinterpret_cast<char*>(&cols), sizeof(int64_t));
        outfile.write(reinterpret_cast<const char*>(tensor.data_ptr<float>()), rows * cols * sizeof(float));

        outfile.close();
    }

    static torch::Tensor readTensor(const std::string& filename, int64_t max_rows = -1) {
        std::ifstream infile(filename, std::ios::binary);
        if (!infile.is_open()) {
            throw StringException("Cannot open file " + filename + " for reading.");
        }

        int64_t rows, cols;
        infile.read(reinterpret_cast<char*>(&rows), sizeof(int64_t));
        infile.read(reinterpret_cast<char*>(&cols), sizeof(int64_t));

        if (max_rows < 0 || max_rows > rows) {
            max_rows = rows;
        }

        infile.seekg(0, std::ios::end);
        const int64_t file_size = infile.tellg();
        infile.seekg(sizeof(int64_t) * 2, std::ios::beg);

        if (const int64_t expected_size = sizeof(int64_t) * 2 + max_rows * cols * sizeof(float); file_size < expected_size) {
            infile.close();
            throw StringException("File size is insufficient. Expected at least " +
                              std::to_string(expected_size) + " bytes, but got " +
                              std::to_string(file_size) + " bytes.");
        }
        torch::Tensor tensor = torch::empty({max_rows, cols}, torch::kFloat32);
        infile.read(reinterpret_cast<char*>(tensor.data_ptr<float>()), max_rows * cols * sizeof(float));
        infile.close();
        return tensor;
    }

    static torch::Tensor vectorToTensor(const std::vector<std::vector<float>>& vec) {
        if (vec.empty() || vec[0].empty()) {
            throw StringException("Error: Input vector cannot be empty.");
        }

        int64_t rows = vec.size();
        int64_t cols = vec[0].size();
        torch::Tensor tensor = torch::empty({rows, cols}, torch::kFloat32);

        for (int64_t i = 0; i < rows; ++i) {
            for (int64_t j = 0; j < cols; ++j) {
                tensor[i][j] = vec[i][j];
            }
        }

        return tensor;
    }
};
inline idx_t expected_range_query_count = 200;
inline idx_t expected_kNN_query_count = 100;
inline std::vector<idx_t> k_list = {100};


inline idx_t count_equal_elements(const std::vector<idx_t> &arr1, const std::vector<idx_t> &arr2) {
    std::vector<idx_t> intersection;
    intersection.reserve(std::min(arr1.size(), arr2.size())); // Reserve space for efficiency
    std::ranges::set_intersection(arr1, arr2, std::back_inserter(intersection));
    return intersection.size();
}

inline auto check_result(std::vector<idx_t> result, std::vector<idx_t> another_result) {
    std::ranges::sort(result);
    std::ranges::sort(another_result);
    return count_equal_elements(result, another_result);
}

// inline std::vector<idx_t> k_list_default = {1,10,20,30,40,50,75,100};

template<typename DIS>
auto check_recall_list(std::vector<idx_t> &result, std::vector<idx_t> &another_result,DIS dis,std::vector<idx_t> &k_list) {
    auto cmp = [&](idx_t first,idx_t second){return dis(first) < dis(second);};
    std::ranges::sort(result,cmp);
    std::ranges::sort(another_result,cmp);
    std::vector<num_t> recalls;
    for(const auto k:k_list) {
        const auto id0 = std::vector(result.begin(),result.begin() +k);
        const auto id1 = std::vector(another_result.begin(),another_result.begin() +k);
        recalls.push_back(static_cast<num_t>(check_result(id0, id1))/static_cast<num_t>(k));
    }
    return recalls;
}

template <typename T, std::size_t DIM>
void replaceInvalidEntriesWithMean(std::vector<std::array<T, DIM>>& data) {
    T sum = 0;
    int count = 0;
    for (const auto& arr : data) {
        for (const auto& element : arr) {
            if (!std::isnan(element) && !std::isinf(element)) {
                sum += element;
                count++;
            }
        }
    }
    T mean = count > 0 ? sum / count : 0;
    for (auto& arr : data) {
        for (auto& element : arr) {
            if (std::isnan(element) || std::isinf(element)) {
                element = mean;
            }
        }
    }
}

template<typename U>
using get_query_type = std::conditional_t<(sizeof(U) <= 4), float, double>;

template<typename U>
constexpr torch::Dtype get_torch_type() {
    if constexpr (std::is_same_v<U, float>) {
        return torch::kFloat32;
    } else if constexpr (std::is_same_v<U, double>) {
        return torch::kFloat64;
    } else if constexpr (std::is_same_v<U, int>) {
        return torch::kInt32;
    } else if constexpr (std::is_same_v<U, long>) {
        return torch::kInt64;
    } else if constexpr (std::is_same_v<U, unsigned char>) {
        return torch::kUInt8;
    } else if constexpr (std::is_same_v<U, char>) {
        return torch::kInt8;
    } else {
        return torch::kFloat32; // as the default type
    }
}
inline void replace_invalid_with_mean(torch::Tensor& tensor) {
    return;
    if (!tensor.is_floating_point()) {
        throw std::invalid_argument("Tensor must be of floating point type.");
    }

    const auto is_nan = tensor.isnan();
    const auto is_inf = tensor.isinf();
    const auto is_invalid = is_nan | is_inf;

    for (int64_t i = 0; i < tensor.size(0); ++i) {
        auto row = tensor[i];
        auto row_is_invalid = is_invalid[i];
        auto row_valid_values = row.masked_select(~row_is_invalid);
        auto row_mean_value = row_valid_values.mean().item<float>();
        tensor[i].masked_fill_(row_is_invalid, row_mean_value);
    }
}

template<typename DT>
class WorkLoad {
    using QT = get_query_type<DT>;//query value type

    static constexpr torch::Dtype QTT = get_torch_type<QT>();
    static constexpr torch::Dtype DTT = get_torch_type<DT>();

    std::mt19937 gen;
    num_t data_density = 0;
    num_t radius_ratio = 0;
    torch::Tensor point_queries;
    torch::Tensor range_queries;
    torch::Tensor similar_range_queries; //center and radius
    torch::Tensor kNN_queries; //center and k

public:
    torch::Tensor dataset;
    idx_t dim;
    idx_t size;
    DT *data_ptr{nullptr};
    WorkLoad(): data_density(0),
                radius_ratio(0),
                point_queries(torch::zeros({0}, torch::TensorOptions().dtype(QTT))),
                range_queries(torch::zeros({0}, torch::TensorOptions().dtype(QTT))),
                similar_range_queries(torch::zeros({0}, torch::TensorOptions().dtype(QTT))),
                kNN_queries(torch::zeros({0}, torch::TensorOptions().dtype(QTT))),
                dataset(torch::zeros({0}, torch::TensorOptions().dtype(DTT))), dim(0), size(0) {
    }

    WorkLoad(const std::string & file_name,
             const idx_t n,
             const idx_t points,
             const idx_t ranges,
             const idx_t similar_ranges,
             const idx_t kNNs,
             const num_t radius_ratio = 1):radius_ratio(radius_ratio){
        load_dataset(file_name,n);
        replace_invalid_with_mean(dataset);
        update_info();
        gen = std::mt19937(42);
        gen_workload(points,ranges,similar_ranges,kNNs);
    }

    const DT *operator  [](const idx_t id) const {
        if(id >= size || id < 0) {
            throw StringException("bad id:"+std::to_string(id));
        }
        return data_ptr + dim * id;
    }

    std::vector<std::vector<idx_t>> groud_truth(idx_t k) const {
        const auto q_size = kNN_queries.size(0);
        std::vector<std::vector<std::pair<num_t, idx_t>>> pre_result(q_size);
        std::vector<std::vector<idx_t>> result(q_size);

        const auto tow_dim_ptr = reinterpret_cast<float(*)[dim]>(data_ptr);
        progress_display display(q_size);
        for(idx_t qi = 0;qi<q_size;++qi) {
            auto q_ptr = get_kNN_query_ptr(qi).first;
            for(idx_t di = 0;di<size;++di) {
                auto d_ptr = tow_dim_ptr[di];
                std::pair<num_t, idx_t> item = {calculate::dis_l2(dim,q_ptr,d_ptr),di};
                pre_result[qi].push_back(item);
            }
            ++display;
        }
        display.restart(q_size);
        for(idx_t qi = 0;qi<q_size;++qi) {
            std::ranges::partial_sort(pre_result[qi],pre_result[qi].begin()+k);
            ++display;
        }
        for(idx_t qi = 0;qi<q_size;++qi) {
            for(idx_t i = 0;i<k;++i) {
                result[qi].push_back(pre_result[qi][i].second);
            }
        }
        return result;
    }

    explicit WorkLoad(const torch::Tensor & tensor){
        gen = std::mt19937(42);
        dataset = tensor.to(DTT);
        update_info();
    }
    explicit WorkLoad(torch::Tensor && tensor){
        gen = std::mt19937(42);
        dataset = std::move(tensor);
        dataset = dataset.to(DTT);
        update_info();
    }

    void shuffle_data() {
        std::vector<int64_t> indices(size);
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 g(rd());
        std::ranges::shuffle(indices, g);
        torch::Tensor shuffled_matrix = dataset.index({torch::tensor(indices)});
        update_info();
    }

    void update_info() {
        data_ptr = static_cast<DT *>(dataset.data_ptr());
        size = dataset.size(0);
        dim = dataset.size(1);
    }

    idx_t point_query_size() const {
        return point_queries.size(0);
    }
    idx_t range_query_size() const {
        return range_queries.size(0);
    }
    idx_t similar_range_query_size() const {
        return similar_range_queries.size(0);
    }
    idx_t kNN_query_size() const {
        return kNN_queries.size(0);
    }

    const QT* get_point_query_ptr(const idx_t id) const {
        return point_queries[id].data_ptr<QT>();
    }

    std::pair<QT*,QT*> get_range_query_ptr(const idx_t id) const {
        auto min_ptr = range_queries[id].data_ptr<QT>();
        return std::make_pair(min_ptr,min_ptr + dim);
    }

    std::pair<QT*,QT> get_similar_range_query_ptr(const idx_t id) const {
        const auto vector = similar_range_queries[id];
        return std::make_pair(vector.data_ptr<QT>(),vector[dim].template item<QT>());
    }
    std::pair<QT*,QT> get_kNN_query_ptr(const idx_t id) const {
        const auto vector = kNN_queries[id];
        return std::make_pair(vector.data_ptr<QT>(),vector[dim].template item<QT>());
    }
    ///////////////////////
    torch::Tensor get_point_query(const idx_t id) const {
        return point_queries[id];
    }

    std::pair<torch::Tensor,torch::Tensor> get_range_query(const idx_t id) const {
        auto chunks = torch::chunk(range_queries[id], 2, 0);
        return {chunks[0], chunks[1]};
    }

    std::pair<torch::Tensor,QT> get_similar_range_query(const idx_t id) const {
        const auto vector = similar_range_queries[id];
        return std::make_pair(vector.data_ptr<QT>(),vector[dim].template item<QT>());
    }
    std::pair<torch::Tensor,QT> get_kNN_query(const idx_t id) const {
        const auto vector = kNN_queries[id];
        return std::make_pair(vector.slice(0, 0, dim),vector[dim].template item<QT>());
    }

    void load_dataset(std::string dataset_name,const idx_t n) {
        if(dataset_name == "TinyImages") {
            dataset = TensorStorage::readTensor(father_path+"tiny_images.tensor",n);
        }else if(dataset_name == "Gist") {
            dataset = TensorStorage::readTensor(father_path+"Gist.tensor",n);
        }else if(dataset_name == "GoogleEarth") {
            dataset = TensorStorage::readTensor(father_path+"google_earth.tensor",n);
        }else if(dataset_name == "Audio") {
            dataset = TensorStorage::readTensor(father_path+"Audio.tensor",n);
        }else if(dataset_name == "MNIST") {
            dataset = TensorStorage::readTensor(father_path+"MNIST.tensor",n);
        }else if(dataset_name ==  "CIFAR") {
            dataset = TensorStorage::readTensor(father_path+"CIFAR.tensor",n);
        }else if (dataset_name == "Sift1B") {
            dataset = TensorStorage::readTensor(father_path+"SIFT1B.tensor",n);
        }else if(dataset_name == "Deep1M") {
            dataset = TensorStorage::readTensor(father_path+"DEEP1M.tensor",n);
        }else if(dataset_name == "Sift1M") {
            dataset = TensorStorage::readTensor(father_path+"SIFT1M.tensor",n);
        }else if(dataset_name == "MIRFLICKR") {
            dataset = TensorStorage::readTensor(father_path+"MIRFLICKR.tensor",n);
        }else if(dataset_name == "Word2Vec") {
            dataset = TensorStorage::readTensor(father_path+"Word2Vec.tensor",n);
        }else if(dataset_name.ends_with("_projected")){
            dataset = TensorStorage::readTensor(father_path+ dataset_name + ".tensor",n);
        }else if(dataset_name == "TPC-H") {
            // auto dataset = TPC_H::read_csv_TCP_H<T>();
            // write<float,8>(TPC_H::tcp_bin_data_path,dataset);
            auto input_dataset = TPC_H::read_bin_TCP_H<DT>();
            dim = 8;
            if(input_dataset.size() > n) {
                input_dataset.resize(n);
            }
            size = input_dataset.size();
            dataset = torch::zeros({size,dim}, torch::TensorOptions().dtype(DTT));
            auto ptr = static_cast<DT(*)[dim]>(dataset.data_ptr());
            for(idx_t i = 0;i<size;++i) {
                std::copy_n(input_dataset[i].data(),dim,ptr[i]);
            }
        }
        std::cout <<"name:"<<dataset_name<<" size:"<<dataset.sizes()<<std::endl;
    }

    void gen_workload(idx_t points,
        idx_t ranges,
        idx_t similar_ranges,
        idx_t kNNs) {
        std::cout <<"gen workload:"<<" points:"<< points
        <<" ranges:"<<ranges
        <<" similar_ranges:"<<similar_ranges
        <<" kNNs:"<<kNNs<< std::endl;
        constexpr auto random_ratio = 0.5;
        const torch::Tensor std_var = torch::std(dataset);
        auto scale_ratio = std::pow(
                static_cast<double>(expected_range_query_count) / static_cast<double>(size),
                static_cast<double>(1) / static_cast<double>(dim));
        const auto max_values = std::get<0>(torch::max(dataset, /*dim=*/0));
        const auto min_values = std::get<0>(torch::min(dataset, /*dim=*/0));
        auto full_range = (max_values - min_values).to(QTT);
        data_density = static_cast<QT>(size)/torch::prod(full_range).template item<QT>();
        full_range *= scale_ratio;
        full_range /= 2.0; //left and right has half
        QT radius_mean = 0;
        auto d_ptr = reinterpret_cast<DT(*)[dim]>(data_ptr);
        for(idx_t i = 0;i<1000;++i) {
            const auto random_id_0 = gen() % size;
            const auto random_id_1 = gen() % size;
            auto dis = calculate::dis_l2(dim,d_ptr[random_id_0],d_ptr[random_id_1]);
            radius_mean += dis;
        }
        radius_mean/=(2 *1000);
        radius_mean *= scale_ratio;
        ///////////////
        std::uniform_int_distribution<idx_t> dis(0, size - 1);
        std::uniform_real_distribution<num_t> random_dis(-random_ratio,random_ratio);
        point_queries = torch::zeros({points,dim},torch::TensorOptions().dtype(QTT));
        for (idx_t i = 0; i < points; ++i) {
            const idx_t random_id = dis(gen);
            point_queries[i] = dataset[random_id] + std_var * random_dis(gen);
        }
        range_queries = torch::zeros({ranges,2 * dim},torch::TensorOptions().dtype(QTT));
        for (idx_t i = 0; i < ranges; ++i) {
            const idx_t random_id = dis(gen);
            auto center = dataset[random_id] + std_var * random_dis(gen);
            auto min = center - full_range;
            auto max = center + full_range;
            range_queries[i].slice(0, 0, dim).copy_(min);
            range_queries[i].slice(0, dim, 2 *dim).copy_(max);
        }

        similar_range_queries = torch::zeros({similar_ranges,dim + 1},torch::TensorOptions().dtype(QTT));
        for (idx_t i = 0; i < similar_ranges; ++i) {
            const idx_t random_id = dis(gen);
            auto chosen_center = dataset[random_id] + std_var * random_dis(gen);;
            similar_range_queries[i].slice(0, 0, dim).copy_(chosen_center);
            similar_range_queries[i][dim] = radius_mean * radius_ratio;
        }
        kNN_queries = torch::zeros({kNNs,dim + 1},torch::TensorOptions().dtype(QTT));
        for (idx_t i = 0; i < kNNs; ++i) {
            const idx_t random_id = dis(gen);
            auto chosen_center = dataset[random_id] + std_var * random_dis(gen);;
            kNN_queries[i].slice(0, 0, dim).copy_(chosen_center);
            kNN_queries[i][dim] = expected_kNN_query_count;
        }
    }
    bool check_same(const idx_t *begin, const idx_t *end) const {
        return false;
        for (auto i = begin; i < end - 1; ++i) {
            if (!calculate::equal(dim,dataset[*i].data_ptr<DT>(), dataset[*(i+1)].data_ptr<DT>())) {
                return false;
            }
        }
        return true;
    }

    void point(const QT * point, std::vector<idx_t >& result) const {
        auto d_ptr = reinterpret_cast<DT(*)[dim]>(data_ptr);
        for (idx_t i = 0; i < size; ++i) {
            if (calculate::equal(dim,d_ptr[i], point)) {
                result.push_back(i);
            }
        }
    }
     void range(const QT * min, const QT * max, std::vector<idx_t >& result) const {
        auto d_ptr = reinterpret_cast<DT(*)[dim]>(data_ptr);
        for (idx_t i = 0; i < size; ++i) {
            if (calculate::ge(dim,d_ptr[i], min) && calculate::lt(dim,d_ptr[i], max)) {
                result.push_back(i);
            }
        }
    }
     void similar_range(const num_t* center, const num_t radius, std::vector<idx_t >&result) const {
        auto ptr = reinterpret_cast<DT(*)[dim]>(data_ptr);
        for (idx_t i = 0; i < size; ++i) {
            if (calculate::dis_l2(dim,center, ptr[i]) <= radius) {
                result.push_back(i);
            }
        }
    }
     void kNN_old(const num_t* center, idx_t  k, std::vector<idx_t >& result) const {
        auto ptr = reinterpret_cast<DT(*)[dim]>(data_ptr);
        auto cmp_data = [&](const std::pair<num_t,idx_t> &first,const std::pair<num_t,idx_t> &second) { return first.first < second.first;};
        std::priority_queue<std::pair<num_t,idx_t> , std::vector<std::pair<num_t,idx_t> >, decltype(cmp_data)> dataPQ(cmp_data);
        for(idx_t i = 0;i<size;++i) {
            dataPQ.push(std::make_pair(calculate::dis_l2(dim,ptr[i],center),i));
            if(dataPQ.size() > k) { dataPQ.pop(); }
        }
        while (!dataPQ.empty()) {
            result.push_back(dataPQ.top().second);
            dataPQ.pop();
        }
    }

    void kNN(const num_t *center, const idx_t k, std::vector<idx_t> &result) const {
        auto ptr = reinterpret_cast<DT(*)[dim]>(data_ptr);
        auto cmp_data = [&](const std::pair<num_t, idx_t> &first, const std::pair<num_t, idx_t> &second) { return first.first < second.first; };
        std::vector<std::pair<num_t, idx_t> > dataPQ;
        for (idx_t i = 0; i < size; ++i) {
            dataPQ.push_back(std::make_pair(calculate::dis_l2(dim, ptr[i], center), i));
        }
        // std::ranges::nth_element(dataPQ, dataPQ.begin() + k);
        std::ranges::partial_sort(dataPQ, dataPQ.begin() + k,cmp_data);
        dataPQ.resize(k);
        for (auto snd: dataPQ | std::views::values) {
            result.push_back(snd);
        }
    }
};




inline auto sort_and_split(idx_t * begin, idx_t* end, const idx_t fanout, const std::function<num_t (idx_t)> &value_f) {
    std::sort(begin, end, [&](const idx_t a, const idx_t b) {
        return value_f(a) < value_f(b);
    });
    std::vector<idx_t> result;
    const auto total_size = static_cast<idx_t>(end - begin);
    const auto partition_size = static_cast<num_t>(total_size) / static_cast<num_t>(fanout);
    const auto max_search_steps = static_cast<idx_t>(std::ceil(total_size / fanout))/2;

    for (idx_t i = 0; i < fanout; ++i) {
        auto this_start_id = static_cast<idx_t>(std::round(partition_size * static_cast<num_t>(i + 1)));
        // Double-ended search for a valid split point with search step limit
        bool found = false;
        for (idx_t j = 0; j < max_search_steps; ++j) {
            idx_t left_id = this_start_id - j;
            idx_t right_id = this_start_id + j + 1;
            if(left_id >= total_size) {
                goto END;
            }
            if (left_id > 0 && value_f(begin[left_id]) != value_f(begin[left_id - 1])) {
                this_start_id = left_id;
                found = true;
                break;
            }
            if (right_id < total_size && value_f(begin[right_id]) != value_f(begin[right_id - 1])) {
                this_start_id = right_id;
                found = true;
                break;
            }
        }

        // If no valid split point found, break the loop
        if (!found) {
            break;
        }

        // Push split point if it is different from previous one and within range
        if (result.empty() || this_start_id > result.back()) {
            result.push_back(this_start_id);
        }
    }
    END:
    // Ensure the last segment includes the end
    if (result.empty() || result.back() != total_size) {
        result.push_back(total_size);
    }

    //check_result
    for(idx_t i = 0,end_id=result.size()-1;i<end_id;++i) {
        if(result[i] == result[i+1]) {
            throw StringException("equal split");
        }
        if(const auto id = result[i];value_f(begin[id]) == value_f(begin[id-1])) {
            throw StringException("equal value");
        }
    }
    return result;
}

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

// 对位加法
template <typename T>
std::vector<T> operator+(const std::vector<T>& lhs, const std::vector<T>& rhs) {
    std::vector<T> result(lhs.size());
    calculate::add(lhs.size(),lhs.data(),rhs.data(),result.data());
    return result;
}

// 对位加赋值
template <typename T>
std::vector<T>& operator+=(std::vector<T>& lhs, const std::vector<T>& rhs) {
    calculate::add_from(lhs.size(),lhs.data(),rhs.data());
    return lhs;
}

// 对位减法
template <typename T>
std::vector<T> operator-(const std::vector<T>& lhs, const std::vector<T>& rhs) {
    std::vector<T> result(lhs.size());
    calculate::sub(lhs.size(),lhs.data(),rhs.data(),result.data());
    return result;
}

// 对位减赋值
template <typename T>
std::vector<T>& operator-=(std::vector<T>& vec, const std::vector<T>& rhs) {
    calculate::sub_from(vec.size(),vec.data(),rhs.data());
    return vec;
}

// 数值乘法
template <typename T, typename Num>
std::vector<T> operator*(const std::vector<T>& vec, Num num) {
    std::vector<T> result(vec.size());
    calculate::mul(vec.size(),vec.data(),num,result.data());
    return result;
}

// 数值乘赋值
template <typename T, typename Num>
std::vector<T>& operator*=(std::vector<T>& vec, Num num) {
    calculate::mul_by(vec.size(),vec.data(),num);
    return vec;
}

// 数值除法
template<typename T, typename Num>
std::vector<T> operator/(const std::vector<T> &vec, Num num) {
    std::vector<T> result(vec.size());
    calculate::div(vec.size(), vec.data(), num, result.data());
    return result;
}

// 数值除赋值
template <typename T, typename Num>
std::vector<T>& operator/=(std::vector<T>& vec, Num num) {
    calculate::div_by(vec.size(),vec.data(),num);
    return vec;
}
// 数值除赋值
template <typename T>
num_t norm_l2(const std::vector<T>& vec) {
    return calculate::norm_l2(vec.size(),vec.data());
}

template <typename T>
void random(std::vector<T>& vec) {
    std::uniform_real_distribution<T> dis(-1,1);
    do{
        for(auto &i:vec){
            i = dis(gen);
        }
    }while(norm_l2(vec) == 0);
}


template<class T,typename VALUE,typename DIS>
inline auto choose_centroids(const T *begin, const T *end,VALUE value,DIS dis,idx_t K) {
    const idx_t total_size = end - begin;
    std::vector<decltype(value(*begin))> candidates;
    std::vector<decltype(value(*begin))> centroids;
    const idx_t init_center_candidate = std::min(10 * K, total_size);
    auto random_ids = generate_random_indices(init_center_candidate);
    constexpr num_t out_ratio = 0.1;
    std::uniform_real_distribution<num_t> dis_ratio(-out_ratio,1.0 +out_ratio);
    std::uniform_int_distribution<idx_t> dis_id(0,static_cast<idx_t>(random_ids.size())-1);
    constexpr idx_t CK = 3;
    for (idx_t i = 0; i < init_center_candidate; ++i) {
        decltype(value(*begin)) can_buffer[CK];
        num_t ratio[CK];
        for(idx_t j = 0;j<CK;++j) {
            can_buffer[j] = value(begin[random_ids[dis_id(gen)]]);
            ratio[j] = dis_ratio(gen);
        }
        num_t sum = 0;
        for(const auto j:ratio) {
            sum += j;
        }

        for(auto &j:ratio) {
            j /= sum;
        }

        auto new_can = value(begin[0]);
        new_can *= 0;

        for(idx_t j = 0;j<CK;++j) {
            new_can += can_buffer[j] * ratio[j];
        }
        candidates.push_back(new_can);
    }

    for (idx_t i = 0; i < K && !candidates.empty(); ++i) {
        idx_t max_id = 0;
        num_t max_value = -std::numeric_limits<num_t>::max();
        for (idx_t j = i; j < candidates.size(); ++j) {
            num_t min_value = std::numeric_limits<num_t>::max();
            for (idx_t l = 0; l < i; ++l) {
                //choose the farthest one from existing centroids
                min_value = std::min(min_value,dis(centroids[l], candidates[j]));
            }
            if(min_value > max_value){
                max_value = min_value;
                max_id = j;
            }
        }
        centroids.push_back(candidates[max_id]);
        candidates.erase(candidates.begin() + max_id);
    }
    if(centroids.size() > total_size){
        throw StringException("|centroids| > total size");
    }
    return centroids;
}

template<class T,typename VALUE,typename DIS>
inline auto KMeans_clustering(const T *begin, const T *end,
    VALUE value, DIS dis,const  idx_t K, const idx_t max_It = 10) {
    std::vector<idx_t> assignments(end-begin);
    std::vector<idx_t> counts(K);
    std::vector<decltype(value(*begin))> centroids(K);
    std::vector<decltype(value(*begin))> new_centroids(K);
    centroids = choose_centroids<T,VALUE,DIS>(begin,end,value,dis,K);

    bool changed = true;
    idx_t it_count = 0;
    while (changed) {
        changed = false;
        for (auto& c : new_centroids) {
            c = centroids[0];
            c *= 0;
        }
        for(auto &i:counts){
            i = 0;
        }
        // Assign ids to the nearest centroid
        for (idx_t i = 0,total_size = end - begin; i < total_size; ++i) {
            const auto& point = value(begin[i]);
            idx_t best_index = 0;
            num_t best_dist =dis(point, centroids[0]);
            for (idx_t j = 1; j < centroids.size(); ++j) {
                if (const num_t dist = dis(point, centroids[j]); dist < best_dist) {
                    best_dist = dist;
                    best_index = j;
                }
            }

            if (assignments[i] != best_index) {
                changed = true;
                assignments[i] = best_index;
            }
            new_centroids[best_index] += point;
            ++counts[best_index];
        }
        // Update centroids
        for (idx_t i = 0; i < new_centroids.size(); ++i) {
            if (counts[i] > 0) {
                new_centroids[i] /= static_cast<num_t>(counts[i]);
            }else {
                new_centroids.erase(new_centroids.begin() + i);
                counts.erase(counts.begin() + i);
                --i;
            }
        }
        centroids = new_centroids;
        if(++it_count > max_It){
            break;
        }
    }
    std::vector<std::vector<idx_t>> ids(centroids.size());
    for (idx_t i = 0,total_size = end - begin; i < total_size; ++i) {
        const auto& point = value(begin[i]);
        idx_t best_index = 0;
        num_t best_dist =dis(point, centroids[0]);
        for (idx_t j = 1; j < centroids.size(); ++j) {
            if (const num_t dist = dis(point, centroids[j]); dist < best_dist) {
                best_dist = dist;
                best_index = j;
            }
        }
        ids[best_index].push_back(i);
    }
    return std::make_pair(centroids,ids);
}


template<class T, typename VALUE, typename DIS>
inline auto assign(const T *begin,const  T *end, const std::vector<decltype(std::declval<VALUE>()(*begin))> &centroids, VALUE value, DIS dis) {
    std::vector<std::vector<idx_t>> ids(centroids.size());
    for (idx_t i = 0, total_size = end - begin; i < total_size; ++i) {
        const auto& point = value(begin[i]);
        idx_t best_index = 0;
        num_t best_dist = dis(point, centroids[0]);
        for (idx_t j = 1; j < centroids.size(); ++j) {
            if (const num_t dist = dis(point, centroids[j]); dist < best_dist) {
                best_dist = dist;
                best_index = j;
            }
        }
        ids[best_index].push_back(i);
    }
    return ids;
}

template<class T,typename VALUE,typename DIS>
inline void shrink_clusters(const T *begin, const T * end, VALUE value, DIS dis, idx_t K,
                     std::vector<decltype(value(*begin))> &centroids,
                     std::vector<std::vector<idx_t>> &ids) {
    while(centroids.size() > K){
        //find the smallest cluster
        idx_t smallest_id = -1;
        idx_t smallest_size = std::numeric_limits<idx_t >::max();
        for(idx_t i = 0;i<centroids.size();++i){
            if(ids[i].size() < smallest_size){
                smallest_size = static_cast<idx_t>(ids[i].size());
                smallest_id = i;
            }
        }

        auto to_delete_center = centroids[smallest_id];
        auto to_delete_ids = ids[smallest_id];
        centroids.erase(centroids.begin() + smallest_id);
        ids.erase(ids.begin() + smallest_id);
        for(auto id:to_delete_ids) {
            idx_t nearest_id = -1;
            //find the nearest cluster to add
            num_t nearest_dis = std::numeric_limits<num_t >::max();
            for(idx_t i = 0;i<centroids.size();++i){
                if(auto dis_near = dis(centroids[i],value(begin[id]));dis_near < nearest_dis){
                    nearest_dis = dis_near;
                    nearest_id = i;
                }
            }
            ids[nearest_id].push_back(id);
        }
    }
}

template<class T,typename VALUE,typename DIS>
inline void split_clusters(const T *begin, const T * end, VALUE value, DIS dis, idx_t K,
                     std::vector<decltype(value(*begin))> &centroids,
                     std::vector<std::vector<idx_t>> &ids) {
    while(centroids.size() < K){
        //find the smallest cluster
        idx_t largest_id = -1;
        idx_t largest_size = 0;
        for(idx_t i = 0;i<centroids.size();++i){
            if(ids[i].size() > largest_size){
                largest_size = static_cast<idx_t>(ids[i].size());
                largest_id = i;
            }
        }

        auto to_split_center = centroids[largest_id];
        auto to_split_ids = ids[largest_id];
        auto to_split_data = std::vector<T>();
        for(auto i:to_split_ids) {
            to_split_data.push_back(begin[i]);
        }
        centroids.erase(centroids.begin() + largest_id);
        ids.erase(ids.begin() + largest_id);
        std::pair<std::vector<std::vector<num_t> >, std::vector<std::vector<idx_t> > > result_cluster
                    = KMeans_clustering<T, decltype(value), decltype(dis)>(
                        to_split_data.data(), to_split_data.data() + to_split_data.size(), value, dis, 2);
        for(idx_t i = 0;i<result_cluster.first.size();++i) {
            std::vector<idx_t> new_sub_ids;
            for(idx_t j=0;j<result_cluster.second[i].size();++j) {
                new_sub_ids.push_back(to_split_ids[result_cluster.second[i][j]]);
            }
            centroids.push_back(result_cluster.first[i]);
            ids.push_back(new_sub_ids);
        }
    }
}




#endif //WORKLOAD_HPP
