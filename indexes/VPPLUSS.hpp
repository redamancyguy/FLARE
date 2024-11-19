//
// Created by redam on 7/13/2024.
//

#ifndef VPPLUSS_H
#define VPPLUSS_H

#include <queue>
#include <functional>
#include "DEFINE.h"
#include "WorkLoad.hpp"
#include "Index.hpp"

template<class T>
class VPPLUSS final : public Index<T> {
public:
    enum constructType {
        KMeans = 0,
        KMeansShrink = 1,
    };

    static constructType construct_type;
    idx_t dim;
    static idx_t node_count;
    static idx_t leaf_size;
    static idx_t fanout;
    static bool using_angle;
    static idx_t angle_fanout;
    static idx_t angle_k_means_it;
    constexpr static num_t max_dis = 1e10;

private:
    class Node {
    public:
        bool is_inner;
        idx_t size;

        Node(const bool is_inner,
             const idx_t size) : is_inner(is_inner), size(size) {
            ++node_count;
        }

        ~Node() {
            --node_count;
        }
    };


    class InnerNode : public Node {
    public:
        using slot_t = std::pair<std::pair<num_t *, num_t>, std::vector<std::pair<num_t, Node *> > >; //direction //max 1-cos_value //distance ranges
        num_t *center;
        std::vector<slot_t> sons;
        explicit InnerNode(const idx_t dim, const idx_t input_angle_fanout, num_t *input_center)
            : Node(true, input_angle_fanout) {
            center = new num_t[dim];
            std::copy_n(input_center, dim, center);
            for (idx_t i = 0; i < input_angle_fanout; ++i) {
                auto first_part = std::make_pair(new num_t[dim], static_cast<num_t>(0));
                auto second_part = std::vector<std::pair<num_t, Node *>>();
                sons.push_back(std::make_pair(first_part, second_part));
            }
        }

        ~InnerNode() {
            for (idx_t i = 0; i < sons.size(); ++i) {
                delete[] sons[i].first.first;
            }
            delete[] center;
        }

    public:

        auto cos_value(const idx_t dim, const T *data, const idx_t angle_id) const {
            T dr_o[dim];
            calculate::sub(dim, data, center, dr_o);
            auto value = calculate::cos_value(dim, dr_o, sons[angle_id].first.first);
            return value;
        }

        auto distance(const idx_t dim, const T *data) const {
            return calculate::dis_l2(dim, center, data);
        }

        static idx_t find_slot_id(InnerNode *inner_node, const idx_t angle_id, const num_t value) {
            auto left_id = static_cast<idx_t>(0);
            auto right_id = static_cast<idx_t>(inner_node->sons[angle_id].second.size() - 2);
            auto mid_id = (left_id + right_id) / 2;
            while (mid_id != left_id && mid_id != right_id) {
                if (inner_node->sons[angle_id].second[mid_id].first < value) {
                    left_id = mid_id + 1;
                } else if (inner_node->sons[angle_id].second[mid_id].first > value) {
                    right_id = mid_id - 1;
                } else {
                    break;
                }
                mid_id = (left_id + right_id) / 2;
            }
            while (value < inner_node->sons[angle_id].second[mid_id].first && mid_id > 0) {
                --mid_id;
            }
            while (value >= inner_node->sons[angle_id].second[mid_id].first && mid_id < static_cast<idx_t>(inner_node->sons[angle_id].second.size()) - 1) {
                ++mid_id;
            }
            return mid_id;
        }
    };

    class LeafNode : public Node {
    public:
        using slot_t = idx_t;
        num_t *center;
        explicit LeafNode(const idx_t dim, const idx_t size) : Node(false, size) {
            ids = new slot_t[size];
            center = new num_t[dim];
        }

        ~LeafNode() {
            delete[]center;
            delete[]ids;
        }

        slot_t *ids;
    };

private:
    Node *root;

public:
    ~VPPLUSS() {
        top_down_destroy(root);
    }

    explicit VPPLUSS(WorkLoad<T> &workload): Index<T>(workload), dim(workload.dim) {
        std::vector<idx_t> ids(workload.size);
        for (idx_t i = 0; i < ids.size(); ++i) {
            ids[i] = i;
        }

        root = top_down_construct(ids.data(), ids.data() + ids.size(), workload);
    }

    void top_down_destroy(Node *node) {
        if (node->is_inner) {
            auto inner_node = static_cast<InnerNode *>(node);
            for (idx_t i = 0; i < inner_node->size; ++i) {
                for (idx_t j = 0; j < inner_node->sons[i].second.size(); ++j) {
                    top_down_destroy(inner_node->sons[i].second[j].second);
                }
            }
            delete inner_node;
        } else {
            auto leaf_node = static_cast<LeafNode *>(node);
            delete leaf_node;
        }
    }
// #define print_counstruction

   Node *top_down_construct(idx_t *begin, idx_t *end, const WorkLoad<T> &workload, const idx_t layer = 0) {
#ifdef print_counstruction
        std::cout << end - begin << " "<<std::endl;
        // static int ccc = 0;
        // if (ccc++ % 40 == 0) {
        //     std::cout << std::endl;
        // }
#endif
        if (end - begin <= leaf_size || workload.check_same(begin, end)) {
        LEAF:
            auto node = new LeafNode(dim, end - begin);
            for(idx_t i = 0;i<workload.dim;++i) {
                node->center[i] = 0;
            }
            for (idx_t i = 0; i < node->size; ++i) {
                node->ids[i] = begin[i];
                calculate::add_from(dim, node->center, workload[begin[i]]);
            }
            calculate::div_by(dim,node->center, node->size);
            return node;
        }
        const auto expected_fanout = std::min<idx_t>(
            angle_fanout, std::ceil(static_cast<num_t>(end - begin) / leaf_size));
        int try_clustering_time = 0;
        CLUSTER:
        num_t ratio = std::uniform_real_distribution<num_t>(0.05, 0.95)(gen);
        std::uniform_int_distribution<idx_t> dist(0, end - begin - 1);
        idx_t first_id = dist(gen);
        idx_t second_id = dist(gen);
        std::vector<num_t> first_center(dim);
        std::vector<num_t> second_center(dim);
        calculate::mul(dim, workload[first_id], ratio, first_center.data());
        calculate::mul(dim, workload[second_id], 1 - ratio, second_center.data());
        std::vector<num_t> center = first_center + second_center;
        auto dis_value_f = [&](const idx_t a) -> num_t {
            return calculate::dis_l2(dim, center.data(), workload[a]);
        };
        //this code is using for clustering the data by angle
        auto get_value = [&](const idx_t first) {
            //Get dr_o
            auto result = std::vector<num_t>(dim);
            std::copy_n(workload[first], dim, result.data());
            result -= center;
            const num_t norm = norm_l2(result);
            result /= norm;
            return result;
        };
        auto angle_dis_f = [&](const std::vector<num_t> &first, const std::vector<num_t> &second) {
            //Get cos similarity
            return 1 - calculate::cos_value(dim, first.data(), second.data());
        };
        #ifdef print_counstruction
        puts("clustering");
#endif
        std::pair<std::vector<std::vector<num_t> >, std::vector<std::vector<idx_t> > > result_cluster
                = KMeans_clustering<idx_t, decltype(get_value), decltype(angle_dis_f)>(begin, end, get_value, angle_dis_f, expected_fanout, angle_k_means_it);
        if (construct_type == constructType::KMeansShrink) {
            shrink_clusters(begin, end, get_value, angle_dis_f, static_cast<num_t>(expected_fanout) / 1.5, result_cluster.first, result_cluster.second);
        }
#ifdef print_counstruction
        std::cout <<result_cluster.first.size()<<":"<<result_cluster.second.size()<<std::endl;
        for(auto ii:result_cluster.second) {
            std::cout <<ii.size()<< " ";
        }
        std::cout << std::endl;
#endif
        bool has_zero = false;
        for (auto &i: result_cluster.first) {
            if (norm_l2(i) == 0) {
                random(i);
                has_zero = true;
            }
        }
        if (has_zero) {
            result_cluster.second = assign<idx_t, decltype(get_value), decltype(angle_dis_f)>(begin, end, result_cluster.first, get_value, angle_dis_f);
        }
        if (result_cluster.first.size() == 1 ) {
            if(try_clustering_time > 2) {
                goto LEAF;
            }else {
                ++try_clustering_time;
                goto CLUSTER;
            }
        }
        InnerNode *node = new InnerNode(dim, result_cluster.first.size(), center.data());
#ifdef print_counstruction
        puts("spliting");
#endif
        for (idx_t i = 0; i < result_cluster.first.size(); ++i) {
            if (calculate::norm_l2(dim, result_cluster.first[i].data()) == 0) {
                throw StringException("vector norm is zero !");
            }
            std::vector<idx_t> sub_set;
            std::copy_n(result_cluster.first[i].data(), dim, node->sons[i].first.first);
#ifdef print_counstruction
            for(auto &ii:result_cluster.first[i]) {
                if(std::isnan(ii) || std::isinf(ii)) {
                    std::cout <<int(std::isnan(ii))<<int(std::isinf(ii))<<"angle direction"<< std::endl;
                }
            }
#endif
            node->sons[i].first.second = -1;
            for (idx_t ii = 0; ii < result_cluster.second[i].size(); ++ii) {
                sub_set.push_back(begin[result_cluster.second[i][ii]]);
                std::vector<num_t> dr_o(dim);
                calculate::sub(dim, workload[begin[result_cluster.second[i][ii]]], center.data(), dr_o.data());
                node->sons[i].first.second = std::max(node->sons[i].first.second, angle_dis_f(result_cluster.first[i], dr_o));
            }
            auto expected_fanout_dis = std::min<idx_t>(
                fanout, std::ceil(static_cast<num_t>(sub_set.size()) / leaf_size));
            expected_fanout_dis = std::max<idx_t>(expected_fanout_dis, 1);
            const auto result = sort_and_split(sub_set.data(), sub_set.data() + sub_set.size(), expected_fanout_dis, dis_value_f);
            if (result_cluster.first.size() <= 1 && result.size() <= 1) {
                puts("did not split");
            }
            idx_t last_end_id = 0;
            // new(&node->sons[i].second)std::vector<std::pair<num_t, Node *> >();
            node->sons[i].second.reserve(result.size());
            for (idx_t j = 0; j < result.size(); ++j) {
                auto this_end_id = result[j];
                // if (this_end_id - last_end_id == end - begin) {
                //     puts("there only once split in this angle !");
                // }
                auto son = top_down_construct(
                    sub_set.data() + last_end_id, sub_set.data() + this_end_id, workload, layer + 1);

                if (j < static_cast<idx_t>(result.size()) - 1) {
                    node->sons[i].second.push_back(std::make_pair(dis_value_f(sub_set[this_end_id]), son));
                } else {
                    node->sons[i].second.push_back(std::make_pair(sub_set.empty() ? max_dis : dis_value_f(sub_set.back()), son));
                }
                last_end_id = this_end_id;
            }
        }

        return node;
    }

    void reorganize(WorkLoad<T> &workload) {
        auto leaves = get_all_leaves();
        idx_t new_id_count = 0;
        auto new_data = workload.dataset.clone().detach();
        for (idx_t i = 0; i < leaves.size(); ++i) {
            for(idx_t j = 0;j<leaves[i]->size;++j) {
                idx_t old_id = leaves[i]->ids[j];
                idx_t new_id = new_id_count++;
                leaves[i]->ids[j] = new_id;
                new_data[new_id] = workload.dataset[old_id];
            }
        }
        workload.dataset = std::move(new_data);
        workload.update_info();
    }

    void reorganize_tow_workload(WorkLoad<T> &workload, WorkLoad<T> &workload2) {
        auto leaves = get_all_leaves();
        idx_t new_id_count = 0;
        auto new_data = workload.dataset.clone().detach();
        auto new_data2 = workload2.dataset.clone().detach();
        for (idx_t i = 0; i < leaves.size(); ++i) {
            for(idx_t j = 0;j<leaves[i]->size;++j) {
                idx_t old_id = leaves[i]->ids[j];
                idx_t new_id = new_id_count++;
                leaves[i]->ids[j] = new_id;
                new_data[new_id] = workload.dataset[old_id];
                new_data2[new_id] = workload2.dataset[old_id];
            }
        }
        workload.dataset = std::move(new_data);
        workload.update_info();
        workload2.dataset = std::move(new_data2);
        workload2.update_info();
    }

    [[nodiscard]] std::vector<idx_t> get_all_data() const {
        std::vector<idx_t> result;
        std::queue<Node *> routes;
        routes.push(root);
        while (!routes.empty()) {
            auto node = routes.front();
            routes.pop();
            if (node->is_inner) {
                auto inner_node = static_cast<InnerNode *>(node);
                for (idx_t i = 0; i < inner_node->sons.size(); ++i) {
                    for (idx_t j = 0; j < inner_node->sons[i].second.size(); ++j) {
                        routes.push(inner_node->sons[i].second[j].second);
                    }
                }
            } else {
                auto leaf_node = static_cast<LeafNode *>(node);
                for (idx_t i = 0; i < leaf_node->size; ++i) {
                    result.push_back(leaf_node->ids[i]);
                }
            }
        }
        return result;
    }
    [[nodiscard]] std::vector<LeafNode*> get_all_leaves() const {
        std::vector<LeafNode*> result;
        std::queue<Node *> routes;
        routes.push(root);
        while (!routes.empty()) {
            auto node = routes.front();
            routes.pop();
            if (node->is_inner) {
                auto inner_node = static_cast<InnerNode *>(node);
                for (idx_t i = 0; i < inner_node->sons.size(); ++i) {
                    for (idx_t j = 0; j < inner_node->sons[i].second.size(); ++j) {
                        routes.push(inner_node->sons[i].second[j].second);
                    }
                }
            } else {
                auto leaf_node = static_cast<LeafNode *>(node);
                result.push_back(leaf_node);
            }
        }
        return result;
    }


    void point(const num_t *point, std::vector<idx_t> &result, const WorkLoad<T> &workload) const {
        auto *node = root;
        while (node->is_inner) {
            auto inner_node = static_cast<InnerNode *>(node);
            idx_t min_id = -1;
            num_t min_value = 3;
            for (idx_t i = 0; i < inner_node->size; ++i) {
#ifdef COUNT_SCAN_NUM
                ++scan_node_count;
#endif
                num_t sub_value[dim];
                calculate::sub(dim, point, inner_node->center, sub_value);
                if (auto value = 1 - calculate::cos_value(dim, sub_value, inner_node->sons[i].first.first); value < min_value) {
                    min_value = value;
                    min_id = i;
                }
                //                if (min_value <= inner_node->sons[min_id].first.second) {
                //                    auto dis = calculate::dis_l2(dim,point, inner_node->center);
                //                    auto next_slot_id = InnerNode::find_slot_id(inner_node, min_id, dis);
                //                    node = inner_node->sons[min_id].second[next_slot_id].second;
                //                }
            }
            auto dis = calculate::dis_l2(dim, point, inner_node->center);
            auto next_slot_id = InnerNode::find_slot_id(inner_node, min_id, dis);
            node = inner_node->sons[min_id].second[next_slot_id].second;
        }
        auto *leaf_node = static_cast<LeafNode *>(node);
        for (idx_t i = 0; i < leaf_node->size; ++i) {
#ifdef COUNT_SCAN_NUM
            ++scan_count;
#endif
            if (calculate::equal(dim, workload[leaf_node->ids[i]], point)) {
                result.push_back(leaf_node->ids[i]);
            }
        }
    }

    void similar_range(const num_t *center,
                       const num_t radius,
                       std::vector<idx_t> &result,
                       const WorkLoad<T> &workload) const {
        std::queue<Node *> routes;
        routes.push(root);
        while (!routes.empty()) {
#ifdef COUNT_SCAN_NUM
            ++scan_node_count;
#endif
            auto node = routes.front();
            routes.pop();
            if (node->is_inner) {
                auto *inner_node = static_cast<InnerNode *>(node);
                auto dis = calculate::dis_l2(dim, center, inner_node->center);
                for (idx_t i = 0; i < inner_node->size; ++i) {
                    num_t sub_value[dim];
                    calculate::sub(dim, center, inner_node->center, sub_value);
                    auto cos_theta_q = calculate::cos_value(dim, sub_value, inner_node->sons[i].first.first);
                    auto cos_theta = 1 - inner_node->sons[i].first.second;
                    //theta_q > theta
                    if (cos_theta_q < cos_theta) {
                        //cos(theta_q - theta)
                        auto cos_theta_q_theta = cos_theta_q * cos_theta + std::sqrt(1 - cos_theta_q * cos_theta_q) * std::sqrt(1 - cos_theta * cos_theta);
                        for (idx_t j = 0; j < inner_node->sons[i].second.size(); ++j) {
                            num_t r_min = j > 0 ? inner_node->sons[i].second[j - 1].first : 0;
                            num_t r_max = j < static_cast<idx_t>(inner_node->sons[i].second.size()) - 1 ? inner_node->sons[i].second[j].first : std::numeric_limits<num_t>::max();
                            // num_t r_max = inner_node->sons[i].second[j].first;
                            auto real_dis = (dis - r_min) * (dis - r_max) > 0 ? std::min(std::abs(dis - r_min), std::abs(dis - r_max)) : static_cast<num_t>(0);
                            if (using_angle) {
                                if (j < static_cast<idx_t>(inner_node->sons[i].second.size()) - 1) {
                                    if (dis * cos_theta_q_theta > r_max) {
                                        real_dis = std::sqrt(dis * dis + r_max * r_max - 2 * dis * r_max * cos_theta_q_theta);
                                    } else if (dis * cos_theta_q_theta < r_min) {
                                        real_dis = std::sqrt(dis * dis + r_min * r_min - 2 * dis * r_min * cos_theta_q_theta);
                                    } else {
                                        auto sin_theta_q_theta = std::sqrt(1 - cos_theta_q_theta * cos_theta_q_theta);
                                        real_dis = dis * sin_theta_q_theta;
                                    }
                                }
                            }
                            if (real_dis < radius) {
                                routes.push(inner_node->sons[i].second[j].second);
                            }
                        }
                    } else {
                        for (idx_t j = 0; j < inner_node->sons[i].second.size(); ++j) {
                            num_t r_min = dis - (j > 0 ? inner_node->sons[i].second[j - 1].first : 0);
                            num_t r_max = dis - (j < static_cast<idx_t>(inner_node->sons[i].second.size()) - 1 ? inner_node->sons[i].second[j].first : std::numeric_limits<num_t>::max());
                            auto real_dis = r_min * r_max > 0 ? std::min(std::abs(r_min), std::abs(r_max)) : static_cast<num_t>(0);
                            if (real_dis < radius) {
                                routes.push(inner_node->sons[i].second[j].second);
                            }
                        }
                    }
                }
            } else {
                auto leaf_node = static_cast<LeafNode *>(node);
                for (idx_t i = 0; i < leaf_node->size; ++i) {
#ifdef COUNT_SCAN_NUM
                    ++scan_count;
#endif
                    if (calculate::dis_l2(dim, center, workload[leaf_node->ids[i]]) <= radius) {
                        result.push_back(leaf_node->ids[i]);
                    }
                }
            }
        }
    }

    //    void kNN_one_stage(const Point<T, DIM> &center, const idx_t k,
    //             std::vector<std::pair<Point<T, DIM>, pld_t> > &result) const {
    //        auto cmp_node = [&](const std::pair<num_t, Node *> &first, const std::pair<num_t, Node *> &second) {
    //            return first.first > second.first;
    //        };
    //        std::priority_queue<std::pair<num_t, Node *>, std::vector<std::pair<num_t, Node *> >, decltype(cmp_node)> nodePQ(cmp_node);
    //
    //        auto cmp_data = [&](const std::pair<Point<T, DIM>, pld_t> &first,
    //                            const std::pair<Point<T, DIM>, pld_t> &second) {
    //            return Point<T, DIM>::distance(center, first.first) < Point<T, DIM>::distance(center, second.first);
    //        };
    //        std::priority_queue<std::pair<Point<T, DIM>, pld_t>, std::vector<std::pair<Point<T, DIM>, pld_t> >, decltype(cmp_data)> dataPQ(cmp_data);
    //
    //        nodePQ.push(std::make_pair(static_cast<num_t>(0), root));
    //        while (!nodePQ.empty()) {
    //#ifdef COUNT_SCAN_NUM
    //            ++scan_node_count;
    //#endif
    //            Node *node = nodePQ.top().second;
    //            nodePQ.pop();
    //            if (node->is_inner) {
    //                auto *inner_node = static_cast<InnerNode *>(node);
    //                auto dis = Point<T, DIM>::distance(center, inner_node->center);
    //                for (idx_t i = 0; i < inner_node->size; ++i) {
    //                    auto cos_theta_q = Point<T, DIM>::cosine_value(center - inner_node->center, inner_node->sons[i].first.first);
    //                    auto cos_theta = 1 - inner_node->sons[i].first.second;
    //                    //theta_q > theta
    //                    if (cos_theta_q < cos_theta) {
    //                        //cos(theta_q - theta)
    //                        auto cos_theta_q_theta = cos_theta_q * cos_theta + std::sqrt(1 - cos_theta_q * cos_theta_q) * std::sqrt(1 - cos_theta * cos_theta);
    //
    //                        for (idx_t j = 0; j < inner_node->sons[i].second.size(); ++j) {
    //                            num_t r_min = j > 0 ? inner_node->sons[i].second[j - 1].first : 0;
    //                            num_t r_max = j < inner_node->sons[i].second.size() - 1 ? inner_node->sons[i].second[j].first : std::numeric_limits<num_t>::max();
    //                            // num_t r_max = inner_node->sons[i].second[j].first;
    //                            auto real_dis = (dis - r_min) * (dis - r_max) > 0 ? std::min(std::abs(dis - r_min), std::abs(dis - r_max)) : static_cast<num_t>(0);
    //
    //                            if (using_angle) {
    //                                if (j < inner_node->sons[i].second.size() - 1) {
    //                                    if (dis * cos_theta_q_theta > r_max) {
    //                                        real_dis = std::sqrt(dis * dis + r_max * r_max - 2 * dis * r_max * cos_theta_q_theta);
    //                                    } else if (dis * cos_theta_q_theta < r_min) {
    //                                        real_dis = std::sqrt(dis * dis + r_min * r_min - 2 * dis * r_min * cos_theta_q_theta);
    //                                    } else {
    //                                        auto sin_theta_q_theta = std::sqrt(1 - cos_theta_q_theta * cos_theta_q_theta);
    //                                        real_dis = dis * sin_theta_q_theta;
    //                                    }
    //                                } else {
    //                                    if (dis * cos_theta_q_theta < r_min) {
    //                                        real_dis = std::sqrt(dis * dis + r_min * r_min - 2 * dis * r_min * cos_theta_q_theta);
    //                                    }
    //                                }
    //                            }
    //                            nodePQ.push(std::make_pair(real_dis, inner_node->sons[i].second[j].second));
    //                        }
    //                    } else {
    //                        for (idx_t j = 0; j < inner_node->sons[i].second.size(); ++j) {
    //                            num_t r_min = dis - (j > 0 ? inner_node->sons[i].second[j - 1].first : 0);
    //                            num_t r_max = dis - (j < inner_node->sons[i].second.size() - 1 ? inner_node->sons[i].second[j].first : std::numeric_limits<num_t>::max());
    //                            auto real_dis = r_min * r_max > 0 ? std::min(std::abs(r_min), std::abs(r_max)) : static_cast<num_t>(0);
    //                            nodePQ.push(std::make_pair(real_dis, inner_node->sons[i].second[j].second));
    //                        }
    //                    }
    //                }
    //            } else {
    //                auto *leaf_node = static_cast<LeafNode *>(node);
    //                for (idx_t i = 0; i < leaf_node->size; ++i) {
    //#ifdef COUNT_SCAN_NUM
    //                    ++scan_count;
    //#endif
    //                    dataPQ.push(leaf_node->data[i]);
    //                    if (dataPQ.size() > k) {
    //                        dataPQ.pop();
    //                    }
    //                }
    //            }
    //            if (dataPQ.size() >= k && !nodePQ.empty() &&
    //                nodePQ.top().first > Point<T, DIM>::distance(dataPQ.top().first, center)) {
    //                break;
    //            }
    //        }
    //        while (!dataPQ.empty()) {
    //            result.push_back(dataPQ.top());
    //            dataPQ.pop();
    //        }
    //    }

    static num_t min_dis_to_sector(num_t dis, const num_t cos_theta_q, const num_t cos_theta, num_t r_min, num_t r_max) {
        if (cos_theta_q < cos_theta) {
            //out of sector
            auto cos_theta_q_theta = cos_theta_q * cos_theta + std::sqrt(1 - cos_theta_q * cos_theta_q) * std::sqrt(
                                         1 - cos_theta * cos_theta);
            if (dis * cos_theta_q_theta > r_max) { return std::sqrt(dis * dis + r_max * r_max - 2 * dis * r_max * cos_theta_q_theta); }
            if (dis * cos_theta_q_theta < r_min) { return std::sqrt(dis * dis + r_min * r_min - 2 * dis * r_min * cos_theta_q_theta); }
            auto sin_theta_q_theta = std::sqrt(1 - cos_theta_q_theta * cos_theta_q_theta);
            return dis * sin_theta_q_theta;
        }
        // const auto dis1 = dis - r_max;
        // const auto dis2= r_min - dis;
        // return std::max(dis1,dis2);
        if (dis > r_max) { return dis - r_max; }
        if (dis < r_min) { return r_min - dis; }
        return 0;
    }

    static num_t max_dis_to_sector(const num_t *query, const num_t *pivot, const num_t *v, const num_t dis, const num_t cos_theta_q, const num_t cos_theta, const num_t r_min, const num_t r_max, const idx_t dim) {
        //theta_q < pai-theta
        if (cos_theta_q > -cos_theta) {
            const num_t dis_p_p_ = 1.0/cos_theta * (r_min + r_max) / 2;

            num_t p_[dim];//得到p'的坐标，然后用于判断
            calculate::mul(dim,v,dis_p_p_,p_);//line 5
            calculate::add(dim,pivot,p_,p_);//line 5
            calculate::sub(dim,query,p_,p_);//line 6
            //////////////
            num_t cos_theta_qc = calculate::cos_value(dim,p_,v);//line 6
            num_t sin_theta = std::sqrt(1-cos_theta*cos_theta);
            num_t cos_theta_q_plus_theta = cos_theta_q * cos_theta - std::sqrt(1-cos_theta_q*cos_theta_q) * sin_theta;
            //if theta_q_c > pai/2 - theta //cos_qc < cos(pai/2 - theta)
            if(cos_theta_qc < sin_theta) {
                return std::sqrt(dis * dis + r_max * r_max - 2 * dis * r_max * cos_theta_q_plus_theta);
            }else {
                return std::sqrt(dis * dis + r_min * r_min - 2 * dis * r_min * cos_theta_q_plus_theta);
            }
        }
        return r_max + dis;
    }
public:

    //kNN_two_stage
    void kNN(const num_t *center, const idx_t k,
             std::vector<idx_t> &result, const WorkLoad<T> &workload) const  {
        AkNN(center,k,1,result,workload);
    }
    void AkNN(const num_t *center, const idx_t k,const num_t c,
             std::vector<idx_t> &result, const WorkLoad<T> &workload) const  {
        //kNN_two_stage
        class kNNElement {
        public:
            Node *node;
            idx_t angle_id = -1;
            num_t dis = -1;
            num_t cos_theta_q = -1;
            bool is_node() const {
                return angle_id == -1;
            }
        };
        auto cmp_node = [&](const std::pair<num_t, kNNElement> &first, const std::pair<num_t, kNNElement> &second) { return first.first > second.first; };
        std::priority_queue<std::pair<num_t, kNNElement>, std::vector<std::pair<num_t, kNNElement> >, decltype(cmp_node)> nodePQ(cmp_node);
        idx_t candidate_size = 0;
        auto cmp_data = [&](const std::pair<num_t, LeafNode*> &first, const std::pair<num_t, LeafNode*> &second) { return first.first < second.first; };
        std::priority_queue<std::pair<num_t, LeafNode*>, std::vector<std::pair<num_t, LeafNode*> >, decltype(cmp_data)> dataPQ(cmp_data);
        {
            kNNElement root_element;
            root_element.node = root; //content
            nodePQ.push(std::make_pair(static_cast<num_t>(0), root_element));
        }
        while (!nodePQ.empty()) {
            auto father_dis = nodePQ.top().first;
            kNNElement element = nodePQ.top().second;
            nodePQ.pop();
            if (element.is_node()) {
#ifdef COUNT_SCAN_NUM
                ++scan_node_count;
#endif
                auto *node = element.node;
                if (node->is_inner) {
                    auto *inner_node = static_cast<InnerNode *>(node);
                    auto dis = calculate::dis_l2(dim, center, inner_node->center);
                    for (idx_t i = 0; i < inner_node->size; ++i) {
                        num_t sub_value[dim];
                        calculate::sub(dim, center, inner_node->center, sub_value);
                        auto cos_theta_q = calculate::cos_value(dim, sub_value, inner_node->sons[i].first.first);
                        auto cos_theta = 1 - inner_node->sons[i].first.second;
                        num_t angle_based_dis = min_dis_to_sector(dis, cos_theta_q, cos_theta, 0, std::numeric_limits<num_t>::max());
                        angle_based_dis = std::max(father_dis, angle_based_dis);
                        kNNElement new_element;
                        new_element.dis = dis;
                        new_element.node = inner_node;
                        new_element.angle_id = i;
                        new_element.cos_theta_q = cos_theta_q;
                        nodePQ.push(std::make_pair(angle_based_dis, new_element));
                    }
                } else {
                    auto *leaf_node = static_cast<LeafNode *>(node);
                    dataPQ.push(std::make_pair(calculate::dis_l2(dim, leaf_node->center, center), leaf_node));
                    candidate_size += leaf_node->size;
                    if (candidate_size - dataPQ.top().second->size > k) {
                        candidate_size -= dataPQ.top().second->size;
                        dataPQ.pop();
                    }
                }
            } else {
                auto *inner_node = static_cast<InnerNode *>(element.node);
                auto i = element.angle_id;
                auto cos_theta_q = element.cos_theta_q;
                auto cos_theta = 1 - inner_node->sons[i].first.second;
                auto dis = element.dis;
                for (idx_t j = 0; j < inner_node->sons[i].second.size(); ++j) {
                    num_t r_min = j > 0 ? inner_node->sons[i].second[j - 1].first : 0;
                    num_t r_max = j < static_cast<idx_t>(inner_node->sons[i].second.size()) - 1 ? inner_node->sons[i].second[j].first : std::numeric_limits<num_t>::max();
                    auto real_dis = min_dis_to_sector(dis, cos_theta_q, cos_theta, r_min, r_max);
                    real_dis = std::max(father_dis, real_dis);
                    kNNElement new_element;
                    new_element.node = inner_node->sons[i].second[j].second;
                    nodePQ.push(std::make_pair(real_dis, new_element));
                }
            }
            if (candidate_size >= k && !nodePQ.empty() && c * nodePQ.top().first >= dataPQ.top().first) { break; }
        }
        while (!dataPQ.empty()) {
            auto *leaf = dataPQ.top().second;
            for(idx_t i = 0; i < leaf->size; ++i) {
                result.push_back(leaf->ids[i]);
            }
            dataPQ.pop();
        }
    }


    void kNN2(const num_t *center, const idx_t k,
             std::vector<idx_t> &result, const WorkLoad<T> &workload) const {
        enum class EleType : char { Node = 0, Angle = 1 };
        using kNNElement = std::pair<EleType, std::pair<Node *, idx_t> >;
        auto cmp_node = [&](const std::pair<num_t, kNNElement> &first, const std::pair<num_t, kNNElement> &second) { return first.first > second.first; };
        std::priority_queue<std::pair<num_t, kNNElement>, std::vector<std::pair<num_t, kNNElement> >, decltype(cmp_node)> nodePQ(cmp_node);

        auto cmp_data = [&](const std::pair<num_t, idx_t> &first, const std::pair<num_t, idx_t> &second) { return first.first < second.first; };
        std::priority_queue<std::pair<num_t, idx_t>, std::vector<std::pair<num_t, idx_t> >, decltype(cmp_data)> dataPQ(cmp_data);

        kNNElement root_element;
        root_element.first = EleType::Node; //type
        root_element.second.first = root; //content
        nodePQ.push(std::make_pair(static_cast<num_t>(0), root_element));
        while (!nodePQ.empty()) {
            auto father_dis = nodePQ.top().first;
            if (father_dis < 0) { throw StringException("distance can not be lower than zero !"); }
            kNNElement element = nodePQ.top().second;
            nodePQ.pop();
            if (element.first == EleType::Node) {
#ifdef COUNT_SCAN_NUM
                ++scan_node_count;
#endif
                auto *node = element.second.first;
                if (node->is_inner) {
                    auto *inner_node = static_cast<InnerNode *>(node);
                    auto dis = calculate::dis_l2(dim, center, inner_node->center);
                    for (idx_t i = 0; i < inner_node->size; ++i) {
                        num_t sub_value[dim];
                        calculate::sub(dim, center, inner_node->center, sub_value);
                        auto cos_theta_q = calculate::cos_value(dim, sub_value, inner_node->sons[i].first.first);
                        auto cos_theta = 1 - inner_node->sons[i].first.second;
                        num_t angle_based_dis = min_dis_to_sector(dis, cos_theta_q, cos_theta, 0, std::numeric_limits<num_t>::max());
                        angle_based_dis = std::max(father_dis, angle_based_dis);
                        kNNElement new_element;
                        new_element.first = EleType::Angle; //type is angle
                        new_element.second.first = inner_node;
                        new_element.second.second = i;
                        nodePQ.push(std::make_pair(angle_based_dis, new_element));
                    }
                } else {
                    auto *leaf_node = static_cast<LeafNode *>(node);
                    for (idx_t i = 0; i < leaf_node->size; ++i) {
#ifdef COUNT_SCAN_NUM
                        ++scan_count;
#endif
                        dataPQ.push(std::make_pair(calculate::dis_l2(dim, workload[leaf_node->ids[i]], center), leaf_node->ids[i]));
                        if (dataPQ.size() > k) {
                            dataPQ.pop();
                        }
                    }
                }
            } else if (element.first == EleType::Angle) {
                auto *inner_node = static_cast<InnerNode *>(element.second.first);
                auto i = element.second.second;
                num_t sub_value[dim];
                calculate::sub(dim, center, inner_node->center, sub_value);
                auto cos_theta_q = calculate::cos_value(dim, sub_value, inner_node->sons[i].first.first);
                auto cos_theta = 1 - inner_node->sons[i].first.second;
                auto dis = calculate::dis_l2(dim, center, inner_node->center);
                for (idx_t j = 0; j < inner_node->sons[i].second.size(); ++j) {
                    num_t r_min = j > 0 ? inner_node->sons[i].second[j - 1].first : 0;
                    num_t r_max = j < static_cast<idx_t>(inner_node->sons[i].second.size()) - 1 ? inner_node->sons[i].second[j].first : std::numeric_limits<num_t>::max();
                    auto real_dis = min_dis_to_sector(dis, cos_theta_q, cos_theta, r_min, r_max);
                    real_dis = std::max(father_dis, real_dis);
                    kNNElement new_element;
                    new_element.first = EleType::Node; //type is node
                    new_element.second.first = inner_node->sons[i].second[j].second;
                    nodePQ.push(std::make_pair(real_dis, new_element));
                }
            } else {
                throw StringException("bad element type");
            }
            if (dataPQ.size() >= k && !nodePQ.empty() && nodePQ.top().first > dataPQ.top().first) { break; }
        }
        while (!dataPQ.empty()) {
            result.push_back(dataPQ.top().second);
            dataPQ.pop();
        }
    }
};

template<class T>
idx_t VPPLUSS<T>::node_count = 0;
template<class T>
idx_t VPPLUSS<T>::leaf_size = 25 * DEFAULT_LEAF_SIZE;
template<class T>
idx_t VPPLUSS<T>::fanout = DEFAULT_FANOUT;
template<class T>
idx_t VPPLUSS<T>::angle_fanout = 1.5 * DEFAULT_FANOUT;
template<class T>
bool VPPLUSS<T>::using_angle = true;
template<class T>
idx_t VPPLUSS<T>::angle_k_means_it = 2;
template<class T>
typename VPPLUSS<T>::constructType VPPLUSS<T>::construct_type = constructType::KMeans;

#endif //VPPLUSS_H
