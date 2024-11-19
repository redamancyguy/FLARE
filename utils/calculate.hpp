//
// Created by redamancyguy on 24-8-3.
//

#ifndef CALCULATE_H
#define CALCULATE_H
#include<cmath>
#include "DEFINE.h"

namespace calculate {
    template<class T1, class T2>
    bool equal(const idx_t dim, const T1 *first, const T2 *second) {
        for (idx_t j = 0; j < dim; ++j) {
            if (first[j] != second[j]) {
                return false;
            }
        }
        return true;
    }

    template<class T1, class T2>
    bool gt(const idx_t dim, const T1 *first, const T2 *second) {
        //greater than
        for (idx_t j = 0; j < dim; ++j) {
            if (first[j] <= second[j]) {
                return false;
            }
        }
        return true;
    }

    template<class T1, class T2>
    bool ge(const idx_t dim, const T1 *first, const T2 *second) {
        //greater or equal
        for (idx_t j = 0; j < dim; ++j) {
            if (first[j] < second[j]) {
                return false;
            }
        }
        return true;
    }


    template<class T1, class T2>
    bool lt(const idx_t dim, const T1 *first, const T2 *second) {
        //less than
        for (idx_t j = 0; j < dim; ++j) {
            if (first[j] >= second[j]) {
                return false;
            }
        }
        return true;
    }

    template<class T1, class T2>
    bool le(const idx_t dim, const T1 *first, const T2 *second) {
        //less or euqla
        for (idx_t j = 0; j < dim; ++j) {
            if (first[j] > second[j]) {
                return false;
            }
        }
        return true;
    }


    template<class T>
    T sum(const idx_t dim, T *first) {
        T result = 0;
        for (idx_t j = 0; j < dim; ++j) {
            result += first[j];
        }
        return result;
    }

    template<class T>
    T *neg(const idx_t dim, T *first) {
        //negate;take the negative
        for (idx_t j = 0; j < dim; ++j) {
            first[j] = -first[j];
        }
        return first;
    }

    template<class T1, class T2, class T3>
    void add(const idx_t dim, const T1 *first, const T2 *second, T3 *result) {
        for (idx_t j = 0; j < dim; ++j) {
            result[j] = first[j] + second[j];
        }
    }

    template<class T1, class T2>
    void add_from(const idx_t dim, T1 *first, const T2 *second) {
        for (idx_t j = 0; j < dim; ++j) {
            first[j] += second[j];
        }
    }

    template<class T1, class T2, class T3>
    void sub(const idx_t dim, const T1 *first, const T2 *second, T3 *result) {
        for (idx_t j = 0; j < dim; ++j) {
            result[j] = first[j] - second[j];
        }
    }

    template<class T1, class T2>
    void sub_from(const idx_t dim, T1 *first, const T2 *second) {
        for (idx_t j = 0; j < dim; ++j) {
            first[j] -= second[j];
        }
    }

    template<class T1, class T2>
    void mul(const idx_t dim, const T1 *first, const num_t scalar, T2 *result) {
        for (idx_t j = 0; j < dim; ++j) {
            result[j] = first[j] * scalar;
        }
    }

    template<class T1>
    void mul_by(const idx_t dim, T1 *first, num_t scalar) {
        for (idx_t j = 0; j < dim; ++j) {
            first[j] *= scalar;
        }
    }

    template<class T1, class T2>
    void div(const idx_t dim, const T1 *first, num_t scalar, T2 *result) {
        scalar = 1 / scalar;
        for (idx_t j = 0; j < dim; ++j) {
            result[j] = first[j] * scalar;
        }
    }

    template<class T>
    void div_by(const idx_t dim, T *first, num_t scalar) {
        scalar = 1 / scalar;
        for (idx_t j = 0; j < dim; ++j) {
            first[j] *= scalar;
        }
    }

    template<class T1, class T2>
    num_t dis_l2(const idx_t dim, const T1 *first, const T2 *second) {
#ifdef using_simd
        if constexpr (std::is_same_v<T1, float*> && std::is_same_v<T2, float*>) {
            return simd_calculate::dis_l2(dim,first,second);
        }
#endif
        num_t distance = 0;
        for (idx_t j = 0; j < dim; ++j) {
            const auto temp = first[j] - second[j];
            distance += temp * temp;
        }
        return static_cast<num_t>(std::sqrt(distance));
    }

    template<class T1, class T2, class idx_t>
    num_t dis_l1(const idx_t dim, const T1 *first, const T2 *second) {
        num_t distance = 0;
        for (idx_t j = 0; j < dim; ++j) {
            distance += std::abs(first[j] - second[j]);
        }
        return distance;
    }

    template<class T1, class T2, class idx_t>
    num_t dis_linf(const idx_t dim, const T1 *first, const T2 *second) {
        num_t distance = 0;
        for (idx_t j = 0; j < dim; ++j) {
            const auto temp = std::abs(first[j] - second[j]);
            if (temp > distance) {
                distance = temp;
            }
        }
        return distance;
    }

    template<class T1>
    num_t norm_l2(const idx_t dim, const T1 *first) {
#ifdef using_simd
        if constexpr (std::is_same_v<T1, float*>) {
            return simd_calculate::norm_l2(dim,first);
        }
#endif
        num_t distance = 0;
        for (idx_t j = 0; j < dim; ++j) {
            distance += first[j] * first[j];
        }
        return static_cast<num_t>(std::sqrt(distance));
    }

    template<class T1, class T2>
    num_t dot_product(const idx_t dim, const T1 *first, const T2 *second) {
#ifdef using_simd
        if constexpr (std::is_same_v<T1, float*> && std::is_same_v<T2, float*>) {
            return simd_calculate::dot_product(dim,first,second);
        }
#endif
        num_t result = 0;
        for (idx_t j = 0; j < dim; ++j) {
            result += first[j] * second[j];
        }
        return result;
    }

    template<class T2>
    num_t forward(const idx_t dim, const std::pair<num_t *, num_t> &plane, const T2 *second) {
        return dot_product(dim, plane.first, second) + plane.second;
    }

    template<class T1>
    void random(const idx_t dim, T1 *first, const num_t lower, const num_t upper) {
        // auto dis = std::uniform_real_distribution<num_t>(lower, upper);
        auto dis = std::normal_distribution<num_t>(0, 1);
        for (idx_t j = 0; j < dim; ++j) {
            first[j] = dis(gen);
        }
    }

    template<class T1>
    void normalization(const idx_t dim, T1 *first, const idx_t max_iteration = 1) {
        for (idx_t i = 0; i < max_iteration; ++i) {
            auto norm = norm_l2(dim, first);
            if (norm == 1) {
                break;
            }
            div_by(dim, first, norm);
        }
    }

    template<class T1, class T2>
    num_t cos_value(const idx_t dim, const T1 *first, const T2 *second) {
        auto norm_first = norm_l2(dim, first);
        auto norm_second = norm_l2(dim, second);
        if (norm_first == 0 || norm_second == 0) {
            return 0;
            throw StringException("zero vector !");
        }
        num_t inner_product = dot_product(dim, first, second);
        return inner_product / (norm_first * norm_second);
    }


    template<class T1>
    num_t volume(const idx_t dim, const T1 *first) {
        num_t result = 1;
        for (idx_t j = 0; j < dim; ++j) {
            result *= first[j];
        }
        return result;
    }

    template<class T1>
    num_t dis_to_sphere_l2(const idx_t dim, const std::pair<num_t *, num_t> &sphere, const T1 *first) {
        return dis_l2(dim, sphere.first, first) - sphere.second;
    }

    template<class T, class T2>
    void get_min_max(const idx_t dim, const T **begin, const T **end, T2 *min, T2 *max) {
        for (idx_t i = 0; i < dim; ++i) {
            min[i] = (*begin)[i];
            max[i] = (*begin)[i];
        }
        for (auto it = begin; it < end; ++it) {
            for (idx_t j = 0; j < dim; ++j) {
                min[j] = std::min<T2>(min[j], (*it)[j]);
                max[j] = std::max<T2>(max[j], (*it)[j]);
            }
        }
    }
}


template<class T>
void show(const idx_t dim, T *a) {
    for (idx_t j = 0; j < dim; ++j) {
        std::cout << static_cast<num_t>(a[j]) << " ";
    }
    std::cout << std::endl;
}

inline num_t calculate_ellipsoid_volume(const idx_t dim, const num_t *full_range) {
    num_t product_of_radii = 1.0;
    for (idx_t i = 0; i < dim; ++i) {
        product_of_radii *= full_range[i];
    }
    return std::pow(M_PI, dim / static_cast<num_t>(2.0)) / std::tgamma(dim / static_cast<num_t>(2.0) + 1.0) *
           product_of_radii;
}

inline num_t calculate_equivalent_radius(const idx_t dim, const num_t volume) {
    return std::pow(
        volume * std::tgamma(dim / static_cast<num_t>(2.0) + 1.0) / std::pow(M_PI, dim / static_cast<num_t>(2.0)),
        static_cast<num_t>(1.0) / dim);
}


inline std::vector<idx_t> generate_random_indices(const idx_t init_center_candidate) {
    std::vector<idx_t> indices(init_center_candidate);
    std::iota(indices.begin(), indices.end(), 0);  // Fill with sequential values 0, 1, ..., init_center_candidate-1
    std::random_device rd;
    std::mt19937 g(rd());
    std::ranges::shuffle(indices, g);  // Shuffle to randomize the indices
    return indices;
}

inline std::vector<idx_t> generate_indices(const idx_t init_center_candidate) {
    std::vector<idx_t> indices(init_center_candidate);
    std::iota(indices.begin(), indices.end(), 0);  // Fill with sequential values 0, 1, ..., init_center_candidate-1
    return indices;
}

#endif //CALCULATE_H
