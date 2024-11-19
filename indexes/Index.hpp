//
// Created by redam on 8/8/2024.
//

#ifndef HDLI_INDEX_H
#define HDLI_INDEX_H
#include <vector>
#include "../utils/WorkLoad.hpp"
template<typename T>
class Index {
public:
    explicit Index(const WorkLoad<T> &work_load) {}
    virtual ~Index() {}

    virtual void point(const num_t * point, std::vector<idx_t >& result, const WorkLoad<T>& workload) const {
        workload.point(point,result);
    };
    virtual void range(const num_t * min, const num_t * max, std::vector<idx_t >& result, const WorkLoad<T>& workload) const {
        workload.range(min,max,result);
    };
    virtual void similar_range(const num_t* center, const num_t radius, std::vector<idx_t >&result , const WorkLoad<T>&workload ) const {
        workload.similar_range(center,radius,result);
    };
    virtual void kNN(const num_t* center, idx_t k, std::vector<idx_t >& result, const WorkLoad<T>& workload) const {
        workload.kNN(center,k,result);
    };
    virtual void AkNN(const num_t* center, const idx_t k, const num_t c, std::vector<idx_t >& result, const WorkLoad<T>& workload) const {
        workload.kNN(center,k,result);
    };
};


enum IndexType {
    LS,
    RP,
    VPP_OL,
    VPP,
    VP,
    MVP,
    M,
    KD
};

IndexType index_type = LS;
IndexType stringToIndexType(const std::string& str) {
    if (str == "LS") return LS;
    if (str == "RP") return RP;
    if (str == "VPP-OL") return VPP_OL;
    if (str == "VPP") return VPP;
    if (str == "VP") return VP;
    if (str == "MVP") return MVP;
    if (str == "M") return M;
    if (str == "KD") return KD;
    throw std::invalid_argument("Invalid index type: " + str);
}

std::string indexTypeToString(IndexType type) {
    switch (type) {
        case LS:
            return "LS";
        case RP:
            return "RP";
        case VPP_OL:
            return "VPP-OL";
        case VPP:
            return "VPP";
        case VP:
            return "VP";
        case MVP:
            return "MVP";
        case M:
            return "M";
        case KD:
            return "KD";
        default:
            return "Unknown";
    }
}
#endif //HDLI_INDEX_H
