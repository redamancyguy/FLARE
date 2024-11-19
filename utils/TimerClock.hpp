//
// Created by wenli on 2022/12/29.

//

#ifndef TimerClock_hpp
#define TimerClock_hpp

#include <iostream>
#include <chrono>

class TimerClock {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> _ticker;
public:
    TimerClock() {
        tick();
    }

    ~TimerClock() = default;

    void tick() {
        _ticker = std::chrono::high_resolution_clock::now();
    }

    [[nodiscard]] double second() const {
        return static_cast<double>(nanoSec()) * 1e-9;
    }

    [[nodiscard]] double milliSec() const {
        return static_cast<double>(nanoSec()) * 1e-6;
    }

    [[nodiscard]] double microSec() const {
        return static_cast<double>(nanoSec()) * 1e-3;
    }
    [[nodiscard]] long long nanoSec() const {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now() - _ticker).count();
    }

};


#endif //TimerClock_hpp
