//
// Created by 孙文礼 on 2023/1/11.
//

#ifndef DEFINES_H
#define DEFINES_H



#define CHECK_POINT_RESULT
// #define CHECK_RANGE_RESULT
#define CHECK_SIMILAR_RANGE_RESULT
#define CHECK_KNN_RESULT
#define CHECK_AKNN_RESULT

// #define COUNT_SCAN_NUM

#define USING_INTERCEPT
#define LP2_DISTANCE

#define PRINT_LOCATION() \
std::cout << "File: " << __FILE__ << "\n" \
<< "Line: " << __LINE__ << "\n" \
<< "Function: " << __FUNCTION__ << "\n"

#define PRINT(i) std::cout << (i) << " "
#define PRINTLN(i) std::cout << (i) << std::endl
#define ERR(i) std::cerr << (i) <<  " "
#define ERRLN(i) std::cerr << (i) << std::endl
#include <random>
#include <atomic>
#include <mutex>
#include <iostream>
#include <algorithm>

namespace color {
    constexpr auto reset = "\033[0m";
    constexpr auto red = "\033[31m";
    constexpr auto green = "\033[32m";
    constexpr auto yellow = "\033[33m";
    constexpr auto blue = "\033[34m";
    constexpr auto magenta = "\033[35m";
    constexpr auto cyan = "\033[36m";
    constexpr auto white = "\033[37m";
    constexpr auto bold = "\033[1m";
    constexpr auto underline = "\033[4m";
    constexpr auto bright_red = "\033[91m";
    constexpr auto bright_green = "\033[92m";
    constexpr auto bright_yellow = "\033[93m";
    constexpr auto bright_blue = "\033[94m";
    constexpr auto bright_magenta = "\033[95m";
    constexpr auto bright_cyan = "\033[96m";
    constexpr auto bright_white = "\033[97m";
    class ColorManager {
    public:
        ColorManager() {
            colors = { red, green, yellow,
                    blue, magenta, cyan, white, bright_red, bright_green,
                    bright_yellow, bright_blue, bright_magenta,
                    bright_cyan, bright_white
            };
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(colors.begin(),colors.end(), g);
        }

        std::string getNextColor() {
            return colors[currentIndex++ % colors.size()];
        }

        static std::string resetColor() {
            std::cout <<reset;
            return reset;
        }

    private:
        std::vector<std::string> colors;
        size_t currentIndex = 0;
    };
}

typedef float num_t;
// typedef float num_t;
typedef int64_t idx_t;
// typedef int idx_t;


#define DEFAULT_FANOUT 4
#define DEFAULT_LEAF_SIZE 10

#define COUNT_SCAN_NUM
#ifdef COUNT_SCAN_NUM
idx_t scan_count = 0;
idx_t scan_node_count = 0;
#endif

#if defined _WIN64 || defined _WIN32
#include <windows.h>
static void enable_ansi_escape_sequences() {
    const auto hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut == INVALID_HANDLE_VALUE) {
        return;
    }
    DWORD dwMode = 0;
    if (!GetConsoleMode(hOut, &dwMode)) {
        return;
    }
    dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    SetConsoleMode(hOut, dwMode);
}
#endif
#if defined(unix) || defined(__unix__)
#include <unistd.h>
#include <fstream>
#include <cerrno>
#include <iostream>

#include <fstream>
#include <string>
#include <sstream>

inline long long MemoryInfo() {
    std::ifstream status_file("/proc/self/status");
    long long memory = -1;  // 初始值为 -1, 以防解析失败
    if (status_file.is_open()) {
        std::string line;
        while (std::getline(status_file, line)) {
            if (line.find("VmRSS:") == 0) {
                std::istringstream iss(line.substr(6));  // 读取 "VmRSS:" 后面的部分
                std::string memStr;
                iss >> memStr;  // 只读取第一个数值
                try {
                    memory = std::stoll(memStr);  // 将字符串转换为 long long
                } catch (const std::invalid_argument& e) {
                    // 解析失败，保持 memory 为 -1
                    memory = -1;
                }
                break;
            }
        }
        status_file.close();
    }
    return memory >= 0 ? memory * 1024 : memory;  // 返回以字节为单位的值
}

class MemoryLog {
    long long memory_size = 0;
public:
    void tick() {
        memory_size = MemoryInfo();
    }
    MemoryLog() {
        tick();
    }
    long long get_memory() {
        return MemoryInfo() - memory_size;
    }
};

#else
#include <iostream>
#include <windows.h>
#include <psapi.h>
#pragma comment(lib,"psapi.lib")

idx_t MemoryInfo()
{
    HANDLE handle = GetCurrentProcess();
    PROCESS_MEMORY_COUNTERS pmc;
    GetProcessMemoryInfo(handle, &pmc, sizeof(pmc));
    std::cout << "Memory Usage: " << pmc.WorkingSetSize / (1024 * 1024) << "M/"
    << pmc.PeakWorkingSetSize / (1024 * 1024) << "M + "
    << pmc.PagefileUsage / (1024 * 1024) << "M/"
    << pmc.PeakPagefileUsage / (1024 * 1024) << "M." << std::endl;
    return pmc.WorkingSetSize;
}
#endif

#include <execinfo.h>
#include <iostream>
#include <cstdlib>

inline void printStackTrace() {
    constexpr int MAX_FRAMES = 64;
    void* callstack[MAX_FRAMES];
    const int frames = backtrace(callstack, MAX_FRAMES);
    char** symbols = backtrace_symbols(callstack, frames);

    if (symbols == nullptr) {
        std::cerr << "Failed to retrieve stack trace." << std::endl;
        return;
    }

    std::cout << "Stack trace:" << std::endl;
    for (int i = 0; i < frames; ++i) {
        std::cout << symbols[i] << std::endl;
    }

    free(symbols);
}

#include <string>
#include <stdexcept>
#include <utility>
#include <iostream>
#include <vector>

class StringException final : public std::runtime_error {
public:
    explicit StringException(const std::string &message) : std::runtime_error(message) { }
};




#include <iostream>
#include <string>
#include <mutex>
#include <iomanip>
#include <chrono>
#include <iostream>
#include <mutex>
#include <string>
#include <utility>

class progress_display {
public:
    explicit progress_display(
        const int64_t expected_count,
        std::ostream &os = std::cout,
        std::string s1 = "",
        std::string s2 = "",
        std::string s3 = "")
        : m_os(os), m_s1(std::move(s1)), m_s2(std::move(s2)), m_s3(std::move(s3)) {
        restart(expected_count);
    }

    void restart(const int64_t expected_count) {
        std::unique_lock<std::mutex> lock(mtx);
        _count = _next_tic_count = _tic = 0;
        _expected_count = expected_count;
        m_os << m_s1 << "0%   10   20   30   40   50   60   70   80   90   100%\n"
             << m_s2 << "|----|----|----|----|----|----|----|----|----|----|"
             << std::endl
             << m_s3;
        if (!_expected_count) {
            _expected_count = 1;
        }
        start_time = std::chrono::steady_clock::now();  // Reset start time
    }

    int64_t operator +=(int64_t increment) {
        std::unique_lock<std::mutex> lock(mtx);
        _count += increment;
        if (_count >= _next_tic_count) {
            display_tic();
        }
        return _count;
    }

    int64_t operator ++() {  // Prefix increment
        return operator +=(1);
    }

    // New function for incrementing with extra data
    int64_t increment_with_data(const std::string& extra_data) {
        std::unique_lock<std::mutex> lock(mtx);
        ++_count;
        if (_count >= _next_tic_count) {
            display_tic(extra_data); // Display progress bar with extra data
        }
        return _count;
    }

private:
    std::ostream &m_os;
    const std::string m_s1;
    const std::string m_s2;
    const std::string m_s3;
    mutable std::mutex mtx;
    int64_t _count{0}, _expected_count{0}, _next_tic_count{0};
    int64_t _tic{0};
    std::chrono::steady_clock::time_point start_time;

    void display_tic(const std::string& extra_data = "") {
        const auto tics_needed = static_cast<unsigned>((static_cast<double>(_count) / static_cast<double>(_expected_count)) * 50.0);

        // Move to start of line and clear line for proper overwriting
        m_os << "\r";

        m_os << m_s2 << "|";
        for (unsigned i = 0; i < 50; ++i) {
            if (i < tics_needed) {
                m_os << '*';
            } else {
                m_os << '-';
            }
        }
        m_os << "|";

        // Calculate remaining time
        auto remaining_time = calculate_remaining_time();

        // Append extra data and remaining time to the right of the progress bar
        if (!extra_data.empty()) {
            m_os << " " << extra_data;
        }
        m_os << " Remaining time: " << remaining_time << "s";

        // Ensure the progress and extra data are displayed immediately
        m_os.flush();

        _next_tic_count = static_cast<size_t>(static_cast<double>(_expected_count) * (static_cast<double>(_tic) / 50.0));

        // Check if the task is done (100% completed)
        if (_count == _expected_count) {
            m_os << std::endl;  // Output an extra newline after completion
        }
    }

    // Function to calculate remaining time in seconds
    double calculate_remaining_time() const {
        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_time = now - start_time;

        if (_count == 0) {
            return 0.0;  // Avoid division by zero
        }

        double time_per_count = elapsed_time.count() / _count;
        double remaining_count = _expected_count - _count;

        return time_per_count * remaining_count;
    }
};

inline std::vector<idx_t> parse_k_values(const std::string& k_values_str) {
    std::vector<idx_t> k_values;
    std::stringstream ss(k_values_str);
    std::string item;
    while (std::getline(ss, item, ',')) {
        k_values.push_back(std::stoll(item));
    }
    return k_values;
}

template<typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        os << vec[i];
        if (i != vec.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}
#include <random>
// inline std::random_device rd;
// inline std::mt19937 gen(rd());
inline std::mt19937 gen(42);

#endif //DEFINES_H
