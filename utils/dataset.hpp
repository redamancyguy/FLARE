//
// Created by redam on 7/22/2024.
//

#ifndef DATSET_H
#define DATSET_H
#include <algorithm>
#include <string>
#include <iostream>
#include <vector>
#include <cstddef>
#include <filesystem>
#include <istream>
#include <fstream>
#include <array>
#include <unordered_map>
#include <cstring>


#include "DEFINE.h"


#ifdef UBUNTU_DESKTOP
inline std::string source_path = "/media/redamancyguy/DataBuffer/exp_data/dataset/";
inline std::string father_path = "/media/redamancyguy/3F59-F643/dataset/";
// inline std::string father_path = "/media/redamancyguy/DATA/dataset/";
#elif defined(UBUNTU_DESKTOP_WSL)
inline std::string source_path = "/mnt/e/exp_data/dataset/";
inline std::string father_path = "/mnt/d/dataset/";
#elif defined(UBUNTU_NOTEBOOK)
inline std::string source_path = "/mnt/d/dataset/";
inline std::string father_path = "/mnt/d/dataset/";
#elif defined(UBUNTU_CPU_SERVER)
inline std::string father_path = "/media/redamancyguy/a7f5e0ec-80db-4d7d-a9ac-a8f35282b0d4/dataset/";
#endif

template<class T, std::size_t DIM>
inline std::vector<std::array<T, DIM> > convertToVector(const std::vector<std::array<unsigned char, DIM> > &array,
                                                        const bool add_random = false) {
    std::vector<std::array<T, DIM> > result;
    result.reserve(array.size()); // Reserve space to avoid reallocations
    std::uniform_real_distribution<num_t> dis(-0.05, 0.05);
    for (const auto &item: array) {
        std::array<T, DIM> key{};
        for (std::size_t j = 0; j < DIM; ++j) {
            if (!add_random) {
                key[j] = static_cast<T>(item[j]);
            } else {
                key[j] = static_cast<T>(item[j]) + dis(gen);
            }
        }
        result.push_back(key);
    }
    return result;
}


template<typename T, typename T2, std::size_t DIM>
inline std::array<T, DIM> vectorToArray(const std::vector<T2> &vec) {
    std::array<T, DIM> arr = {};
    std::copy_n(vec.begin(), std::min(DIM, vec.size()), arr.begin());
    return arr;
}

template<class T, idx_t DIM>
inline void write(const std::string &file_path, const std::vector<std::array<T, DIM> > &data) {
    std::ofstream outFile(file_path, std::ios::binary);
    if (!outFile) { throw std::ios_base::failure("Open file failed!"); }
    for (const auto &i: data) { outFile.write(reinterpret_cast<const char *>(&i), sizeof(std::array<T, DIM>)); }
    outFile.close();
}

template<class T, idx_t DIM>
inline std::vector<std::array<T, DIM> > read(const std::string &file_path, const idx_t n = -1) {
    std::vector<std::array<T, DIM> > data;
    std::ifstream inFile(file_path, std::ios::binary);
    if (!inFile) {
        throw std::ios_base::failure("Open file failed!");
    }
    inFile.seekg(0, std::ios::end);
    idx_t total_size = inFile.tellg() / sizeof(std::array<T, DIM>);
    if (n != -1) {
        total_size = std::min(n, total_size);
    }
    inFile.seekg(0, std::ios::beg);
    data.resize(total_size);
    for (auto &i: data) {
        inFile.read(reinterpret_cast<char *>(&i), sizeof(std::array<T, DIM>));
    }
    inFile.close();
    return data;
}





namespace TPC_H {
    inline auto tcp_bin_data_path = father_path + "TCP_H_data/lineitem.bin";
    inline auto tcp_csv_data_path = father_path + "TCP_H_data/lineitem.csv";

    template<class T, idx_t DIM = 8>
    inline auto read_bin_TCP_H(const std::string &filename = tcp_bin_data_path) {
        return read<T, DIM>(filename);
    }

    template<class T, idx_t DIM = 8>
    inline auto read_csv_TCP_H(const std::string &filename = tcp_csv_data_path) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::ios_base::failure("open file failed !");
        }
        std::string line;
        std::vector<std::array<T, DIM> > data;
        std::getline(file, line);
        std::cout << line << std::endl;
        idx_t count = 0;
        std::stringstream ss(line);
        std::string cell;
        std::vector<T> row;
        while (std::getline(file, line)) {
            std::array<T, DIM> p;
            if (constexpr int range = 1000000; count % range == 0) {
                std::cout << count / range << ":" << file.tellg() << std::endl;
            }
            ss = std::stringstream(line);
            cell.clear();
            row.clear();
            while (std::getline(ss, cell, ',')) { row.push_back(std::stod(cell)); }
            data.push_back(vectorToArray<T, T, DIM>(row));
            ++count;
        }
        file.close();
        return data;
    }
}

inline std::vector<std::array<float, 192> > read_audio(const idx_t size = -1) {
    const std::string file = father_path + "Audio.data";
    std::ifstream in(file.c_str(), std::ios::binary);
    if (!in) {
        std::cerr << "Fail to find data file!" << std::endl;
        exit(1); // Use a non-zero exit code for failure
    }

    unsigned int header[3] = {};
    in.read(reinterpret_cast<char *>(header), sizeof(header));
    idx_t N = header[1];
    const auto dim = header[2];
    if (dim != 192) {
        throw std::runtime_error("Bad dimension number!");
    }
    if (size > 0 && size < N) {
        N = size;
    }
    std::vector<std::array<float, 192> > result;
    result.resize(N);
    for (int i = 0; i < N; ++i) {
        in.read(reinterpret_cast<char *>(&result[i]), sizeof(std::array<float, 192>));
    }
    return result;
}

namespace MNIST {
    inline int ReverseInt(const int i) {
        const unsigned char ch1 = i & 255;
        const unsigned char ch2 = (i >> 8) & 255;
        const unsigned char ch3 = (i >> 16) & 255;
        const unsigned char ch4 = (i >> 24) & 255;
        return (static_cast<int>(ch1) << 24) + (static_cast<int>(ch2) << 16) + (static_cast<int>(ch3) << 8) + ch4;
    }

    inline std::vector<unsigned char> read_Mnist_Label(const std::string &filename) {
        std::vector<unsigned char> labels;
        if (std::ifstream file(filename, std::ios::binary); file.is_open()) {
            int magic_number = 0;
            int number_of_images = 0;
            file.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number));
            file.read(reinterpret_cast<char *>(&number_of_images), sizeof(number_of_images));
            magic_number = ReverseInt(magic_number);
            number_of_images = ReverseInt(number_of_images);
            for (int i = 0; i < number_of_images; i++) {
                unsigned char label = 0;
                file.read(reinterpret_cast<char *>(&label), sizeof(label));
                labels.push_back(label);
            }
        }
        return labels;
    }

    inline std::vector<std::vector<unsigned char> > read_Mnist_Images(
        const std::string &filename = father_path + "MNIST/t10k-images.idx3-ubyte") {
        std::vector<std::vector<unsigned char> > images;
        if (std::ifstream file(filename, std::ios::binary); file.is_open()) {
            int magic_number = 0;
            int number_of_images = 0;
            int n_rows = 0;
            int n_cols = 0;
            file.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number));
            file.read(reinterpret_cast<char *>(&number_of_images), sizeof(number_of_images));
            file.read(reinterpret_cast<char *>(&n_rows), sizeof(n_rows));
            file.read(reinterpret_cast<char *>(&n_cols), sizeof(n_cols));
            magic_number = ReverseInt(magic_number);
            number_of_images = ReverseInt(number_of_images);
            n_rows = ReverseInt(n_rows);
            n_cols = ReverseInt(n_cols);

            for (int i = 0; i < number_of_images; i++) {
                std::vector<unsigned char> tp;
                for (int r = 0; r < n_rows; r++) {
                    for (int c = 0; c < n_cols; c++) {
                        unsigned char image = 0;
                        file.read(reinterpret_cast<char *>(&image), sizeof(image));
                        tp.push_back(image);
                    }
                }
                images.push_back(tp);
            }
        }
        return images;
    }

#include <vector>
#include <array>
#include <utility> // For std::pair

    template<class T, idx_t DIM = 28 * 28>
    inline auto read_mnist_test() {
        const auto images = read_Mnist_Images();
        std::vector<std::array<T, DIM> > dataset;
        for (std::int64_t i = 0; i < images.size(); ++i) {
            auto array_ = vectorToArray<T, DIM>(images[i]);
            dataset.push_back(array_);
        }
        return dataset;
    }

    template<class T, idx_t DIM = 28 * 28>
    inline auto read_mnist_train(const idx_t size = -1) {
        auto images = read_Mnist_Images(father_path + "MNIST/train-images.idx3-ubyte");
        if (size > 0 && size < images.size()) {
            images.resize(size);
        }
        std::vector<std::array<T, DIM> > dataset;
        for (std::size_t i = 0; i < images.size(); ++i) {
            auto array_ = vectorToArray<T, unsigned char, DIM>(images[i]);
            dataset.push_back(array_);
        }
        return dataset;
    }
}

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace process_files {
    #include <functional>

    namespace fs = std::filesystem;
    using FileHandler = std::function<bool(const fs::path &)>;

    inline void process_files_in_directory(const fs::path &directory, const FileHandler &handler) {
        if (!fs::exists(directory) || !fs::is_directory(directory)) {
            std::cout << "Invalid directory path." << std::endl;
            return;
        }

        for (const auto &entry: fs::recursive_directory_iterator(directory)) {
            if (fs::is_regular_file(entry.path())) {
                if(auto result = handler(entry.path()); !result) {
                    return;
                }
            }
        }
    }
}

namespace IMAGE {
    namespace fs = std::filesystem;
    const static auto image_path = father_path + "earth_images/images/";

    inline std::vector<std::string> getFilesInDirectory(const std::string &directoryPath) {
        std::vector<std::string> fileNames;
        for (const auto &entry: fs::directory_iterator(directoryPath)) {
            if (entry.is_regular_file()) {
                fileNames.push_back(entry.path().filename().string());
            }
        }
        return fileNames;
    }


    // split the image data into small block as a vector
    template<std::size_t DIM>
    inline std::vector<std::array<unsigned char, DIM> > splitImageToBlocks(
        const std::string &imagePath, const int blockWidth, const int blockHeight) {
        // Validate that DIM matches the block size
        if (blockWidth * blockHeight * 3 != DIM) {
            throw StringException("dimension does not match image block size!");
        }


        int width, height, channels;
        unsigned char *data = stbi_load(imagePath.c_str(), &width, &height, &channels, 0);
        std::vector<std::array<unsigned char, DIM> > blocks;
        if (width * height * channels / DIM == 0) {
            return blocks;
        }
        std::cout << width << ":" << height << ":" << channels << ":" << DIM << std::endl;
        blocks.reserve(width * height * channels / DIM);
        if (!data) {
            std::cerr << "Failed to load image: " << imagePath << std::endl;
            return blocks;
        }

        const int numHorizontalBlocks = width / blockWidth;
        const int numVerticalBlocks = height / blockHeight;

        for (int y = 0; y < numVerticalBlocks; ++y) {
            for (int x = 0; x < numHorizontalBlocks; ++x) {
                std::array<unsigned char, DIM> block;
                const int startX = x * blockWidth;
                const int startY = y * blockHeight;
                int block_id = 0;
                for (int dy = 0; dy < blockHeight; ++dy) {
                    for (int dx = 0; dx < blockWidth; ++dx) {
                        const int pixelX = startX + dx;
                        const int pixelY = startY + dy;
                        for (int channel = 0; channel < channels; ++channel) {
                            const int idx = (pixelY * width + pixelX) * channels + channel;
                            block[block_id++] = data[idx];
                        }
                    }
                }
                if (block_id != DIM) {
                    //pass
                    // std::cerr << "Mismatch in block dimensions!"<<std::endl;
                } else {
                    blocks.push_back(block);
                }
            }
        }
        stbi_image_free(data);
        return blocks;
    }


    template<std::size_t DIM>
    inline std::vector<std::array<unsigned char, DIM> > read(const std::string &file_path, const idx_t n = -1) {
        std::vector<std::array<unsigned char, DIM> > data;
        std::ifstream inFile(file_path, std::ios::binary);

        if (!inFile) {
            throw std::ios_base::failure("Open file failed!");
        }

        // Get the total size of the file in bytes
        inFile.seekg(0, std::ios::end);
        const idx_t file_size = inFile.tellg();
        inFile.seekg(0, std::ios::beg);

        // Calculate the number of items
        const idx_t item_size = sizeof(std::array<unsigned char, DIM>);
        const idx_t total_items = file_size / item_size;

        // Determine the number of items to read
        const idx_t items_to_read = (n == static_cast<idx_t>(-1)) ? total_items : std::min(n, total_items);
        data.reserve(items_to_read);

        std::array<unsigned char, DIM> item;
        for (idx_t i = 0; i < items_to_read; ++i) {
            inFile.read(reinterpret_cast<char *>(&item), item_size);
            if (!inFile) {
                throw std::ios_base::failure("Read file failed!");
            }
            data.push_back(item);
        }

        inFile.close();
        return data;
    }

    template<std::size_t DIM>
    inline void save_all_images_to_vector(const std::string &bin_data_path, const int blockWidth, const int blockHeight, const idx_t n = -1) {
        int64_t n_samples = 0;
        int64_t dim = DIM;
        if (blockWidth * blockHeight * 3 != DIM) {
            throw StringException("dimension does not match image block size!");
        }
        std::ofstream outFile(bin_data_path, std::ios::binary);
        if (!outFile) { throw std::ios_base::failure("open file failed!"); }
        idx_t count = 0;
        for (const auto &i: getFilesInDirectory(image_path)) {
            if (n > 0 && count > n) {
                break;
            }
            ++count;
            std::string path = image_path + i;
            std::cout << path << std::endl;
            for (auto &j: splitImageToBlocks<DIM>(path, blockWidth, blockHeight)) {
                outFile.write(reinterpret_cast<char *>(&j), sizeof(j));
                outFile.flush();
            }
        }
        outFile.close();
    }


    inline std::vector<float> loadImageAsFloatVector(const char *filename, int &width, int &height, int &channels) {
        unsigned char *data = stbi_load(filename, &width, &height, &channels, 0);
        if (!data) {
            std::cerr << "Failed to load image: " << filename << std::endl;
            return {};
        }
        const size_t numPixels = width * height * channels;
        std::vector<float> floatData(numPixels);
        for (size_t i = 0; i < numPixels; ++i) {
            floatData[i] = static_cast<float>(data[i]) / 255.0f;
        }
        stbi_image_free(data);
        return floatData;
    }
}


namespace cifar_100 {
    constexpr int NUM_IMAGES = 50000;
    constexpr int IMAGE_SIZE = 32;
    constexpr int NUM_CHANNELS = 3;
    constexpr int IMAGE_BYTES = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS;
    constexpr int LABEL_BYTES = 2;

    inline std::vector<unsigned char> readFile(const std::string &filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            throw StringException("open file error: ");
        }
        return std::vector<unsigned char>(std::istreambuf_iterator<char>(file), {});
    }

    inline void parseCIFAR100(const std::vector<unsigned char> &data,
                              std::vector<std::array<unsigned char, IMAGE_BYTES> > &images,
                              std::vector<int> &fine_labels, std::vector<int> &coarse_labels) {
        int index = 0;
        for (int i = 0; i < NUM_IMAGES; ++i) {
            int fine_label = data[index++];
            int coarse_label = data[index++];
            fine_labels.push_back(fine_label);
            coarse_labels.push_back(coarse_label);

            std::array<unsigned char, IMAGE_BYTES> img;
            for (int j = 0; j < IMAGE_BYTES; ++j) {
                img[j] = data[index++];
            }
            images.push_back(img);
        }
    }

    inline void savePNG(const std::string &filename, const std::array<unsigned char, IMAGE_BYTES> &image) {
        std::vector<unsigned char> rgb_image(IMAGE_BYTES);
        for (int c = 0; c < NUM_CHANNELS; ++c) {
            for (int i = 0; i < IMAGE_SIZE * IMAGE_SIZE; ++i) {
                rgb_image[i * NUM_CHANNELS + c] = image[c * IMAGE_SIZE * IMAGE_SIZE + i];
            }
        }
        if (!stbi_write_png(filename.c_str(), IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS, rgb_image.data(),
                            IMAGE_SIZE * NUM_CHANNELS)) {
            std::cerr << "can save file: " << filename << std::endl;
        }
    }

    inline auto read_cifar_100_uchar(const std::string &file_path = father_path + "cifar-100-binary/train.bin") {
        const std::vector<unsigned char> data = readFile(file_path);
        std::vector<std::array<unsigned char, IMAGE_BYTES> > images;
        std::vector<int> fine_labels;
        std::vector<int> coarse_labels;
        parseCIFAR100(data, images, fine_labels, coarse_labels);
        // for (int i = 0; i < 5; ++i) { // Save only first 5 images for demonstration
        //     std::string filename = "image_" + std::to_string(i) + ".png";
        //     savePNG(filename, images[i]);
        //     std::cout << "Saved " << filename << " with fine label: " << fine_labels[i] << ", coarse label: " << coarse_labels[i] << std::endl;
        // }
        return images;
    }

    template<class T = unsigned char>
    inline auto read_cifar_100(const idx_t size = -1) {
        const std::string file_path = father_path + "cifar-100-binary/train.bin";
        auto data = read_cifar_100_uchar(file_path);
        if (size > 0 && size < data.size()) {
            data.resize(size);
        }
        std::vector<std::array<T, IMAGE_BYTES> > result; // Ensure std::array and pld_t are defined
        for (int i = 0; i < data.size(); ++i) {
            std::array<T, IMAGE_BYTES> new_array;
            for (int j = 0; j < IMAGE_BYTES; ++j) {
                new_array[j] = data[i][j];
            }
            result.emplace_back(new_array);
        }
        return result;
    }
}

inline std::vector<std::vector<float> > read_fvecs(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    std::vector<std::vector<float> > data;
    idx_t count = 0;
    while (file) {
        int dim;
        file.read(reinterpret_cast<char *>(&dim), sizeof(int));
        if (!file) break; // End of file
        std::vector<float> vec(dim);
        file.read(reinterpret_cast<char *>(vec.data()), dim * sizeof(float));
        if (!file) break; // End of file

        data.push_back(vec);
        ++count;
    }
    return data;
}


inline std::vector<std::vector<uint8_t> > read_bvecs(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::vector<std::vector<uint8_t> > data;

    while (file) {
        int dim;
        file.read(reinterpret_cast<char *>(&dim), sizeof(int));
        if (!file) break; // End of file

        std::vector<uint8_t> vec(dim);
        file.read(reinterpret_cast<char *>(vec.data()), dim);
        if (!file) break; // End of file

        data.push_back(vec);
    }
    return data;
}


template<class T>
inline auto read_deep_1m(const idx_t size = -1) {
    std::vector<std::vector<float> > data = read_fvecs(father_path + "deep1M/deep1M_base.fvecs");
    if (size > 0 && size < data.size()) {
        data.resize(size);
    }
    std::vector<std::array<T, 256> > result;
    for (idx_t i = 0; i < data.size(); ++i) {
        if (data[i].size() != 256) {
            throw std::runtime_error("bad data dimension!");
        }
        std::array<T, 256> array;
        std::copy(data[i].begin(), data[i].end(), array.begin());
        result.push_back(array);
    }
    return result;
}

template<class T>
inline auto read_sift_1m(const idx_t size = -1) {
    std::vector<std::vector<float> > data = read_fvecs(father_path + "sift_1m/sift_base.fvecs");
    if (size > 0 && size < data.size()) {
        data.resize(size);
    }
    std::vector<std::array<T, 128> > result;
    for (idx_t i = 0; i < data.size(); ++i) {
        if (data[i].size() != 128) {
            throw std::runtime_error("bad data dimension!");
        }
        std::array<T, 128> array;
        std::copy(data[i].begin(), data[i].end(), array.begin());
        result.push_back(array);
    }
    return result;
}

template<class T>
inline auto read_gist(const idx_t size = -1) {
    std::vector<std::vector<float> > data = read_fvecs(father_path + "gist/gist_base.fvecs");
    if (size > 0 && size < data.size()) {
        data.resize(size);
    }
    std::vector<std::array<T, 960> > result;
    for (idx_t i = 0; i < data.size(); ++i) {
        if (data[i].size() != 960) {
            throw std::runtime_error("bad data dimension!");
        }
        std::array<T, 960> array;
        std::copy(data[i].begin(), data[i].end(), array.begin());
        result.push_back(array);
    }
    return result;
}


namespace tiny_images {
    constexpr int DESCRIPTOR_SIZE = 512; // 根据实际的描述符大小调整

    template<class T>
    inline auto readTinyGist(const idx_t num = -1) {
        const std::string &filename = father_path + "tiny_images/tinygist80million.bin";
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            throw StringException("Cannot open file: " + filename);
        }

        // Calculate the number of descriptors
        file.seekg(0, std::ios::end);
        const std::streampos fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        idx_t numDescriptors = fileSize / DESCRIPTOR_SIZE;
        if (num > 0) {
            numDescriptors = num;
        }
        std::vector<std::array<float, DESCRIPTOR_SIZE> > descriptors(numDescriptors,
                                                                     std::array<float, DESCRIPTOR_SIZE>());
        // Read descriptors
        for (idx_t i = 0; i < numDescriptors; ++i) {
            file.read(reinterpret_cast<char *>(descriptors[i].data()), DESCRIPTOR_SIZE * sizeof(float));
            if (!file) {
                throw StringException("Error reading file at descriptor ");
            }
        }
        std::vector<std::array<T, DESCRIPTOR_SIZE> > result;
        for (idx_t i = 0; i < descriptors.size(); ++i) {
            result.emplace_back(descriptors[i]);
        }
        return result;
    }


    inline auto readTinyMetadata(const std::string &filename = father_path + "tiny_images/tiny_metadata.bin",
                                 const idx_t num = -1) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            throw StringException("Cannot open file: " + filename);
        }
        constexpr int RECORD_SIZE = sizeof(int);
        file.seekg(0, std::ios::end);
        const std::streampos fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        const idx_t numRecords = fileSize / RECORD_SIZE;
        std::vector<int> labels(numRecords);

        file.read(reinterpret_cast<char *>(labels.data()), fileSize);

        for (idx_t i = 0; i < 10; ++i) {
            std::cout << "Label " << i << ": " << labels[i] << std::endl;
        }
        return labels;
    }
}

namespace ANN_SIFT1B {
    constexpr int DESCRIPTOR_SIZE = 128; // SIFT 描述符的维度

    float max_threshold = std::numeric_limits<float>::max(); // Or any other threshold you deem too large
    inline bool checkArray(const std::array<float, 128> &descriptor) {
        for (float value: descriptor) {
            if (std::isnan(value)) {
                std::cout << "Found NaN value!" << std::endl;
                return true;
            }
            if (std::isinf(value)) {
                std::cout << "Found Inf value!" << std::endl;
                return true;
            }
            if (value > max_threshold) {
                std::cout << "Found extremely large value: " << value << std::endl;
                return true;
            }
        }
        return false;
    }


    inline auto readSIFT1B(const idx_t num = -1) {
        const std::string filename = father_path + "1milliard.p1.siftbin";
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            throw StringException("Cannot open file: " + filename);
        }
        file.seekg(0, std::ios::end);
        idx_t fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        int dim;
        file.read(reinterpret_cast<char *>(&dim), sizeof(int));
        if (dim != DESCRIPTOR_SIZE) {
            throw StringException("bad dimension !");
        }
        fileSize -= sizeof(int);
        idx_t numDescriptors = fileSize / (DESCRIPTOR_SIZE * sizeof(unsigned char));
        if (num > 0 && num < numDescriptors) {
            numDescriptors = num;
        }
        std::vector<std::array<unsigned char, DESCRIPTOR_SIZE> > descriptors(
            numDescriptors, std::array<unsigned char, DESCRIPTOR_SIZE>());

        for (idx_t i = 0; i < numDescriptors; ++i) {
            file.read(reinterpret_cast<char *>(descriptors[i].data()), DESCRIPTOR_SIZE * sizeof(unsigned char));
            if (!file) {
                throw StringException("Error reading file at descriptor ");
            }
        }

        return descriptors;
    }
}


inline void load_data_LSH_APG(const std::string &path) {
    std::ifstream in(path.c_str(), std::ios::binary);
    while (!in) {
        printf("Fail to find data file!\n");
        exit(0);
    }
    std::vector<std::vector<float> > data;
    unsigned int header[3] = {};
    assert(sizeof header == 3 * 4);
    in.read(reinterpret_cast<char *>(header), sizeof(header));
    assert(header[0] == sizeof(float));
    const unsigned long long N = header[1];
    const unsigned long long dim = header[2];

    data.resize(N);
    for (int i = 0; N; ++i) {
        break;
        data[i].resize(dim);
        //in.seekg(sizeof(float), std::ios::cur);
        in.read(reinterpret_cast<char *>(data[i].data()), dim * sizeof(float));
    }

    std::cout << "Load from new file: " << path << "\n";
    std::cout << "N=    " << N << "\n";
    std::cout << "dim=  " << dim << "\n\n";

    in.close();
}

// void printHeapMemoryUsage() {
//     struct mallinfo mi = mallinfo();
//
//     std::cout << "Total allocated space (arena): " << mi.arena / 1024 << " KB" << std::endl;
//     std::cout << "Number of free chunks (ordblks): " << mi.ordblks << std::endl;
//     std::cout << "Total free space (fordblks): " << mi.fordblks / 1024 << " KB" << std::endl;
//     std::cout << "Total allocated memory by malloc (uordblks): " << mi.uordblks / 1024 << " KB" << std::endl;
// }


// inline void saveStringToFile(const std::string &data, const std::string &filePath) {
//     if (std::ofstream outFile(filePath); outFile.is_open()) {
//         outFile << data;
//         outFile.close();
//         std::cout << "Data successfully saved to " << filePath << std::endl;
//     } else {
//         std::cerr << "Error: Could not open file " << filePath << std::endl;
//     }
// }
//
// inline std::string loadFileToString(const std::string &filePath) {
//     std::ifstream inFile(filePath);
//     if (!inFile) {
//         throw std::runtime_error("Error: Could not open file " + filePath);
//     }
//
//     std::string fileContent((std::istreambuf_iterator<char>(inFile)),
//                             std::istreambuf_iterator<char>());
//     return fileContent;
// }


inline idx_t kNN_query_num = 100;
inline idx_t range_query_num = 1;
const std::vector<std::string> datasets = {
    // "TPC-H",
    "Word2Vec",
    "MIRFLICKR",
    "TinyImages",
    "Sift1B",
    "Deep1M",
    "MNIST",
    "Gist",
    "Sift1M",
    // "CIFAR",
    "Audio",
    "GoogleEarth",
};
inline idx_t point_query_num = 1;
inline idx_t dataset_size = 1e8;
// inline idx_t dataset_size = 2e7;
// inline idx_t dataset_size = 1e4;
// inline idx_t dataset_size = -1;


#endif //DATSET_H
