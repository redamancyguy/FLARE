#include <iostream>
#include <fstream>
#include <torch/torch.h>
#include <vector>
#include <stdexcept>
class ReadSourceFile{
public:
    static void convert_number_batch() {//Word2Vec 1.3m 300
        auto filename = source_path + "numberbatch-en-19.08.txt/numberbatch-en.txt";
        auto target_path = father_path + "tensors/Word2Vec1.tensor";
        std::ifstream infile(filename);
        std::ofstream outfile(target_path);
        if (!infile.is_open()) {
            throw MyException("Cannot open file " + filename + " for reading.");
        }
        if (!outfile.is_open()) {
            throw MyException("Cannot open file " + target_path + " for writing.");
        }
        int64_t n_sample = 0;
        int64_t dim = 300;
        outfile.write(reinterpret_cast<const char*>(&n_sample), sizeof(int64_t));
        outfile.write(reinterpret_cast<const char*>(&dim), sizeof(int64_t));
        std::string line;
        std::vector<float> vector_values;
        while (std::getline(infile, line)) {
            std::istringstream iss(line);
            if (std::string word; !(iss >> word)) { continue; }
            float value;
            vector_values.clear();
            while (iss >> value) {
                vector_values.push_back(value);
            }
            if(dim != vector_values.size()) {
                continue;
            }
            outfile.write(reinterpret_cast<const char*>(vector_values.data()), sizeof(float)*dim);
            ++n_sample;
        }
        std::cout <<filename<<" converted:"<<n_sample<<" dim:"<<dim<< std::endl;
        outfile.seekp(0);
        outfile.write(reinterpret_cast<const char*>(&n_sample), sizeof(int64_t));
        outfile.write(reinterpret_cast<const char*>(&dim), sizeof(int64_t));
        outfile.close();
        infile.close();
    }
    static void convert_Word2Vec() {
        std::string filePath = source_path+"crawl-300d-2M.vec";
        auto target_path = father_path + "tensors/Word2Vec.tensor";
        std::ofstream outfile(target_path);
        std::ifstream infile(filePath);
        if (!infile.is_open()) {
            throw std::runtime_error("Cannot open file " + filePath);
        }
        if (!outfile.is_open()) {
            throw MyException("Cannot open file " + target_path + " for writing.");
        }

        int64_t dim;
        int64_t n_sample = 0;
        std::string line;
        std::getline(infile, line);
        std::istringstream headerStream(line);
        int64_t nWords;
        headerStream >> nWords >> dim;
        outfile.write(reinterpret_cast<const char*>(&nWords), sizeof(int64_t));
        outfile.write(reinterpret_cast<const char*>(&dim), sizeof(int64_t));
        std::vector<float> vectorValues(dim);
        progress_display display(nWords);
        for (int64_t i = 0; i < nWords; ++i) {
            std::getline(infile, line);
            std::istringstream lineStream(line);
            std::string word;
            lineStream >> word;
            for (int64_t j = 0; j < dim; ++j) {
                lineStream >> vectorValues[j];
            }
            outfile.write(reinterpret_cast<const char*>(vectorValues.data()), sizeof(float)*dim);
            ++n_sample;
            ++display;
        }
        std::cout <<filePath<<" converted:"<<n_sample<<" dim:"<<dim<< std::endl;
        outfile.close();
        infile.close();
    }
    static void convert_tiny_images() {
        std::string filename = source_path + "/tiny_images/tinygist80million.bin";
        std::ifstream infile(filename, std::ios::binary | std::ios::ate);
        if (!infile) {
            throw MyException("open file errir !");
        }
        auto target_path = father_path + "tensors/tiny_images.tensor";
        std::ofstream outfile(target_path);
        if (!outfile.is_open()) {
            throw MyException("Cannot open file " + target_path + " for writing.");
        }
        infile.seekg(0, std::ios::end);
        std::streamsize fileSize = infile.tellg();
        infile.seekg(0, std::ios::beg);
        int64_t dim = 384;
        int64_t n_sample = fileSize / (dim * sizeof(float));
        outfile.write(reinterpret_cast<const char*>(&n_sample), sizeof(int64_t));
        outfile.write(reinterpret_cast<const char*>(&dim), sizeof(int64_t));
        std::array<float,384> item;
        progress_display display(n_sample);
        for (size_t i = 0; i < n_sample; ++i) {
            infile.read(reinterpret_cast<char*>(item.data()), dim * sizeof(float));
            if (!infile) {
                throw MyException("read file error !");
            }
            outfile.write(reinterpret_cast<const char*>(item.data()), sizeof(float)*dim);
            ++display;
        }
        outfile.close();
        infile.close();
    }
    static void convert_Gist() {
        std::ifstream file(source_path + "gist/gist_base.fvecs", std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file");
        }
        file.seekg(0, std::ios::end);
        std::streamsize fileSize = file.tellg();
        file.seekg(0, std::ios::beg);
        auto target_path = father_path + "tensors/Gist.tensor";
        std::ofstream outfile(target_path);
        if (!outfile.is_open()) {
            throw MyException("Cannot open file " + target_path + " for writing.");
        }
        int64_t dim = 960;
        int64_t n_sample = fileSize / (dim * sizeof(float) + sizeof(int));
        outfile.write(reinterpret_cast<const char*>(&n_sample), sizeof(int64_t));
        outfile.write(reinterpret_cast<const char*>(&dim), sizeof(int64_t));
        std::vector<float> vec(dim);
        progress_display display(n_sample);
        for(idx_t i = 0;i<n_sample;++i) {
            file.read(reinterpret_cast<char *>(&dim), sizeof(int));
            if (!file) break; // End of file
            if(dim != 960) {
                throw MyException("bad dim");
            }
            file.read(reinterpret_cast<char *>(vec.data()), dim * sizeof(float));
            outfile.write(reinterpret_cast<const char*>(vec.data()), sizeof(float)*dim);
            ++display;
            if (!file) break; // End of file
        }
        outfile.close();
    }
    static void convert_goole_earth() {

        constexpr idx_t height = 8;
        constexpr idx_t width = 8;
        constexpr idx_t DIM = height * width * 3;
        auto image_path = source_path + "earth_images/images/";
        auto target_path = father_path + "google_earth.tensor";
        std::ofstream outfile(target_path, std::ios::binary);
        if (!outfile.is_open()) {
            throw MyException("Cannot open file " + target_path + " for writing.");
        }
        int64_t dim = DIM;
        int64_t n_sample = 0;
        int64_t image_n = 0;
        int64_t image_n_limit = 1000;
        outfile.write(reinterpret_cast<const char*>(&n_sample), sizeof(int64_t));
        outfile.write(reinterpret_cast<const char*>(&dim), sizeof(int64_t));
        std::unordered_map<num_t, std::vector<std::array<unsigned char, DIM>>> already;
        auto process_func = [&](const process_files::fs::path& filePath) {
            already.clear();
            std::cout <<n_sample<< " Lambda processing file: " << filePath.string() << std::endl;
            for (auto &j : IMAGE::splitImageToBlocks<DIM>(filePath, width, height)) {
                bool flag = false;
                num_t norm = 0;
                for (idx_t k = 0; k < DIM; ++k) {
                    norm += j[k] * j[k];
                }
                if (already.contains(norm)) {
                    for (const auto &k : already[norm]) {
                        if (k == j) {
                            flag = true;
                            break;
                        }
                    }
                }
                if (flag) {
                    continue;
                }
                ++n_sample;
                std::array<float,DIM> tmp_float;
                for(int64_t ii = 0;ii<DIM;++ii) {
                    tmp_float[ii] = j[ii];
                }
                outfile.write(reinterpret_cast<char*>(tmp_float.data()), sizeof(tmp_float));
            }
            if(++image_n >= image_n_limit) {
                return false;
            }
            return true;
        };
        process_files::process_files_in_directory(std::filesystem::path(image_path),process_func);
        outfile.seekp(0);
        outfile.write(reinterpret_cast<const char*>(&n_sample), sizeof(int64_t));
        outfile.close();
    }
    static void convert_MIRFLICKR() {

        constexpr idx_t height = 8;
        constexpr idx_t width = 8;
        constexpr idx_t DIM = height * width * 3;
        auto image_path = source_path +"MIRFLICKR_images";
        auto target_path = father_path + "MIRFLICKR.tensor";
        std::ofstream outfile(target_path, std::ios::binary);
        if (!outfile.is_open()) {
            throw MyException("Cannot open file " + target_path + " for writing.");
        }
        int64_t dim = DIM;
        int64_t n_sample = 0;
        int64_t n_limit = 100e6;
        outfile.write(reinterpret_cast<const char*>(&n_sample), sizeof(int64_t));
        outfile.write(reinterpret_cast<const char*>(&dim), sizeof(int64_t));
        std::unordered_map<num_t, std::vector<std::array<unsigned char, DIM>>> already;
        auto process_func = [&](const process_files::fs::path& filePath) {
            already.clear();
            std::cout <<n_sample<< " Lambda processing file: " << filePath.string() << std::endl;
            for (auto &j : IMAGE::splitImageToBlocks<DIM>(filePath, width, height)) {
                bool flag = false;
                num_t norm = 0;
                for (idx_t k = 0; k < DIM; ++k) {
                    norm += j[k] * j[k];
                }
                if (already.contains(norm)) {
                    for (const auto &k : already[norm]) {
                        if (k == j) {
                            flag = true;
                            break;
                        }
                    }
                }
                if (flag) {
                    continue;
                }
                ++n_sample;
                std::array<float,DIM> tmp_float;
                for(int64_t ii = 0;ii<DIM;++ii) {
                    tmp_float[ii] = j[ii];
                }
                outfile.write(reinterpret_cast<char*>(tmp_float.data()), sizeof(tmp_float));
                if(n_sample > n_limit) {
                    return false;
                }
            }
            return true;
        };
        process_files::process_files_in_directory(std::filesystem::path(image_path),process_func);
        outfile.seekp(0);
        outfile.write(reinterpret_cast<const char*>(&n_sample), sizeof(int64_t));
        outfile.close();
    }
    static void convert_audio() {
        const std::string file = source_path + "Audio.data";
        std::ifstream in(file.c_str(), std::ios::binary);
        if (!in) {
            {throw std::ios_base::failure("Open file failed!");}
        }
        unsigned int header[3] = {};
        in.read(reinterpret_cast<char *>(header), sizeof(header));

        auto target_path = father_path + "Audio.tensor";
        std::ofstream outfile(target_path, std::ios::binary);
        if (!outfile.is_open()) {
            throw StringException("Cannot open file " + target_path + " for writing.");
        }
        int64_t n_sample = header[1];
        int64_t dim = header[2];
        outfile.write(reinterpret_cast<const char*>(&n_sample), sizeof(int64_t));
        outfile.write(reinterpret_cast<const char*>(&dim), sizeof(int64_t));
        if (dim != 192) {
            throw std::runtime_error("Bad dimension number!");
        }
        std::array<float, 192> item;
        for (int i = 0; i < n_sample ; ++i) {
            in.read(reinterpret_cast<char *>(item.data()), sizeof(std::array<float, 192>));
            if (!in) {
                throw std::ios_base::failure("Read file failed!");
            }
            outfile.write(reinterpret_cast<char*>(item.data()), sizeof(item));
        }
        in.close();
        outfile.close();
    }
    static void convert_CIFAR() {

        constexpr idx_t DIM = cifar_100::IMAGE_BYTES;
        const std::string file_path = source_path + "cifar-100-binary/train.bin";
        const std::vector<unsigned char> data = cifar_100::readFile(file_path);
        auto target_path = father_path + "CIFAR.tensor";
        std::ofstream outfile(target_path, std::ios::binary);
        if (!outfile.is_open()) {
            throw MyException("Cannot open file " + target_path + " for writing.");
        }
        int64_t n_sample = cifar_100::NUM_IMAGES;
        int64_t dim = DIM;
        outfile.write(reinterpret_cast<const char*>(&n_sample), sizeof(int64_t));
        outfile.write(reinterpret_cast<const char*>(&dim), sizeof(int64_t));
        int index = 0;
        for (int i = 0; i < n_sample; ++i) {
            int fine_label = data[index++];
            int coarse_label = data[index++];
            std::array<float, DIM> tmp_float;
            for (int j = 0; j < DIM; ++j) {
                tmp_float[j] = data[index++];
            }
            outfile.write(reinterpret_cast<char*>(tmp_float.data()), sizeof(tmp_float));
        }
        outfile.close();
    }
    static void convert_SIFT1B() {
        constexpr idx_t DIM = 128;
        // std::ifstream file(source_path + "1milliard.p1.siftbin", std::ios::binary);
        std::fstream file(source_path + "1milliard.p1.siftbin", std::ios::in | std::ios::out | std::ios::binary);
        if (!file) {
            throw MyException("Cannot open file" );
        }
        file.seekg(0, std::ios::end);
        idx_t fileSize = file.tellg();
        file.seekg(0, std::ios::beg);
        int64_t n_sample = fileSize / ( sizeof(int) + DIM * sizeof(unsigned char));
        int64_t dim = DIM;
        auto target_path = father_path + "SIFT1B.tensor";
        std::ofstream outfile(target_path, std::ios::binary);
        if (!outfile.is_open()) {
            throw MyException("Cannot open file " + target_path + " for writing.");
        }
        outfile.write(reinterpret_cast<const char*>(&n_sample), sizeof(int64_t));
        outfile.write(reinterpret_cast<const char*>(&dim), sizeof(int64_t));
        int64_t batch_size = 1000;
        std::vector<unsigned char> buffer(DIM * batch_size);
        std::vector<std::array<float, DIM>> tmp_float_batch(batch_size);
        progress_display display(n_sample);

        for (idx_t i = 0; i < n_sample; i += batch_size) {
            idx_t current_batch_size = std::min(batch_size, n_sample - i);
            file.read(reinterpret_cast<char*>(buffer.data()), sizeof(int));
            file.read(reinterpret_cast<char*>(buffer.data()), current_batch_size * DIM * sizeof(unsigned char));
            if (!file) {
                throw MyException("Error reading file at descriptor");
            }
            for (idx_t b = 0; b < current_batch_size; ++b) {
                for (int j = 0; j < DIM; ++j) {
                    tmp_float_batch[b][j] = buffer[b * DIM + j];
                }
            }
            outfile.write(reinterpret_cast<char*>(tmp_float_batch.data()), current_batch_size * sizeof(std::array<float, DIM>));
            display += current_batch_size;
        }
        outfile.close();
    }
    static void convert_DEEP1M() {
        constexpr idx_t DIM  = 256;
        std::array<float,DIM> vec;
        std::ifstream file(source_path + "deep1M/deep1M_base.fvecs", std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file");
        }
        file.seekg(0, std::ios::end);
        idx_t fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        int64_t n_sample = fileSize / (sizeof(vec)+sizeof(int));
        int64_t dim = DIM;

        auto target_path = father_path + "DEEP1M.tensor";
        std::ofstream outfile(target_path, std::ios::binary);
        if (!outfile.is_open()) {
            throw MyException("Cannot open file " + target_path + " for writing.");
        }
        outfile.write(reinterpret_cast<const char*>(&n_sample), sizeof(int64_t));
        outfile.write(reinterpret_cast<const char*>(&dim), sizeof(int64_t));
        progress_display display(n_sample);
        for(idx_t i = 0;i<n_sample;++i) {
            file.read(reinterpret_cast<char *>(&dim), sizeof(int));
            if(dim != DIM) {
                throw MyException("bad dim");
            }
            file.read(reinterpret_cast<char *>(vec.data()), dim * sizeof(float));

            if (!file) {
                throw MyException("read the end ??");
            }
            outfile.write(reinterpret_cast<char*>(vec.data()), dim * sizeof(float));
            ++display;
        }
        outfile.close();
    }
    static void convert_SIFT1M() {
        constexpr idx_t DIM  = 128;
        std::array<float,DIM> vec;
        std::ifstream file(source_path + "sift_1m/sift_base.fvecs", std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file");
        }
        file.seekg(0, std::ios::end);
        idx_t fileSize = file.tellg();
        file.seekg(0, std::ios::beg);
        int64_t n_sample = fileSize / (sizeof(vec) + sizeof(int));
        int64_t dim = DIM;
        auto target_path = father_path + "SIFT1M.tensor";
        std::ofstream outfile(target_path, std::ios::binary);
        if (!outfile.is_open()) {
            throw MyException("Cannot open file " + target_path + " for writing.");
        }
        outfile.write(reinterpret_cast<const char*>(&n_sample), sizeof(int64_t));
        outfile.write(reinterpret_cast<const char*>(&dim), sizeof(int64_t));
        progress_display display(n_sample);
        for(idx_t i = 0;i<n_sample;++i) {
            file.read(reinterpret_cast<char *>(&dim), sizeof(int));
            if(dim != DIM) {
                std::cout <<dim<< std::endl;
                throw MyException("bad dim");
            }
            file.read(reinterpret_cast<char *>(vec.data()), dim * sizeof(float));

            if (!file) {
                throw MyException("read the end ??");
            }
            outfile.write(reinterpret_cast<char*>(vec.data()), dim * sizeof(float));
            ++display;
        }
        outfile.close();
    }
    static void convert_MNIST() {
        auto target_path = father_path + "MNIST.tensor";
        std::ofstream outfile(target_path, std::ios::binary);
        if (!outfile.is_open()) {
            throw MyException("Cannot open file " + target_path + " for writing.");
        }

        auto data_path = source_path + "MNIST/train-images.idx3-ubyte";
        if (std::ifstream file(data_path, std::ios::binary); file.is_open()) {
            int magic_number = 0;
            int number_of_images = 0;
            int n_rows = 0;
            int n_cols = 0;
            file.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number));
            file.read(reinterpret_cast<char *>(&number_of_images), sizeof(number_of_images));
            file.read(reinterpret_cast<char *>(&n_rows), sizeof(n_rows));
            file.read(reinterpret_cast<char *>(&n_cols), sizeof(n_cols));
            magic_number = MNIST::ReverseInt(magic_number);
            number_of_images = MNIST::ReverseInt(number_of_images);
            n_rows = MNIST::ReverseInt(n_rows);
            n_cols = MNIST::ReverseInt(n_cols);

            int64_t dim = 784;
            int64_t n_sample = number_of_images;

            outfile.write(reinterpret_cast<const char*>(&n_sample), sizeof(int64_t));
            outfile.write(reinterpret_cast<const char*>(&dim), sizeof(int64_t));
            for (int i = 0; i < n_sample; i++) {
                std::vector<float> tp;
                for (int r = 0; r < n_rows; r++) {
                    for (int c = 0; c < n_cols; c++) {
                        unsigned char image = 0;
                        file.read(reinterpret_cast<char *>(&image), sizeof(image));
                        tp.push_back(image);
                    }
                }
                if(tp.size() != dim) {
                    throw MyException("bad dim !");
                }
                outfile.write(reinterpret_cast<char*>(tp.data()), sizeof(float)*dim);
            }
        }
        outfile.close();
    }
};

