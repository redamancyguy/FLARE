#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "dataset.hpp"
#include "TensorStorage.hpp"
#include "WorkLoad.hpp"

int main() {
    // ReadSourceFile::convert_Word2Vec();
    // ReadSourceFile::convert_tiny_images();
    // ReadSourceFile::convert_Gist();
    // ReadSourceFile::convert_audio();
    // ReadSourceFile::convert_goole_earth();
    // ReadSourceFile::convert_CIFAR();
    ReadSourceFile::convert_SIFT1B();
    // ReadSourceFile::convert_DEEP1M();
    // ReadSourceFile::convert_SIFT1M();
    // ReadSourceFile::convert_MIRFLICKR();
    // ReadSourceFile::convert_MNIST();
    auto dataset = TensorStorage::readTensor(father_path+"Word2Vec.tensor");
    std::cout <<dataset.sizes()<<std::endl;
    // dataset = TensorStorage::readTensor(father_path+"tiny_images.tensor");
    // std::cout <<dataset.sizes()<<std::endl;
    // dataset = TensorStorage::readTensor(father_path+"Gist.tensor");
    // std::cout <<dataset.sizes()<<std::endl;
    // dataset = TensorStorage::readTensor(father_path+"MIRFLICKR.tensor");
    // std::cout <<dataset.sizes()<<std::endl;
    // dataset = TensorStorage::readTensor(father_path+"google_earth.tensor");
    // std::cout <<dataset.sizes()<<std::endl;
    // dataset = TensorStorage::readTensor(father_path+"Audio.tensor");
    // std::cout <<dataset.sizes()<<std::endl;
    // dataset = TensorStorage::readTensor(father_path+"MNIST.tensor");
    // std::cout <<dataset.sizes()<<std::endl;
    // dataset = TensorStorage::readTensor(father_path+"CIFAR.tensor");
    // std::cout <<dataset.sizes()<<std::endl;
    // dataset = TensorStorage::readTensor(father_path+"SIFT1B.tensor",100e6);
    // std::cout <<dataset.sizes()<<std::endl;
    // dataset = TensorStorage::readTensor(father_path+"DEEP1M.tensor");
    // std::cout <<dataset.sizes()<<std::endl;
    // dataset = TensorStorage::readTensor(father_path+"SIFT1M.tensor");
    // std::cout <<dataset.sizes()<<std::endl;
    return 0;
}
