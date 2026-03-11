#include "../include/utils.h"
#include <fstream>
#include <iostream>

void createBinaryFile(const std::string& filename, const std::vector<float>& data)
{
    std::ofstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Error creating file");
    }

    file.write((char*)data.data(), data.size() * sizeof(float));
    file.close();

    std::cout << "File created: " << filename << std::endl;
}

std::vector<float> readBinaryFile(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary);
    
    if (!file.is_open()) {
        throw std::runtime_error("Error opening file");
    }
    
    file.seekg(0, std::ios::end);
    int size = file.tellg() / sizeof(float);
    file.seekg(0, std::ios::beg);
    
    std::vector<float> data(size);
    file.read((char*)data.data(), size * sizeof(float));

    return data;
}

std::vector<float> generateRandomVector(int size){
    srand(10);
    std::vector<float> randomVector(size);
    for(int i = 0; i < size; i++)
        randomVector[i] = (float)rand() / RAND_MAX * 100;
    return randomVector;
}

void checkValidInput(const std::vector<float>& a, const std::vector<float>& b){
    if(a.size() != b.size())
        throw std::invalid_argument("Error: vectors must be the same size!");
}

void printResult(const std::vector<float>& a, const std::vector<float>& b,const std::vector<float>& c) {
    std::cout << "Result (A + B):" << std::endl;
    for (int i = 0; i < c.size(); i++)
        std::cout << "a[" << i << "] = " << a[i] << ", b[" << i << "] = " << b[i] << ", c[" << i << "] = " << c[i] << std::endl;

}

void writeResultToFile(const std::vector<float>& c){
    std::ofstream file("../result.txt");
    if(!file.is_open())
        throw std::runtime_error("error creating result file");
    
    for(int i = 0; i < c.size(); i++)
        file << "c[" << i << "] = " << c[i] << std::endl;
    file.close();
}