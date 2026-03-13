#pragma once
#include <vector>
#include <string>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <ctime>


std::vector<float> readBinaryFile(const std::string& filename);
void createBinaryFile(const std::string& filename, const std::vector<float>& data);
std::vector<float> generateRandomVector(int size);
void checkValidInput(const std::vector<float>& a, const std::vector<float>& b);
void printResult(const std::vector<float>& a, const std::vector<float>& b,const std::vector<float>& c);
void printResultCrossB(const std::vector<float>& a, const std::vector<float>& b,const std::vector<float>& c,unsigned int n);
void writeResultToFile(const std::vector<float>& c);