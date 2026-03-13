#pragma once
#include <vector>
#include <string>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <ctime>


/**
 * @brief Reads a binary file containing 32-bit floating point numbers.
 * @param filename The path to the binary file
 * @return std::vector<float> A vector containing the float data;
 */
std::vector<float> readBinaryFile(const std::string& filename);

/**
 * @brief Creates a new binary file and writes the contents of a float vector to it.
 * @param filename The destination path for the binary file.
 * @param data The vector of floats to be written to the file.
 */
void createBinaryFile(const std::string& filename, const std::vector<float>& data);

/**
 * @brief Generates a vector of a specific size filled with random floating point numbers.
 * @param size The number of elements to generate.
 * @return std::vector<float> A vector containing with random values between 0 and RAND_MAX.
 */
std::vector<float> generateRandomVector(int size);

/**
 * @brief Validates input vectors before GPU processing.(check vectors are same size)
 * @param a The first input vector.
 * @param b The second input vector.
 */
void checkValidInput(const std::vector<float>& a, const std::vector<float>& b);

/**
 * @brief Prints the elements of the input and result vectors to the console.
 * * @param a The first input vector.
 * @param b The second input vector.
 * @param c The result vector (a + b).
 */
void printResult(const std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& c);

/**
 * @brief Prints results specifically for operations involving an 'n' stride or cross-vector logic.
 * * @param a The first input vector.
 * @param b The second input vector.
 * @param c The result vector.
 * @param n The stride or offset value used during the GPU calculation.
 */
void printResultCrossB(const std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& c, unsigned int n);

/**
 * @brief Writes the final processed result vector to a default or specified output file.
 * * @param c The result vector to be saved to disk.
 */
void writeResultToFile(const std::vector<float>& c);