#pragma once

#include <algorithm>
#include <vector>

namespace stats {
//--------------------------------------------------------------------------------------------------
// mean
//--------------------------------------------------------------------------------------------------
template <class T> static T mean(const std::vector<T>& data) {
  return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
}

//--------------------------------------------------------------------------------------------------
// std_dev
//--------------------------------------------------------------------------------------------------
template <class T> static T std_dev(const std::vector<T>& data) {
  const T m = mean(data);
  const T v = variance(data, m);
  return std::sqrt(v);
}

//--------------------------------------------------------------------------------------------------
// variance
//--------------------------------------------------------------------------------------------------
template <class T> 
static T variance(const std::vector<T>& data, T mean) {
  return std::accumulate(data.begin(), data.end(), 0.0,
                         [mean](T accumulator, T val) {
                           return accumulator + (val - mean) * (val - mean);
                         }) / 
                    (data.size() - 1);
}

//--------------------------------------------------------------------------------------------------
// t_value
//--------------------------------------------------------------------------------------------------
template <class T> 
static T t_value(const std::vector<T>& group1, const std::vector<T>& group2) {
  if (group1.size() != group2.size()) {
    throw std::invalid_argument("The two groups must have the same size");
  }

  T mean1 = mean(group1);
  T mean2 = mean(group2);
  T variance1 = variance(group1, mean1);
  T variance2 = variance(group2, mean2);
  
  const size_t size = group1.size();
  T pooledVariance = ((size - 1) * variance1 + (size - 1) * variance2) / (size + size - 2);
  T standardError = sqrt(2 * pooledVariance / size);
  
  return (mean1 - mean2) / standardError;
}

//--------------------------------------------------------------------------------------------------
// median
//--------------------------------------------------------------------------------------------------
template <class T> 
T median(const std::vector<T>& data) {
  std::vector<T> sorted_data = data;

  auto n = sorted_data.size() / 2;
  std::nth_element(sorted_data.begin(), sorted_data.begin() + n, sorted_data.end());

  if (sorted_data.size() % 2 == 0) {
    T max_of_lower_half = *std::max_element(sorted_data.begin(), sorted_data.begin() + n);
    return (max_of_lower_half + sorted_data[n]) / 2;
  } else {
    return sorted_data[n];
  }
}
} // namespace stats
