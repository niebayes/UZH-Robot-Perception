#include <algorithm>
#include <cassert>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

//@ref https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle

// Fisher–Yates_shuffle
std::vector<int> FisherYatesShuffle(std::size_t size, std::size_t max_size,
                                    std::mt19937& gen) {
  assert(size < max_size);
  std::vector<int> b(size);

  for (std::size_t i = 0; i != max_size; ++i) {
    std::uniform_int_distribution<> dis(0, i);
    std::size_t j = dis(gen);
    if (j < b.size()) {
      if (i < j) {
        b[i] = b[j];
      }
      b[j] = i;
    }
  }
  return b;
}

int main() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::vector<int> b = FisherYatesShuffle(10, 100, gen);

  std::copy(b.begin(), b.end(), std::ostream_iterator<int>(std::cout, " "));
  return 0;
}
