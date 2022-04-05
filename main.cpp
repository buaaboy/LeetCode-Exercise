#include <iostream>
#include "algorithm"
#include "string"
#include "sstream"
using namespace std;
#include "Solution.h"

int lowBit(int x) {
    return x & -x;
}

int main() {
    std::cout << "Hello, World!" << std::endl;
    Solution solution;
    solution.testhello();
    vector<int>obj {1,3};
    vector<int> objj {2};
    vector<vector<int>>obj2{{1,2},{3},{3},{}};

    cout << solution.countPrimeSetBits(6, 10) << endl;
    return 0;
}

