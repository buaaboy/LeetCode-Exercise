#include <iostream>
#include "algorithm"
#include "string"
#include "sstream"
using namespace std;
#include "Solution.h"


int main() {
    std::cout << "Hello, World!" << std::endl;
    Solution solution;
    solution.testhello();
    vector<int>obj {1,2,2};
    vector<vector<int>>obj2{{1,2},{3},{3},{}};

    cout << solution.allPathsSourceTarget(obj2)[0][0] << endl;
    return 0;
}

