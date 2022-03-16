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
    vector<int>obj {2,3,1};
    vector<vector<int>>obj2{{1,1,0},{1,1,0},{0,0,1}};

    cout << solution.findCircleNum(obj2)<< endl;
    return 0;
}

