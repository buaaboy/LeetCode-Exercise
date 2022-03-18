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
    vector<string>obj {"w","wo","wor","worl", "world"};
    vector<vector<int>>obj2{{0,0,1,1,0,0},{0,0,0,0,1,1},{1,0,1,1,0,0},{0,0,1,1,0,0},{0,0,0,0,0,0},{0,0,1,0,0,0}};

    cout << solution.shortestPathBinaryMatrix(obj2) << endl;
    return 0;
}

