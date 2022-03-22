//
// Created by 刘博一 on 2/19/22.
//

#ifndef LEETCODE_SOLUTION_H
#define LEETCODE_SOLUTION_H


#include <vector>
using namespace std;

class Solution {
public:
    int DEBUG;

    vector<int> twoSum(vector<int>& nums, int target);

    void testhello();

    int search(vector<int>& nums, int target);

    vector<int> sortedSquares(vector<int>& nums);

    void rotate(vector<int>& nums, int k);

    string reverseOnlyLetters(string s);

    void moveZeroes(vector<int>& nums); // LeetCode 283

    vector<int> twoSum2(vector<int>& numbers, int target); // LeetCode 167

    int tribonacci(int n); // LeetCode 1137

    int rob(vector<int>& nums); // LeetCode 198

    int rob2(vector<int>& nums); // LeetCode 213

    int minCostClimbingStairs(vector<int>& cost); // LeetCode 746

    int deleteAndEarn(vector<int>& nums); // LeetCode 740

    vector<int> findBall(vector<vector<int>>& grid); // LeetCode 1706

    string reverseWords(string s); // LeetCode 557

    void reverseString(vector<char>& s); // LeetCode 344

    bool canJump(vector<int>& nums); // LeetCode 55

    int jump(vector<int>& nums); // LeetCode 45

    bool containsDuplicate(vector<int>& nums); // LeetCode 217

    int maxSubArray(vector<int>& nums); // LeetCode 53

    string complexNumberMultiply(string num1, string num2); // LeetCode 537

    int maximumDifference(vector<int>& nums); // LeetCode 2016

    string optimalDivision(vector<int>& nums); // LeetCode 553

    int lengthOfLongestSubstring(string s); // LeetCode 3

    bool checkInclusion(string s1, string s2); // LeetCode 567

    int maxSubarraySumCircular(vector<int>& nums); // LeetCode 918

    vector<vector<int>> floodFill(vector<vector<int>>& image, int sr, int sc, int newColor); // LeetCode 733

    int maxAreaOfIsland(vector<vector<int>>& grid); // LeetCode 695

    int getMaxLen(vector<int>& nums); // LeetCode 1567

    int maxScoreSightseeingPair(vector<int>& values); // LeetCode 1014

    int maxProfit2(vector<int>& prices); // LeetCode 122

    int maxProfitCold(vector<int>& prices); // LeetCode 309

    vector<vector<int>> updateMatrix(vector<vector<int>>& mat); // LeetCode 542

    int orangesRotting(vector<vector<int>>& grid); // LeetCode 994

    int maxProfitWithFee(vector<int>& prices, int fee); // LeetCode 714

    bool wordBreak(string s, vector<string>& wordDict); // LeetCode 139

    int trap(vector<int>& height); // LeetCode 42

    int addDigits(int num); // LeetCode 258

    vector<vector<int>> combine(int n, int k); // LeetCode 77

    vector<string> letterCasePermutation(string s); // LeetCode 784

    long long subArrayRanges(vector<int>& nums); // LeetCode 2104

    int numberOfArithmeticSlices(vector<int>& nums); // LeetCode 413

    int minimumTotal(vector<vector<int>>& triangle); // LeetCode 120

    int numDecodings(string s); // LeetCode 16

    int nthUglyNumber2(int n); // LeetCode 264

    int numTrees(int n); // LeetCode 96

    vector<int> getRow(int rowIndex); // LeetCode 119

    int minFallingPathSum(vector<vector<int>>& matrix); // LeetCode 931

    vector<int> searchRange(vector<int>& nums, int target); // LeetCode 34

    int search_reverse(vector<int>& nums, int target); // LeetCode 33

    bool searchMatrix(vector<vector<int>>& matrix, int target); // LeetCode 74

    int findMin(vector<int>& nums); // LeetCode 153

    int peakIndexInMountainArray(vector<int>& arr); // LeetCode 853

    int findPeakElement(vector<int>& nums); // LeetCode 162

    int uniquePaths(int m, int n); // LeetCode 62

    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid); // LeetCode 63

    int minPathSum(vector<vector<int>>& grid); // LeetCode 64

    vector<vector<int>> threeSum(vector<int>& nums); // LeetCode 15

    string longestPalindrome(string s); // LeetCode 5

    bool backspaceCompare(string s, string t); // LeetCode 844

    vector<string> findRestaurant(vector<string>& list1, vector<string>& list2); // LeetCode 599

    vector<vector<int>> intervalIntersection(vector<vector<int>>& firstList, vector<vector<int>>& secondList); // LeetCode 986

    int maxArea(vector<int>& height); // LeetCode 11

    vector<int> findAnagrams(string s, string p); // LeetCode 438

    int numSubarrayProductLessThanK(vector<int>& nums, int k); // LeetCode 713

    int minSubArrayLen(int target, vector<int>& nums); // LeetCode 209

    int numIslands(vector<vector<char>>& grid); // LeetCode 200

    int findCircleNum(vector<vector<int>>& isConnected); // LeetCode 547

    int countMaxOrSubsets(vector<int>& nums); // LeetCode 2044

    bool isPalindrome(int x); // LeetCode 9

    vector<vector<int>> subsets(vector<int>& nums); // LeetCode 78

    string longestWord(vector<string>& words); // LeetCode 720

    int shortestPathBinaryMatrix(vector<vector<int>>& grid); // LeetCode 1091

    vector<vector<int>> subsetsWithDup(vector<int>& nums); // LeetCode 90

    void solve(vector<vector<char>>& board); // LeetCode 130

    vector<vector<int>> permuteUnique(vector<int>& nums); // LeetCode 47
};


#endif //LEETCODE_SOLUTION_H
