//
// Created by 刘博一 on 2/19/22.
//

#include <iostream>
#include <queue>
#include "Solution.h"
#include "vector"
#include "string"
#include "map"
#include "unordered_set"
#include "set"
using namespace std;

int DEBUG = 0;

// LC 1
vector<int> Solution::twoSum(vector<int> &nums, int target) {
    int n = nums.size();
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (nums[i] + nums[j] == target) {
                return {i, j};
            }
        }
    }
    return {};
}

void Solution::testhello() {
    cout << "123" << endl;
}

int Solution::search(vector<int> &nums, int target) {
    int begin = 0;
    int end = nums.size() - 1;
    int mid = begin + (end - begin) / 2;
    while (begin <= end) {
        mid = begin + (end - begin) / 2;
        if (nums[mid] == target) {
            return mid;
        } else if (nums[mid] > target) {
            end = mid - 1;
        } else {
            begin = mid + 1;
        }
    }
    return -1;
}

vector<int> Solution::sortedSquares(vector<int> &nums) {
    int n = nums.size();
    vector<int> ret(n);
    int pos = n - 1;
    for (int i = 0, j = n - 1; i <= j;) {
        if (abs(nums[i]) > abs(nums[j])) {
            ret[pos] = nums[i] * nums[i];
            i++;
        } else {
            ret[pos] = nums[j] * nums[j];
            j--;
        }
        pos--;
    }
    return ret;
}

void Solution::rotate(vector<int> &nums, int k) {
    // method1
    /*
    int n = nums.size();
    vector<int> newArr(n);
    for (int i = 0; i < n; ++i) {
        newArr[(i + k) % n] = nums[i];
    }
    return nums.assign(newArr.begin(), newArr.end());
     */

    // method2
    int n = nums.size();
    reverse(nums.begin(), nums.end());
    reverse(nums.begin(), nums.begin() + k % n);
    reverse(nums.begin() + k % n + 1, nums.end());
}

// LC 917
string Solution::reverseOnlyLetters(string s) {
    int len = s.size();
    for (int i = 0, j = len - 1; i <= j;) {
        if (isalpha(s[i]) && isalpha(s[j])) {
            swap(s[i], s[j]);
            i++;
            j--;
        } else {
            if (!isalpha(s[i])) {
                i++;
            }
            if (!isalpha(s[j])) {
                j--;
            }
        }
    }
    return s;
}

void Solution::moveZeroes(vector<int> &nums) {
    int n = nums.size(), i=0, j=0;
    for(i=0;i<n;i++) {
        nums[j]=nums[i];
        if(nums[j]) {
            j++;
        }
    }
    for(;j<n;j++) {
        nums[j]=0;
    }
}

vector<int> Solution::twoSum2(vector<int> &numbers, int target) {
    int begin = 0;
    int end = numbers.size() - 1;
    while(true) {
        int sum = numbers[begin] + numbers[end];
        if (sum == target) {
            return vector<int> {begin + 1, end + 1};
        } else if (sum < target) {
            begin++;
        } else {
            end--;
        }
    }
}

vector<vector<long>> multiply(vector<vector<long>>& a, vector<vector<long>>& b) {
    vector<vector<long>> c(3, vector<long>(3));
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    return c;
}

vector<vector<long>> pow(vector<vector<long>>& a, long n) {
    vector<vector<long>> ret = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    while (n > 0) {
        if ((n & 1) == 1) {
            ret = multiply(ret, a);
        }
        n >>= 1;
        a = multiply(a, a);
    }
    return ret;
}

int Solution::tribonacci(int n) {
    if (n == 0) {
        return 0;
    }
    if (n <= 2) {
        return 1;
    }
    vector<vector<long>> q = {{1, 1, 1}, {1, 0, 0}, {0, 1, 0}};
    vector<vector<long>> res = pow(q, n);
    return res[0][2];
}

int Solution::rob(vector<int> &nums) {
    int n = nums.size();
    vector<int>dp(n);
    dp[0]=nums[0];
    if (n > 1) {
        dp[1]= max(nums[0], nums[1]);
        for (int i = 2; i < n; ++i) {
            dp[i] = max(dp[i-1], dp[i-2] + nums[i]);
        }
    }
    return *(dp.end() - 1);
}

int Solution::minCostClimbingStairs(vector<int> &cost) {
    int n = cost.size();
    vector<int>dp(n+1);
    dp[0]=0;
    if (n > 1) {
        dp[1]=0;
        for (int i = 2; i <= n; ++i) {
            dp[i] = min(dp[i-2]+cost[i-2], dp[i-1]+cost[i-1]);
        }
    }
    return *(dp.end() - 1);
}

int Solution::deleteAndEarn(vector<int> &nums) {
    int n = nums.size();
    sort(nums.begin(), nums.end());
    int ans = 0;
    vector<int>sum {nums[0]};

    for (int i = 1; i < n; ++i) {
        if (nums[i] == nums[i-1]) {
            sum.back() += nums[i];
        } else if (nums[i] == nums[i-1] + 1) {
            sum.push_back(nums[i]);
        } else {
            ans += rob(sum);
            sum = {nums[i]};
        }
    }
    ans += rob(sum);
    return ans;
}

int Solution::rob2(vector<int> &nums) {
    int n = nums.size();
    vector<int>dp(n-1); // this ranges (0, n-2)
    vector<int>dp2(n); // this ranges (1, n-1)

    if (n == 1) {
        return nums[0];
    }

    dp[0]=nums[0];
    dp2[0]=0;
    dp[1]= max(nums[0], nums[1]);
    dp2[1] = nums[1];
    for (int i = 2; i < n-1; ++i) {
        dp[i] = max(dp[i-1], dp[i-2] + nums[i]);
    }
    for (int i = 2; i < n; ++i) {
        dp2[i] = max(dp2[i-1], dp2[i-2] + nums[i]);
    }


//    if (Solution::DEBUG) {
//        cout << "dp:" << endl;
//        for (int i = 0; i < n; ++i) {
//            cout << dp[i] << endl;
//        }
//        cout << "dp2:" << endl;
//        for (int i = 0; i < n; ++i) {
//            cout << dp2[i] << endl;
//        }
//    }
    return max(*(dp.end() - 1), *(dp2.end() - 1));
}

int findBallTravel(int column, vector<vector<int>> &grid) {
    int row = 0;
    int cur = column;
    int m = grid.size();
    int n = grid[0].size();
    while (row < m) {
        if (grid[row][cur] == 1) {
            if (cur == n - 1 || grid[row][cur + 1] == -1) {
                return -1;
            } else {
                row++;
                cur++;
            }
        } else {
            if (grid[row][cur] == -1) {
                if (cur == 0 || grid[row][cur - 1] == 1) {
                    return -1;
                } else {
                    row++;
                    cur--;
                }
            }
        }
    }
    return cur;
}

vector<int> Solution::findBall(vector<vector<int>> &grid) {
    int m = grid.size();
    int n = grid[0].size();
    vector<int> ans;

    for (int i = 0; i < n; ++i) {
        ans.push_back(findBallTravel(i, grid));
    }
    return ans;
}

void Solution::reverseString(vector<char> &s) {
    int i=0, j=s.size()-1;
    while(i <= j) {
        swap(s[i], s[j]);
        i++;
        j--;
    }
}

string Solution::reverseWords(string s) {
    int end = 0;
    int len = s.size();
    int start = 0;
    while(end < len) {
        while (s[end] == ' ') {
            start++;
            end++;
        }
        while (end < len && s[end] != ' ') {
            end++;
        }
        int i=start, j=end - 1;
        while (i <= j) {
            swap(s[i], s[j]);
            i++;
            j--;
        }
        start = end;
    }
    return s;
}

bool Solution::canJump(vector<int> &nums) {
    // method1 dp
    /*
    int n = nums.size(), sum;
    vector<int>dp(n);
    for (int i = 0; i < n; ++i) {
        sum = 0;
        if (i==0) {
            sum = 1;
        }
        for (int j = 0; j < i; ++j) {
            sum += dp[j] * (nums[j] + j >= i) ? 1 : 0;
        }
        dp[i] = sum > 0 ? 1 : 0;
    }
    return (dp[n-1] != 0);
     */
    //method2 greedy
    int maxWay = 0;
    int n = nums.size();
    for (int i = 0; i < n-1; ++i) {
        if (i > maxWay) {
            break;
        }
        maxWay = max(maxWay, i + nums[i]);
    }
    return maxWay >= (n -1);
}

int Solution::jump(vector<int> &nums) {
    // method1 dp
    /*
    int n = nums.size(), sum;
    vector<int>dp(n);
    for (int i = 0; i < n; ++i) {
        dp[i] = 0x7fffffff;
        if (i == 0) dp[i] = 0;
        for (int j = 0; j < i; ++j) {
            if (dp[j] != 0x7fffffff && nums[j] + j >= i) {
                dp[i] = min(dp[i], dp[j] + 1);
            }
        }
    }
    return dp[n-1];
     */
    // method2 greedy
    int maxWay = 0;
    int n = nums.size();
    int sum = 0;
    int endLocation = 0;
    for (int i = 0; i < n-1; ++i) {
        maxWay = max(maxWay, i + nums[i]);
        if (i == endLocation) {
            sum++;
            endLocation = maxWay;
        }
    }
    return sum;
}

bool Solution::containsDuplicate(vector<int> &nums) {
    unordered_set<int> hashset;
    int n = nums.size();
    for (int i = 0; i < n; ++i) {
        if (hashset.find(nums[i]) != hashset.end()) {
            return true;
        }
        hashset.insert(nums[i]);
    }
    return false;
}

int Solution::maxSubArray(vector<int> &nums) {
    return 0;
}

string Solution::complexNumberMultiply(string num1, string num2) {
    struct complex {
        int realPart;
        int virtualPart;
    };

    int index1 = num1.find("+");
    int index2 = num2.find("+");

    complex complex1 = {
            stoi(num1.substr(0, index1)),
            stoi(num1.substr(index1 + 1, num1.size() - index1 - 2))
    };

    complex complex2 = {
            stoi(num2.substr(0, index2)),
            stoi(num2.substr(index2 + 1, num2.size() - index2 - 2))
    };

    return to_string(complex1.realPart * complex2.realPart - complex1.virtualPart * complex2.virtualPart) + "+" +
            to_string(complex1.realPart * complex2.virtualPart + complex1.virtualPart * complex2.realPart) + "i";
}

int Solution::maximumDifference(vector<int> &nums) {
    int minimum = 0x7fffffff;
    int n = nums.size();
    int maxSub = -1;
    for (int i = 0; i < n; ++i) {
        minimum = min(minimum, nums[i]);
        if (nums[i] > minimum) {
            maxSub = max(nums[i] - minimum, maxSub);
        }
    }
    return maxSub;
}


string Solution::optimalDivision(vector<int> &nums) {
    int length = nums.size();
    if (length == 1) {
        return to_string(nums[0]);
    }
    if (length == 2) {
        return to_string(nums[0]) + "/" + to_string(nums[1]);
    }
    string temp = "";
    temp += to_string(nums[0]);
    temp += "/(";
    for (int i = 1; i < length - 1; ++i) {
        temp += to_string(nums[i]);
        temp += "/";
    }
    temp += to_string(nums[length - 1]);
    temp += ")";
    return temp;
}

int Solution::lengthOfLongestSubstring(string s) {
    set<char> charset;
    int begin = 0;
    int end = 0;
    int maxlength = 0;
    while(end < s.size()) {
        if (charset.find(s[end]) == charset.end()) {
            charset.insert(s[end]);
        } else {
            while(s[begin] != s[end]) {
                charset.erase(s[begin]);
                begin++;
            }
            begin++;
        }
        maxlength = max(maxlength, end - begin + 1);
        end++;
    }
    return maxlength;
}

bool Solution::checkInclusion(string s1, string s2) {
    int length1 = s1.size();
    int length2 = s2.size();
    vector<int> count1(26), count2(26);
    if (length1 > length2) {
        return false;
    }
    for (int i = 0; i < length1; ++i) {
        count1[s1[i] - 'a']++;
        count2[s2[i] - 'a']++;
    }
    if (count1 == count2) return true;
    for (int i = length1; i < length2; ++i) {
        count2[s2[i] - 'a']++;
        count2[s2[i - length1] - 'a']--;
        if (count1 == count2) return true;
    }
    return false;
}

int Solution::maxSubarraySumCircular(vector<int> &nums) {
    int total = 0, maxSum = nums[0], curMax = 0, minSum = nums[0], curMin = 0;
    for(int a: nums) {
        curMax = max(curMax + a, a);
        maxSum = max(maxSum, curMax);
        curMin = min(curMin + a, a);
        minSum = min(minSum, curMin);
        total += a;
    }
    return maxSum > 0 ? max(maxSum, total - minSum) : maxSum;
}

vector<vector<int>> Solution::floodFill(vector<vector<int>> &image, int sr, int sc, int newColor) {
    const int dx[4] = {1, 0, 0, -1};
    const int dy[4] = {0, 1, -1, 0};
    int currColor = image[sr][sc];
    if (currColor == newColor) return image;
    int n = image.size(), m = image[0].size();
    queue<pair<int, int>> que;
    que.emplace(sr, sc);
    image[sr][sc] = newColor;
    while (!que.empty()) {
        int x = que.front().first, y = que.front().second;
        que.pop();
        for (int i = 0; i < 4; i++) {
            int mx = x + dx[i], my = y + dy[i];
            if (mx >= 0 && mx < n && my >= 0 && my < m && image[mx][my] == currColor) {
                que.emplace(mx, my);
                image[mx][my] = newColor;
            }
        }
    }
    return image;
}

int Solution::maxAreaOfIsland(vector<vector<int>> &grid) {
    int m = grid.size();
    int n = grid[0].size();
    int maxScale = 0;
    int curMax = 0;
    const int dx[4] = {1, 0, 0, -1};
    const int dy[4] = {0, 1, -1, 0};
    queue<pair<int, int>> que;
    vector<vector<int>> marked(m, vector<int>(n, 0));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (!marked[i][j] && grid[i][j] == 1) {
                curMax = 0;
                que.emplace(i, j);
                marked[i][j] = 1;
                while (!que.empty()) {
                    int x = que.front().first, y = que.front().second;
                    que.pop();
                    curMax++;
//                    marked[x][y] = 1;
                    for (int k = 0; k < 4; k++) {
                        int mx = x + dx[k], my = y + dy[k];
                        if (mx >= 0 && mx < m && my >= 0 && my < n && grid[mx][my] == 1 && marked[mx][my] == 0) {
                            que.emplace(mx, my);
                            marked[mx][my] = 1;
                        }
                    }
                }
                maxScale = max(maxScale, curMax);
            }
        }

    }
    return maxScale;
}

int Solution::getMaxLen(vector<int> &nums) {
    int n = nums.size();
    vector<int> dp_positive(n);
    vector<int> dp_negative(n);
    if (nums[0] > 0) {
        dp_positive[0] = 1;
    } else if (nums[0] < 0) {
        dp_negative[0] = 1;
    }
    for (int i = 1; i < n; ++i) {
        if (nums[i] == 0) {
            dp_positive[i] = 0;
            dp_negative[i] = 0;
        } else if (nums[i] > 0) {
            dp_positive[i] = dp_positive[i-1] + 1;

            dp_negative[i] = dp_negative[i-1] != 0 ? dp_negative[i-1] + 1 : 0;
        } else {
            dp_positive[i] = dp_negative[i-1] != 0 ? dp_negative[i-1] + 1 : 0;
            dp_negative[i] = dp_positive[i-1] + 1;
        }
    }
    return *max_element(dp_positive.begin(), dp_positive.end());
}

int Solution::maxScoreSightseeingPair(vector<int> &values) {
    int length = values.size();
    int ans = 0;
    int maxValue = values[0];
    for (int i = 1; i < length; ++i) {
        ans = max(ans, maxValue + values[i] - i);
        maxValue = max(maxValue, values[i] + i);
    }
    return ans;
}

int Solution::maxProfit2(vector<int> &prices) {
    // solution1 dp
    /*
    int n = prices.size();
    vector<int> own(n);
    vector<int> unown(n);
    own[0] = -prices[0];
    unown[0] = 0;
    for (int i = 1; i < n; ++i) {
        own[i] = max(own[i-1], unown[i-1] - prices[i]);
        unown[i] = max(unown[i-1], own[i-1] + prices[i]);
    }
    return unown[n-1];
     */
    // solution2 greedy
    int n = prices.size();
    int profit = 0;
    for (int i = 1; i < n; ++i) {
        profit += (prices[i] > prices[i-1]) ? (prices[i] - prices[i-1]) : 0;
    }
    return profit;
}

int Solution::maxProfitCold(vector<int> &prices) {
    int n = prices.size();
//    vector<int> own_cold(n);
    vector<int> unown_cold(n);
    vector<int> own_uncold(n);
    vector<int> unown_uncold(n);
    own_uncold[0] = -prices[0];
    unown_cold[0] = unown_uncold[0] = 0;
    for (int i = 1; i < n; ++i) {
        unown_cold[i] = own_uncold[i-1] + prices[i];
        own_uncold[i] = max(own_uncold[i-1], unown_uncold[i-1] - prices[i]);
        unown_uncold[i] = max(unown_cold[i-1], unown_uncold[i-1]);
    }
    return max(unown_uncold[n-1], unown_cold[n-1]);
}

vector<vector<int>> Solution::updateMatrix(vector<vector<int>> &mat) {
    int m = mat.size(), n = mat[0].size();
    vector<vector<int>> direction = {{1,0}, {-1,0}, {0,-1}, {0,1}};
    vector<vector<int>> arrived(m, vector<int>(n));
    vector<vector<int>> ans(m, vector<int>(n));

    queue<pair<int, int>> q;

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (mat[i][j] == 0) {
                arrived[i][j] = 1;
                ans[i][j] = 0;
                q.emplace(i,j);
            }
        }
    }

    int i, j;
    while (!q.empty()) {
        i = q.front().first;
        j = q.front().second;
        q.pop();
        for (int k = 0; k < 4; ++k) {
            int m1 = i + direction[k][0];
            int n1 = j + direction[k][1];
            if (m1 >= 0 && m1 < m && n1 >= 0 && n1 < n && arrived[m1][n1] == 0) {
                ans[m1][n1] = ans[i][j] + 1;
                arrived[m1][n1] = 1;
                q.emplace(m1, n1);
            }
        }
    }
    return ans;
}

int Solution::orangesRotting(vector<vector<int>> &grid) {
    int m = grid.size(), n = grid[0].size();
    vector<vector<int>> direction = {{1,0}, {-1,0}, {0,-1}, {0,1}};
    vector<vector<int>> arrived(m, vector<int>(n));

    queue<pair<int, int>> orange;
    int sum = 0;

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (grid[i][j] == 2) {
                orange.emplace(i, j);
                arrived[i][j] = 1;
            }
            if (grid[i][j] == 1) sum++;
        }
    }

    if (sum == 0) {
        return 0;
    }

    int i, j, time=0;
    while(!orange.empty()) {

        int length = orange.size();
        time++;
        for (int l = 0; l < length; ++l) {
            i = orange.front().first;
            j = orange.front().second;
            orange.pop();
            for (int k = 0; k < 4; ++k) {
                int m1 = i + direction[k][0];
                int n1 = j + direction[k][1];

                if (m1 >= 0 && m1 < m && n1 >= 0 && n1 < n && grid[m1][n1] == 1 && arrived[m1][n1] == 0) {
                    orange.emplace(m1, n1);
                    arrived[m1][n1] = 1;
                    sum--;
                }
            }
        }
    }

    return sum != 0 ? -1 : time-1;
}

int Solution::maxProfitWithFee(vector<int> &prices, int fee) {
    int n = prices.size();
    vector<int> own(n);
    vector<int> unown(n);
    own[0] = -prices[0] - fee;
    unown[0] = 0;
    for (int i = 1; i < n; ++i) {
        own[i] = max(own[i-1], unown[i-1] - prices[i] - fee);
        unown[i] = max(unown[i-1], own[i-1] + prices[i]);
    }
    return max(own[n-1], unown[n-1]);
}

bool Solution::wordBreak(string s, vector<string> &wordDict) {
    int length = s.size();
    unordered_set<string>wordSet(wordDict.size());
    for(string str: wordDict) {
        wordSet.insert(str);
    }

    vector<int> canMake(length+1);
    canMake[0] = 1;
    for (int i = 1; i <= length; ++i) {
        for (int j = 0; j < i; ++j) {
            if (canMake[j] == 1 && wordSet.find(s.substr(j, i-j)) != wordSet.end()) {
                canMake[i] = 1;
                break;
            }
        }
    }
    return canMake[length];
}

int Solution::trap(vector<int> &height) {
    int n = height.size();
    int leftMax = 0;
    int rightMax = 0;
    int left = 0;
    int right = n - 1;
    int ans = 0;
    while(left < right) {
        leftMax = max(leftMax, height[left]);
        rightMax = max(rightMax, height[right]);
        if (height[left] < height[right]) {
            ans += leftMax - height[left];
            left++;
        } else {
            ans += rightMax - height[right];
            right--;
        }
    }
    return ans;
}

int Solution::addDigits(int num) {
    return (num - 1) % 9 + 1;
}

vector<vector<int>> combine_ans;
vector<int> combine_temp;

void combine_dfs(int cur, int n, int k) {
    if (combine_temp.size() + (n - cur + 1) < k) {
        return;
    } // prune
    if (combine_temp.size() == k) {
        combine_ans.push_back(combine_temp);
        return;
    }
    combine_temp.push_back(cur);
    combine_dfs(cur + 1, n, k);
    combine_temp.pop_back(); // take into account

    combine_dfs(cur+1, n, k);
}

vector<vector<int>> Solution::combine(int n, int k) {
    combine_dfs(1, n, k);
    return combine_ans;
}

vector<string> letterCasePermutation_ans;
vector<int> letterCasePermutation_location;
string letterCasePermutation_temp;

void letterCasePermutation_dfs(int layer, int n) {
    if (layer == n) {
        letterCasePermutation_ans.push_back(letterCasePermutation_temp);
        return;
    }
    letterCasePermutation_temp[letterCasePermutation_location[layer]] -= 32;
    letterCasePermutation_dfs(layer + 1, n);
    letterCasePermutation_temp[letterCasePermutation_location[layer]] += 32;

    letterCasePermutation_dfs(layer + 1, n);
}

vector<string> Solution::letterCasePermutation(string s) {
    for (int i = 0; i < s.size(); ++i) {
        if (isalpha(s[i])) {
            letterCasePermutation_location.push_back(i);
            if(s[i] >= 'A' && s[i] <= 'Z') {
                s[i] += 32;
            }
        }
    }
    letterCasePermutation_temp = s;
    letterCasePermutation_dfs(0, letterCasePermutation_location.size());
    return letterCasePermutation_ans;
}

long long Solution::subArrayRanges(vector<int> &nums) {
    return 0;
}

int Solution::numberOfArithmeticSlices(vector<int> &nums) {
    int n = nums.size();
    int cur = 0;
    int sum = 0;
    for (int i = 0; i < n; ++i) {
        if (i == 0 || i == 1) {
            cur++;
        } else {
            if (2 * nums[i-1] == nums[i] + nums[i-2]) {
                cur = (cur < 3) ? 3 : cur + 1;
                sum += (cur - 2);
            } else {
                cur = 1;
            }
        }
    }
    return sum;
}

int Solution::minimumTotal(vector<vector<int>> &triangle) {
    // dp[i][j]=min(dp[i-1][j], dp[i-1][j-1])+triangle[i][j]
    int n = triangle.size();
    vector<int> ans(n);
    ans[0] = triangle[0][0];
    for (int i = 1; i < n; ++i) {
        for (int j = i; j >= 0; --j) {
            if (j == 0) {
                ans[j] += triangle[i][j];
            } else if (j == i) {
                ans[j] = ans[j-1] + triangle[i][j];
            } else {
                ans[j] = min(ans[j-1], ans[j]) + triangle[i][j];
            }
        }
    }
    return *min_element(ans.begin(), ans.end());
}

int Solution::numDecodings(string s) {
    /*
     * 226
     * 2 2 6
     * 2 26
     * 22 6
     * dp[i] = dp[i-1] + dp[i-2]
     */
    int n = s.size();
    vector<int> dp(n+1);
    dp[0] = s[0] == '0' ? 0 : 1;
    for (int i = 1; i <= n; ++i) {
        if (s[i-1] != '0') {
            dp[i] += dp[i-1];
        }
        if (i > 1 && s[i-2] != '0' && stoi(s.substr(i-2, 2)) <= 26) {
            dp[i] += dp[i-2];
        }
    }
    return dp[n];
}

int Solution::nthUglyNumber2(int n) {
    vector<int> dp(n + 1);
    dp[1] = 1;
    int p2 = 1, p3 = 1, p5 = 1;
    for (int i = 2; i <= n; i++) {
        int num2 = dp[p2] * 2, num3 = dp[p3] * 3, num5 = dp[p5] * 5;
        dp[i] = min(min(num2, num3), num5);
        if (dp[i] == num2) {
            p2++;
        }
        if (dp[i] == num3) {
            p3++;
        }
        if (dp[i] == num5) {
            p5++;
        }
    }
    return dp[n];
}

// LC 96
int Solution::numTrees(int n) {
    // [1...i-1] [i] [i+1...n]
    vector<int> dp(n+1);
    dp[0] = 1;
    dp[1] = 1;
    for (int i = 2; i <= n; ++i) {
        for (int j = 1; j <= i; ++j) {
            dp[i] += dp[j - 1] * dp[i - j];
        }
    }
    return dp[n];
}

// LC 119
vector<int> Solution::getRow(int rowIndex) {
    vector<int> row(rowIndex + 1);
    row[0] = 1;
    for (int i = 1; i <= rowIndex; ++i) {
        row[i] = 1LL * row[i - 1] * (rowIndex - i + 1) / i;
    }
    return row;
}

// LC 931
int Solution::minFallingPathSum(vector<vector<int>> &matrix) {
    int n = matrix.size();
    vector<vector<int>> dp(n, vector<int>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == 0) {
                dp[0][j] = matrix[0][j];
            } else {
                dp[i][j] = matrix[i][j];
                if (j == 0) {
                    dp[i][j] += min(dp[i-1][j], dp[i-1][j+1]);
                } else if (j == n-1) {
                    dp[i][j] += min(dp[i-1][j-1], dp[i-1][j]);
                } else {
                    dp[i][j] += min(dp[i-1][j-1], min(dp[i-1][j], dp[i-1][j+1]));
                }
            }
        }
    }
    return *min_element(dp[n-1].begin(), dp[n-1].end());
}

vector<int> Solution::searchRange(vector<int> &nums, int target) {
    // nums[mid] < target -> [mid+1, end]
    // [begin, mid]
    int n = nums.size();
    vector<int> ans;
    int begin = 0, end = n - 1;
    if (nums.empty()) {
        ans.push_back(-1);
        ans.push_back(-1);
        return ans;
    }
    while(begin < end) {
        int mid = begin + (end - begin) / 2;
        if (nums[mid] < target) {
            begin = mid + 1;
        } else {
            end = mid;
        }
    }

    ans.push_back(nums[end] == target ? end : -1);
    // nums[mid] > target -> [begin, mid-1]
    begin = 0;
    end = n - 1;
    while(begin < end) {
        int mid = begin + (end - begin + 1) / 2;
        if (nums[mid] > target) {
            end = mid - 1;
        } else {
            begin = mid;
        }
    }
    ans.push_back(nums[begin] == target ? begin : -1);
    return ans;
}

int Solution::search_reverse(vector<int> &nums, int target) {
    // 4 5 6 1 2 3
    // nums[mid] > nums[0] --> whether in normal order
    // nums[mid] > target --> which part
    int n = nums.size();
    int begin = 0, end = n-1;
    if (nums.empty()) {
        return -1;
    }
    if (nums.size() == 1) {
        return (nums[0] == target) ? 0 : -1;
    }
    while(begin < end) {
        int mid = begin + (end - begin) / 2;
        if (nums[mid] == target) {
            return mid;
        }

        if (nums[0] == target) {
            return 0;
        }
        if (nums[n-1] == target) {
            return n-1;
        }
        if(nums[mid] > nums[0]) {
            if (nums[mid] > target && target > nums[0]) {
                end = mid;
            } else {
                begin = mid + 1;
            }
        } else {
            if (nums[n-1] > target && target > nums[mid]) {
                begin = mid + 1;
            } else {
                end = mid;
            }
        }
    }
    if (nums[begin] == target) return begin;
    if (nums[end] == target) return end;
    return -1;
}

bool Solution::searchMatrix(vector<vector<int>> &matrix, int target) {
    int m = matrix.size();
    int n = matrix[0].size();
    int begin = 0;
    int end = m - 1;
    int targetRow = 0;
    while(begin < end) {
        int mid = begin + ((end - begin + 1) >> 1);
        if (matrix[mid][0] == target) {
            targetRow = mid;
            break;
        } else if(matrix[mid][0] > target) {
            end = mid - 1;
            targetRow = end;
        } else {
            begin = mid;
            targetRow = begin;
        }
    }

    begin = 0;
    end = n - 1;

    while(begin < end) {
        int mid = begin + ((end - begin) >> 1);
        if(matrix[targetRow][mid] == target) {
            return true;
        } else if(matrix[targetRow][mid] > target) {
            end = mid;
        } else {
            begin = mid + 1;
        }
    }
    return (matrix[targetRow][end] == target);
}

int Solution::findMin(vector<int> &nums) {
    int n = nums.size();
    int begin = 0, end = n - 1, pivot = nums[0], endpivot = nums[n-1];
    while(begin < end - 1) {
        int mid = begin + (end - begin) / 2;
        if(nums[mid] < pivot) {
            end = mid;
        } else {
            if (nums[mid] < endpivot) {
                end = mid;
            } else {
                begin = mid;
            }
        }
    }
    return min(nums[begin], nums[end]);
}

int Solution::peakIndexInMountainArray(vector<int> &arr) {
    int n = arr.size();
    int begin = 0, end = n - 1;
    while(begin < end) {
        int mid = begin + (end - begin) / 2;
        if (arr[mid] >= arr[max(0, mid - 1)] && arr[mid] >= arr[min(n-1, mid + 1)]) {
            return mid;
        } else if (arr[mid] >= arr[max(0, mid - 1)]) {
            begin = mid + 1;
        } else {
            end = mid;
        }
    }
    return begin;
}

int Solution::findPeakElement(vector<int> &nums) {
    int n = nums.size();
    int begin = 0, end = n - 1;
    while(begin < end) {
        int mid = begin + (end - begin) / 2;
        if (nums[mid] >= nums[max(0, mid - 1)] && nums[mid] >= nums[min(n-1, mid + 1)]) {
            return mid;
        } else if (nums[mid] >= nums[max(0, mid - 1)]) {
            begin = mid + 1;
        } else {
            end = mid;
        }
    }
    return begin;
}

int Solution::uniquePaths(int m, int n) {
    // method1
    /*
    vector<vector<int>> dp(m, vector<int>(n));
    dp[0][0]=1;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            dp[i][j] += (i == 0) ? 0 : dp[i-1][j];
            dp[i][j] += (j == 0) ? 0 : dp[i][j-1];
        }
    }
    return dp[m-1][n-1];
    */
    // method2
    vector<int> dp(n);
    dp[0] = 1;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            dp[j] += (j == 0) ? 0 : dp[j-1];
        }
    }
    return dp[n-1];
}

int Solution::uniquePathsWithObstacles(vector<vector<int>> &obstacleGrid) {
    int m = obstacleGrid.size(), n = obstacleGrid[0].size();
    vector<vector<int>> dp(m, vector<int>(n));
    dp[0][0]=1;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (obstacleGrid[i][j] == 0) {
                dp[i][j] += (i == 0) ? 0 : dp[i-1][j];
                dp[i][j] += (j == 0) ? 0 : dp[i][j-1];
            } else {
                dp[i][j] = 0;
            }
        }
    }
    return dp[m-1][n-1];
    return 0;
}

int Solution::minPathSum(vector<vector<int>> &grid) {
    int m = grid.size(), n = grid[0].size();
    int maxValue = 0x7fffffff;
    vector<vector<int>> dp(m, vector<int>(n));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if(i == 0 && j == 0) {
                dp[i][j] = grid[0][0];
            } else {
                dp[i][j] = min((i == 0) ? maxValue : dp[i-1][j], (j == 0) ? maxValue : dp[i][j-1]) + grid[i][j];
            }

        }
    }
    return dp[m-1][n-1];
}

vector<vector<int>> Solution::threeSum(vector<int> &nums) {
    sort(nums.begin(), nums.end());
    vector<vector<int>> ans;
    int n = nums.size();
    if(nums.size() <= 2) {
        return {};
    }
    for (int i = 0; i < n; ++i) {
        if(nums[i] == nums[i+1]) {
            continue;
        }
        int k = n - 1;
        for (int j = i+1; j < n; ++j) {
            if(nums[j] == nums[j+1]) {
                continue;
            }
            while(j < k && nums[j] + nums[k] > -nums[i]) {
                k--;
            }
            if(nums[j] + nums[k] + nums[i] == 0) {
                ans.push_back({nums[i], nums[j], nums[k]});
            }
        }
    }
    return ans;
}

string Solution::longestPalindrome(string s) {
    // brute O(n^3) -> O(n^2) through O(n) check palindrome
    // dp O(n^2)
    int l = s.size();
    if(l <= 1) {
        return s;
    }
    if(l == 2) {
        return s[0] == s[1] ? s : s.substr(0,1);
    }
    vector<vector<int>> dp(l, vector<int>(l));
    for (int i = 0; i < l; ++i) {
        dp[i][i] = 1;
    }
    int maxVal = 0;
    pair<int, int> indexPair = {0, 0};
    for (int i = 2; i <= l; ++i) { // i is length
        for (int j = 0; j < l; ++j) { // j is the start index
            if(j+i-1 >= l) break;
            if (s[j] == s[j+i-1]) {
                if (i == 2) {
                    dp[j][j+i-1] = 2;
                } else {
                    dp[j][j+i-1] = dp[j+1][j+i-2] == 0 ? 0 : dp[j+1][j+i-2] + 2;
                }
                if (maxVal < dp[j][j + i - 1]) {
                    indexPair.first = j;
                    indexPair.second = j + i - 1;
                    maxVal = dp[j][j+i-1];
                }
            } else {
                dp[j][j+i-1] = 0;
            }
        }
    }
    return s.substr(indexPair.first, indexPair.second - indexPair.first + 1);
}

bool Solution::backspaceCompare(string s, string t) {
    int slen = s.size(), tlen = t.size();
    int i = slen - 1, j = tlen - 1;
    int icount = 0, jcount = 0;
    while (i >= 0 && j >= 0) {
        while (i >= 0) {
            if (s[i] == '#') {
                icount++, i--;
            } else if (icount > 0) {
                icount--, i--;
            } else {
                break;
            }
        }
        while (j >= 0) {
            if (t[j] == '#') {
                jcount++, j--;
            } else if (jcount > 0) {
                jcount--, j--;
            } else {
                break;
            }
        }
        if (i >= 0 && j >= 0) {
            if (s[i] != t[j]) {
                return false;
            }
        } else {
            if (i >= 0 || j >= 0) {
                return false;
            }
        }
        i--, j--;
    }
    return true;
}

vector<string> Solution::findRestaurant(vector<string> &list1, vector<string> &list2) {
    map<string, int> keySet;
    // 将list1插入map中
    for (int i = 0; i < list1.size(); ++i) {
        keySet.insert(pair<string, int>(list1[i], i));
    }
    int minSum = 0x7fffffff;
    vector<string> ans;
    for (int i = 0; i < list2.size(); ++i) {
        string str = list2[i];
        if (keySet.find(str) == keySet.end()) {
            continue; // 如果查不到，直接跳过
        } else {
            if (i + keySet[str] < minSum) { // 可以更新
                ans.clear();
                minSum = i + keySet[str];
                ans.push_back(str);
            } else if (i + keySet[str] == minSum) { // 直接添加
                ans.push_back(str);
            }
        }
    }
    return ans;
}

vector<vector<int>> Solution::intervalIntersection(vector<vector<int>> &firstList, vector<vector<int>> &secondList) {
    int i = 0, j = 0;
    int leni = firstList.size(), lenj = secondList.size();
    vector<vector<int>> ans;
    while (i < leni && j < lenj) {
        int starti = firstList[i][0];
        int endi = firstList[i][1];
        int startj = secondList[j][0];
        int endj = secondList[j][1];
        if (starti > endj) {
            j++;
        } else if (startj > endi) {
            i++;
        } else {
            vector<int> temp;
            temp.push_back(max(starti, startj));
            temp.push_back(min(endi, endj));
            ans.push_back(temp);
            if (endi < endj) {
                i++;
            } else {
                j++;
            }
        }
    }
    return ans;
}

int Solution::maxArea(vector<int> &height) {
    int begin = 0, end = height.size() - 1, ans = 0;
    while(begin < end) {
        if(height[begin] < height[end]) {
            ans = max(ans, (end - begin) * min(height[begin], height[end]));
            begin++;
        } else {
            ans = max(ans, (end - begin) * min(height[begin], height[end]));
            end--;
        }
    }
    return ans;
}

vector<int> Solution::findAnagrams(string s, string p) {
    int lens = s.size(), lenp = p.size();
    vector<int>count(26);
    vector<int>temp(26);
    vector<int>ans;
    if (lens < lenp) {
        return ans;
    }
    for (int i = 0; i < lenp; ++i) {
        count[p[i] - 'a']++;
    }
    for (int i = 0; i < lenp; ++i) {
        temp[s[i] - 'a']++;
    }
    for (int i = 0; i < lens - lenp + 1; ++i) {
        if (count == temp) {
            ans.push_back(i);
        }
        if (i == lens - lenp) {
            break;
        }
        temp[s[i] - 'a']--;
        temp[s[i+lenp] - 'a']++;
    }
    return ans;
}

int Solution::numSubarrayProductLessThanK(vector<int> &nums, int k) {
    if(k <= 1) {
        return 0;
    }
    int n = nums.size();
    int j = 0, ans = 0;
    int temp = 1;
    for (int i = 0; i < n; ++i) {
        temp *= nums[i];
        while (temp >= k) {
            temp /= nums[j];
            j++;
        }
        ans += i - j + 1;
    }
    return ans;
}

int Solution::minSubArrayLen(int target, vector<int> &nums) {
    int i = 0;
    int maxLen = 0x7fffffff;
    int test = INT_MAX;
    int sum = 0;
    for (int j = 0; j < nums.size(); ++j) {
        sum += nums[j];
        while(i < j && sum - nums[i] >= target) {
            sum -= nums[i];
            i++;
        }
        if (sum >= target) {
            maxLen = min(maxLen, j - i + 1);
        }
    }
    if (maxLen == 0x7fffffff) maxLen = 0;
    return maxLen;
}

int Solution::numIslands(vector<vector<char>> &grid) {
    int m = grid.size(), n = grid[0].size(), count = 0;
    vector<vector<int>> arrived(m, vector<int>(n));
    queue<pair<int, int>> q;
    vector<int> dir1{1,-1,0,0};
    vector<int> dir2{0,0,1,-1};
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (arrived[i][j] == 1 || grid[i][j] == '0') {
                continue;
            }
            q.emplace(i, j);
            arrived[i][j] = 1;
            while(!q.empty()) {
                int x0 = q.front().first;
                int y0 = q.front().second;
                for (int k = 0; k < 4; ++k) {
                    int d1 = dir1[k] + x0;
                    int d2 = dir2[k] + y0;
                    if(d1>=0 && d1<m && d2>=0 && d2<n && grid[d1][d2] == '1' && arrived[d1][d2] == 0) {
                        q.emplace(d1, d2);
                        arrived[d1][d2] = 1;
                    }
                }

                q.pop();
            }
            count++;
        }
    }
    return count;
}

int Solution::findCircleNum(vector<vector<int>> &isConnected) {
    int citynum = isConnected.size(), count = 0;
    queue<int> q;
    vector<int> arrived(citynum);
    for (int i = 0; i < citynum; ++i) {
        if (arrived[i] == 1) {
            continue;
        }
        arrived[i] = 1;
        count++;
        q.emplace(i);
        while(!q.empty()) {
            int t = q.front();
            for (int j = 0; j < citynum; ++j) {
                if (isConnected[t][j] && arrived[j] == 0) {
                    arrived[j] = 1;
                    q.emplace(j);
                }
            }
            q.pop();
        }
    }
    return count;
}

int Solution::countMaxOrSubsets(vector<int> &nums) {
    return 0;
}

bool Solution::isPalindrome(int x) {
    if (x < 0 || (x % 10 == 0 && x != 0)) {
        return false;
    }

    int revertedNumber = 0;
    while (x > revertedNumber) {
        revertedNumber = revertedNumber * 10 + x % 10;
        x /= 10;
    }
    return x == revertedNumber || x == revertedNumber / 10;
}

vector<vector<int>> ans_subsets;
vector<int> temp_subsets;

void dfs_subsets(int index, int content, vector<int> &nums) {
    if (index == content) {
        ans_subsets.push_back(temp_subsets);
        return;
    }
    temp_subsets.push_back(nums[index]);
    dfs_subsets(index + 1, content, nums);
    temp_subsets.pop_back();
    dfs_subsets(index + 1, content, nums);
}

vector<vector<int>> Solution::subsets(vector<int> &nums) {
    int n = nums.size();
    dfs_subsets(0, n, nums);
    return ans_subsets;
}

bool compare_longestWord(string stra, string strb) {
    return stra.size() < strb.size() || (stra.size() == strb.size() && (stra < strb));
}

string Solution::longestWord(vector<string> &words) {
    sort(words.begin(), words.end(), compare_longestWord);
    int n = words.size();
    unordered_set<string> strSet(n);
    strSet.insert("");
    string maxStr = "";
    int maxLen = 0;
    for (int i = 0; i < n; ++i) {
        int l = words[i].size();
        if (strSet.find(words[i].substr(0, l-1)) != strSet.end()) { // found
            strSet.insert(words[i]);
            if (l > maxLen) {
                maxLen = l;
                maxStr = words[i];
            }
        }
    }
    return maxStr;
}

int Solution::shortestPathBinaryMatrix(vector<vector<int>> &grid) {
    if(grid[0][0]==1)return -1;
    int n=grid.size(),ans=1;
    const int dire[8][2]={{1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,-1},{-1,1}};
    queue<pair<int,int> > q;
    q.emplace(0,0);         //从0,0开始
    grid[0][0]=1;           //标记为1代表走过
    while(!q.empty()){      //bfs
        int m=q.size();
        while(m--){
            auto [x,y]=q.front();
            q.pop();
            if(x==n-1&&y==n-1)return ans;
            for(int i=0;i<8;i++){                       //遍历八个方向的
                int nx=x+dire[i][0];
                int ny=y+dire[i][1];
                if(nx<0||ny<0||nx>=n||ny>=n)continue;   //判断是否越界
                if(grid[nx][ny]==0){        //判断是否能走
                    q.emplace(nx,ny);
                    grid[nx][ny]=1;         //标记
                }
            }
        }
        ans++;          //记录循环次数
    }
    return -1;
}

vector<vector<int>> ans_subsetsWithDup;
vector<int> temp_subsetsWithDup;

void dfs_subsetsWithDup(int pre, int layer, vector<int> &nums) {
    if (layer == nums.size()) {
        ans_subsetsWithDup.push_back(temp_subsetsWithDup);
        return;
    }

    // not choose
    dfs_subsetsWithDup(0, layer+1, nums);

    if (pre == 0 && layer > 0 && nums[layer] == nums[layer - 1]) {
        return;
    }
    // choose
    temp_subsetsWithDup.push_back(nums[layer]);
    dfs_subsetsWithDup(1, layer+1, nums);
    temp_subsetsWithDup.pop_back();
}


vector<vector<int>> Solution::subsetsWithDup(vector<int> &nums) {
    sort(nums.begin(), nums.end());
    dfs_subsetsWithDup(0, 0, nums);

    return ans_subsetsWithDup;
}

void Solution::solve(vector<vector<char>> &board) {
    int m = board.size(), n = board[0].size();
    vector<int> dirx = {1,-1,0,0};
    vector<int> diry = {0,0,1,-1};
    queue<pair<int, int>> temp;
    vector<vector<int>> arrive(m, vector<int>(n));
    vector<pair<int, int>> toSolve;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (board[i][j] == 'X' || arrive[i][j]) {
            } else {
                while(!temp.empty()) temp.pop();
                toSolve.clear();
                temp.emplace(i,j);
                arrive[i][j] = 1;
                int flag = 0;
                while(!temp.empty()) {
                    int x = temp.front().first;
                    int y = temp.front().second;
                    toSolve.emplace_back(x, y);
                    temp.pop();
                    if (x == m-1 || x == 0 || y == 0 || y == n-1) {
                        flag = 1;
                    }
                    for (int k = 0; k < 4; ++k) {
                        int x0 = x + dirx[k];
                        int y0 = y + diry[k];
                        if (x0 == m || x0 < 0 || y0 < 0 || y0 == n || arrive[x0][y0] || board[x0][y0] == 'X') {
                            continue;
                        }
                        arrive[x0][y0] = 1;
                        temp.emplace(x0, y0);
                    }
                }
                if (flag == 0) {
                    for (int k = 0; k < toSolve.size(); ++k) {
                        board[toSolve[k].first][toSolve[k].second] = 'X';
                    }
                }
            }
        }
    }
}

vector<vector<int>> ans_permuteUnique;
vector<int> temp_permuteUnique;

void dfs_permuteUnique(int layer, vector<int> &nums, vector<int> &mark) {
    int n = nums.size();
    if (layer == n) {
        ans_permuteUnique.emplace_back(temp_permuteUnique);
        return;
    }
    for (int i = 0; i < n; ++i) {
        if (mark[i] == 0) {
            temp_permuteUnique.emplace_back(nums[i]);
            mark[i] = 1;
            if (i == 0 || !(nums[i] == nums[i-1] && mark[i-1] == 1)) {
                dfs_permuteUnique(layer+1, nums, mark);
            }

            mark[i] = 0;
            temp_permuteUnique.pop_back();
        }
    }
}

vector<vector<int>> Solution::permuteUnique(vector<int> &nums) {
    sort(nums.begin(), nums.end());
    vector<int> mark(nums.size());
    dfs_permuteUnique(0, nums, mark);
//    for (int i = 0; i < ans_permuteUnique.size(); ++i) {
//        for (int j = 0; j < ans_permuteUnique[0].size(); ++j) {
//            cout << ans_permuteUnique[i][j] << " ";
//        }
//        cout << endl;
//    }
    return ans_permuteUnique;
}

vector<vector<int>> ans_allPathSourceTarget;
vector<int> temp_allPathSourceTarget;

void dfs_allPathSourceTarget(int layer, int point, vector<vector<int>> &graph, vector<int> &visited) {
    int n = graph.size();
    if (visited[n-1]) {
        ans_allPathSourceTarget.emplace_back(temp_allPathSourceTarget);
        return;
    }
    if (layer == n) {
        return;
    }
    vector<int> able = graph[point];
    for(int num: able) {
        if (!visited[num]) {
            visited[num] = 1;
            temp_allPathSourceTarget.emplace_back(num);
            dfs_allPathSourceTarget(layer+1, num, graph, visited);
            temp_allPathSourceTarget.pop_back();
            visited[num] = 0;
        }
    }
}

vector<vector<int>> Solution::allPathsSourceTarget(vector<vector<int>> &graph) {
    int n = graph.size();
    vector<int> visited(n);
    visited[0] = 1;
    temp_allPathSourceTarget.emplace_back(0);
    dfs_allPathSourceTarget(0, 0, graph, visited);
//    for (int i = 0; i < ans_allPathSourceTarget.size(); ++i) {
//        for (int j = 0; j < ans_allPathSourceTarget[i].size(); ++j) {
//            cout << ans_allPathSourceTarget[i][j] << " ";
//        }
//        cout << endl;
//    }
    return ans_allPathSourceTarget;
}

vector<vector<int>> ans_combinationSum;
vector<int> temp_combinationSum;

void dfs_combinationSum(int cur, int target, vector<int> &candidates, int index) {
    if (cur > target) {
        return;
    }
    if (cur == target) {
        ans_combinationSum.emplace_back(temp_combinationSum);
        return;
    }
    if (index == candidates.size()) {
        return;
    }
    int c = candidates[index];
    // use
    temp_combinationSum.emplace_back(c);
    dfs_combinationSum(cur+c, target, candidates, index);
    temp_combinationSum.pop_back();

    dfs_combinationSum(cur, target, candidates, index+1);
}

vector<vector<int>> Solution::combinationSum(vector<int> &candidates, int target) {
    dfs_combinationSum(0, target, candidates,0);
    return ans_combinationSum;
}

int getStep_findKth(int cur, long n) {
    // cur is the current father node, n is the max number
    // include the father node
    long step = 0;
    long first = cur;
    long last = cur;
    while(first <= n) {
        step += min(n, last) - first + 1;
        first = first * 10;
        last = last * 10 + 9;
    }
    return step;
}

int Solution::findKthNumber(int n, int k) {
    int cur = 1;
    while(k > 1) {
        int step = getStep_findKth(cur, n);
        if (k > step) {
            cur += 1;
            k -= step;
        } else {
            cur *= 10;
            k--;
        }
    }
    return cur;
}

vector<vector<int>> ans_combinationSum2;
vector<int> temp_combinationSum2;

void dfs_combinationSum2(vector<int> &candidates, int target, int index, int cur, vector<int> &mark) {
    int n = candidates.size();
    if (cur > target) {
        return;
    }
    if (cur == target) {
        ans_combinationSum2.emplace_back(temp_combinationSum2);
        return;
    }
    if (index == n) {
        return;
    }

    if (index == 0 || !(mark[index-1] == 0 && candidates[index] == candidates[index-1])) {
        temp_combinationSum2.emplace_back(candidates[index]);
        mark[index] = 1;
        dfs_combinationSum2(candidates, target, index+1, cur+candidates[index], mark);
        mark[index] = 0;
        temp_combinationSum2.pop_back();
    }


    dfs_combinationSum2(candidates, target, index+1, cur, mark);
}

vector<vector<int>> Solution::combinationSum2(vector<int> &candidates, int target) {
    int n = candidates.size();
    sort(candidates.begin(), candidates.end());
    vector<int> mark(n);
    dfs_combinationSum2(candidates, target, 0, 0, mark);
//    for (int i = 0; i < ans_combinationSum2.size(); ++i) {
//        for (int j = 0; j < ans_combinationSum2[i].size(); ++j) {
//            cout << ans_combinationSum2[i][j] << " ";
//        }
//        cout << endl;
//    }
    return ans_combinationSum2;
}

int Solution::lengthOfLIS(vector<int> &nums) {
    int len = 1, n = (int)nums.size();
    if (n == 0) {
        return 0;
    }
    vector<int> d(n + 1, 0);
    d[len] = nums[0];
    for (int i = 1; i < n; ++i) {
        if (nums[i] > d[len]) {
            d[++len] = nums[i];
        } else {
            int l = 1, r = len, pos = 0; // 如果找不到说明所有的数都比 nums[i] 大，此时要更新 d[1]，所以这里将 pos 设为 0
            while (l <= r) {
                int mid = (l + r) >> 1;
                if (d[mid] < nums[i]) {
                    pos = mid;
                    l = mid + 1;
                } else {
                    r = mid - 1;
                }
            }
            d[pos + 1] = nums[i];
        }
    }
    return len;
}

int Solution::trailingZeroes(int n) {
    return n / 5 + n / 25 + n / 125 + n / 625 + n / 3125;
}

int Solution::findNumberOfLIS(vector<int> &nums) {
    // method1 dp

    int n = nums.size();
    vector<int> dp(n);
    vector<int> count(n);
    dp[0] = 1;
    count[0] = 1;
    int maxLen = 0;
    int ans = 0;
    for (int i = 0; i < n; ++i) {
        dp[i]=1;
        count[i]=1;
        for (int j = 0; j < i; ++j) {
            if (nums[j] < nums[i]) {
                if (dp[i] < dp[j] + 1) {
                    dp[i] = dp[j] + 1;
                    count[i] = count[j];
                } else if (dp[i] == dp[j] + 1) {
                    count[i]+=count[j];
                }
            }
        }
        // 最后的返回值应该是所有最大长度的所有count的总和
        if (dp[i] > maxLen) {
            maxLen = dp[i];
            ans = count[i];
        } else if (dp[i] == maxLen) {
            ans += count[i];
        }
    }
    return ans;
}

int Solution::longestCommonSubsequence(string text1, string text2) {
    int l1 = text1.size(), l2 = text2.size();
//    vector<vector<int>> dp(l1+1, vector<int>(l2+1));
//    for (int i = 1; i <= l1; ++i) {
//        dp[i][1] = dp[i-1][1];
//        for (int j = 1; j <= l2; ++j) {
//
//            if (text1[i-1] == text2[j-1]) {
//                dp[i][j] = dp[i-1][j-1] + 1;
//            } else {
//                dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
//            }
//        }
//    }
//    return dp[l1][l2];

    vector<vector<int>> dp(2, vector<int>(l2+1));
    for (int i = 1; i <= l1; ++i) {
        for (int j = 1; j <= l2; ++j) {
            if (text1[i-1] == text2[j-1]) {
                dp[(i+1)%2][j] = dp[i%2][j-1] + 1;
            } else {
                dp[(i+1)%2][j] = max(dp[i%2][j], dp[(i+1)%2][j-1]);
            }
        }
    }
    return max(dp[0][l2], dp[1][l2]);
}

int getCount_maxConsecutive(char c, string answerKey, int k) {
    int len = answerKey.size();
    int maxLen = 0;
    for (int i = 0, j = 0, sum = 0; i < len; ++i) {
        if (answerKey[i] != c) {
            sum++;
        }
        while (k < sum) {
            if (answerKey[j] != c) {
                sum--;
            }
            j++;
        }
        maxLen = max(maxLen, i-j+1);
    }
    return maxLen;
}

int Solution::maxConsecutiveAnswers(string answerKey, int k) {
    return max(getCount_maxConsecutive('F', answerKey, k), getCount_maxConsecutive('T', answerKey, k));
}

vector<string> ans_generateParenthesis;
string temp_generateParenthesis;

void dfs_generateParenthesis(int leftNum, int rightNum, int n) {
    if (leftNum == n && rightNum == n) {
        ans_generateParenthesis.emplace_back(temp_generateParenthesis);
        return;
    }
    if (leftNum < n) {
        temp_generateParenthesis.push_back('(');
        dfs_generateParenthesis(leftNum+1, rightNum, n);
        temp_generateParenthesis.pop_back();
    }

    if (leftNum > rightNum) {
        temp_generateParenthesis.push_back(')');
        dfs_generateParenthesis(leftNum, rightNum+1, n);
        temp_generateParenthesis.pop_back();
    }
}

vector<string> Solution::generateParenthesis(int n) {
    dfs_generateParenthesis(0, 0, n);
    for (int i = 0; i < ans_generateParenthesis.size(); ++i) {
        cout << ans_generateParenthesis[i] << endl;
    }
    return ans_generateParenthesis;
}

vector<int> Solution::countBits(int n) {
    /*
     * 0
     * 1
     * 10
     * 11
     * 100
     * 101
     * 110
     * 111
     */
    vector<int> ans(n);
    for (int i = 0; i < n; ++i) {
        if (i % 2 == 1) {
            ans[i] = ans[i-1] + 1;
        } else {
            ans[i] = ans[i/2];
        }
    }
    return ans;
}

string Solution::convert(string s, int numRows) {
    // 0 4 8
    // 1 3 5 7
    // 2 6 10

    // 0 6 12
    // 1 5 7 11
    // 2 4 8 10
    // 3 9 15
    int len = s.size();
    int step = 2 * (numRows - 1);
    if (step == 0) step++;
    string newstr = "";
    for (int i = 0; i < numRows; ++i) {
        if (i == 0 || i == numRows - 1) {
            for (int j = i; j < len; j+=step) {
                newstr.push_back(s[j]);
            }
        } else {
            for (int j = 0; j < len; j+=step) {
                if (j+i < len) {
                    newstr.push_back(s[j+i]);
                }
                if (j+step-i < len) {
                    newstr.push_back(s[j+step-i]);
                }
            }
        }

    }
    return newstr;
}

int Solution::romanToInt(string s) {
    int len = s.size();
    int sum = 0;
    for (int i = 0; i < len; ++i) {
        if (s[i] == 'I') {
            if (i+1 < len && (s[i+1] == 'V' || s[i+1] == 'X')) {
                if (s[i+1] == 'V') sum += 4;
                if (s[i+1] == 'X') sum += 9;
                i++;
            } else {
                sum += 1;
            }
        }

        else if (s[i] == 'V') sum += 5;

        else if (s[i] == 'X') {
            if (i+1 < len && (s[i+1] == 'L' || s[i+1] == 'C')) {
                if (s[i+1] == 'L') sum += 40;
                if (s[i+1] == 'C') sum += 90;
                i++;
            } else {
                sum += 10;
            }
        }

        else if (s[i] == 'L') sum += 50;

        else if (s[i] == 'C') {
            if (i+1 < len && (s[i+1] == 'D' || s[i+1] == 'M')) {
                if (s[i+1] == 'D') sum += 400;
                if (s[i+1] == 'M') sum += 900;
                i++;
            } else {
                sum += 100;
            }
        }

        else if (s[i] == 'D') sum += 500;
        else if (s[i] == 'M') sum += 1000;
    }
    return sum;
}

bool Solution::canReorderDoubled(vector<int> &arr) {
    sort(arr.begin(), arr.end());
    map<int, int> cntMap;
    int len = arr.size();
    for (int i = 0; i < len; ++i) {
        if (cntMap.find(arr[i]) != cntMap.end()) {
            cntMap[arr[i]]++;
        } else {
            cntMap.insert(pair<int, int>(arr[i], 1));
        }
    }
    int sep=0;
    int flag=0;
    while(sep<len && arr[sep] <= 0) sep++;
    for (int i = sep; i < len; ++i) {
        if (i>=len) break;
        if (cntMap.find(arr[i]) != cntMap.end() && cntMap.find(2*arr[i]) != cntMap.end()) {
            cntMap[arr[i]]--;
            cntMap[2*arr[i]]--;
            if (cntMap[arr[i]] == 0) cntMap.erase(cntMap.find(arr[i]));
            if (cntMap[2*arr[i]] == 0) cntMap.erase(cntMap.find(2*arr[i]));
        }
    }
    for (int i = sep-1; i >= 0; --i) {
        if (i<0) break;
        if (cntMap.find(arr[i]) != cntMap.end() && cntMap.find(2*arr[i]) != cntMap.end()) {
            cntMap[arr[i]]--;
            cntMap[2*arr[i]]--;
            if (cntMap[arr[i]] == 0) cntMap.erase(cntMap.find(arr[i]));
            if (cntMap[2*arr[i]] == 0) cntMap.erase(cntMap.find(2*arr[i]));
        }
    }
    return cntMap.size() == 0;
}

