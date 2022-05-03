剑指offer

09.用两个栈实现一个队列。队列的声明如下，请实现它的两个函数 appendTail 和 deleteHead ，分别完成在队列尾部插入整数和在队列头部删除整数的功能。(若队列中没有元素，deleteHead 操作返回 -1 )

 

示例 1：

输入：
["CQueue","appendTail","deleteHead","deleteHead"]
[[],[3],[],[]]
输出：[null,null,3,-1]
示例 2：

输入：
["CQueue","deleteHead","appendTail","appendTail","deleteHead","deleteHead"]
[[],[],[5],[2],[],[]]
输出：[null,-1,null,null,5,2]
提示：

1 <= values <= 10000
最多会对 appendTail、deleteHead 进行 10000 次调用

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class CQueue {
private:
    stack<int> inStack, outStack;
    void in2out(){
        while(!inStack.empty()){
            outStack.push(inStack.top());
            inStack.pop();
        }
    }
public:
    CQueue() {

    }
    
    void appendTail(int value) {
        inStack.push(value);
    }
    
    int deleteHead() {
        if(outStack.empty()) {
            if(inStack.empty()) return -1;
            in2out();
        }
        int head = outStack.top();
        outStack.pop();
        return head;
    }

};

/**
 * Your CQueue object will be instantiated and called as such:
 * CQueue* obj = new CQueue();
 * obj->appendTail(value);
 * int param_2 = obj->deleteHead();
 */
```

30.定义栈的数据结构，请在该类型中实现一个能够得到栈的最小元素的 min 函数在该栈中，调用 min、push 及 pop 的时间复杂度都是 O(1)。

 

示例:

MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.min();   --> 返回 -3.
minStack.pop();
minStack.top();      --> 返回 0.
minStack.min();   --> 返回 -2.


提示：

各函数的调用总次数不超过 20000 次

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/bao-han-minhan-shu-de-zhan-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。



```c++
class MinStack {
public:
    stack<int> x_stack, min_stack;
    /** initialize your data structure here. */
    MinStack() {
        min_stack.push(INT_MAX);
    }
    
    void push(int x) {
        x_stack.push(x);
        min_stack.push(::min(min_stack.top(), x));
    }
    
    void pop() {
        x_stack.pop();
        min_stack.pop();
    }
    
    int top() {
        return x_stack.top();
    }
    
    int min() {
        return min_stack.top();
    }
};

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack* obj = new MinStack();
 * obj->push(x);
 * obj->pop();
 * int param_3 = obj->top();
 * int param_4 = obj->min();
 */
```

06 输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。

 

**示例 1：**

```
输入：head = [1,3,2]
输出：[2,3,1]
```

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    vector<int> reversePrint(ListNode* head) {
        stack<int> container;
        while(head != NULL){
            container.push(head->val);
            head = head->next;
        }
        vector<int> res;
        while(!container.empty()){
            res.push_back(container.top());
            container.pop();
        }
        return res;
    }
};
```

24 定义一个函数，输入一个链表的头节点，反转该链表并输出反转后链表的头节点。

 

示例:

输入: 1->2->3->4->5->NULL
输出: 5->4->3->2->1->NULL


限制：

0 <= 节点个数 <= 5000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/fan-zhuan-lian-biao-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode* pre = nullptr;
        ListNode* cur = head;
        while(cur){
            ListNode* next = cur->next;
            cur->next = pre;
            pre = cur;
            cur = next;
        }
        return pre;
    }
};
```

35 请实现 copyRandomList 函数，复制一个复杂链表。在复杂链表中，每个节点除了有一个 next 指针指向下一个节点，还有一个 random 指针指向链表中的任意节点或者 null。

 

示例 1：



输入：head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
输出：[[7,null],[13,0],[11,4],[10,2],[1,0]]
示例 2：



输入：head = [[1,1],[2,1]]
输出：[[1,1],[2,1]]
示例 3：



输入：head = [[3,null],[3,0],[3,null]]
输出：[[3,null],[3,0],[3,null]]
示例 4：

输入：head = []
输出：[]
解释：给定的链表为空（空指针），因此返回 null。


提示：

-10000 <= Node.val <= 10000
Node.random 为空（null）或指向链表中的节点。
节点数目不超过 1000 。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/fu-za-lian-biao-de-fu-zhi-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    Node* copyRandomList(Node* head) {
        if(head == NULL) return NULL;
        unordered_map<Node*, Node*> map;
        Node* cur = head;
        while(cur != NULL){
            map[cur] = new Node(cur->val);
            cur = cur->next;
        }
        cur = head;
        while(cur){
            map[cur]->next = map[cur->next];
            map[cur]->random = map[cur->random];
            cur = cur->next;
        }
        return map[head];

    }
};
```

leetcode

94 给定一个二叉树的根节点 root ，返回 它的 中序 遍历 。

 

示例 1：


输入：root = [1,null,2,3]
输出：[1,3,2]
示例 2：

输入：root = []
输出：[]
示例 3：

输入：root = [1]
输出：[1]


提示：

树中节点数目在范围 [0, 100] 内
-100 <= Node.val <= 100


进阶: 递归算法很简单，你可以通过迭代算法完成吗？

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/binary-tree-inorder-traversal
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。



```c++
class Solution {
public:
    void inorder(TreeNode* root, vector<int>& res){
        if(root == nullptr) return;
        inorder(root->left, res);
        res.push_back(root->val);
        inorder(root->right, res);
    }
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> res;
        inorder(root, res);
        return res;
    }
};

class Solution {
public:
    
    vector<int> inorderTraversal(TreeNode* root) {
        stack<TreeNode*> container;
        vector<int> res;
        while(root != nullptr || !container.empty()){
            while(root != nullptr){
                container.push(root);
                root = root->left;
            }
            root = container.top();
            res.push_back(root->val);
            container.pop();
            root = root->right;
        }
        return res;
    }
};
```

95 给你一个整数 n ，请你生成并返回所有由 n 个节点组成且节点值从 1 到 n 互不相同的不同 二叉搜索树 。可以按 任意顺序 返回答案。

 

示例 1：


输入：n = 3
输出：[[1,null,2,null,3],[1,null,3,2],[2,1,3],[3,1,null,null,2],[3,2,null,1]]
示例 2：

输入：n = 1
输出：[[1]]


提示：

1 <= n <= 8

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/unique-binary-search-trees-ii
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

二叉搜索树任意节点，其左子树上任意节点均小于该节点，右子树上任意节点均大于该节点。

```c++
class Solution {
public:
    vector<TreeNode*> generateTrees(int start, int end){
        if(start > end) return {nullptr};
        vector<TreeNode*> all_trees;
        for(int i = start; i <= end; i++){
            vector<TreeNode*> left_trees = generateTrees(start, i - 1);
            vector<TreeNode*> right_trees = generateTrees(i + 1, end);

            for(auto& left: left_trees){
                for(auto& right: right_trees){
                    TreeNode* curr = new TreeNode(i);
                    curr->left = left;
                    curr->right = right;
                    all_trees.emplace_back(curr);
                }
            }
        }
        return all_trees;
    }
    vector<TreeNode*> generateTrees(int n) {
        if(n < 1) return {nullptr};
        return generateTrees(1, n);
    }
};
```

05 请实现一个函数，把字符串 s 中的每个空格替换成"%20"。

 

示例 1：

输入：s = "We are happy."
输出："We%20are%20happy."


限制：

0 <= s 的长度 <= 10000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/ti-huan-kong-ge-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    string replaceSpace(string s) {
        vector<char> strings;
        for(int i = 0; i < s.size(); i++)
            if(s[i] != ' ') strings.push_back(s[i]);
            else {strings.push_back('%');strings.push_back('2');strings.push_back('0');}
        string res = "";
        for(char s: strings) res += s;
        return res;
    }
};
```

58 字符串的左旋转操作是把字符串前面的若干个字符转移到字符串的尾部。请定义一个函数实现字符串左旋转操作的功能。比如，输入字符串"abcdefg"和数字2，该函数将返回左旋转两位得到的结果"cdefgab"。

 

示例 1：

输入: s = "abcdefg", k = 2
输出: "cdefgab"
示例 2：

输入: s = "lrloseumgh", k = 6
输出: "umghlrlose"


限制：

1 <= k < s.length <= 10000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/zuo-xuan-zhuan-zi-fu-chuan-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    string reverseLeftWords(string str,int n){
        if(str.size()==0) return str;
        string s = str.substr(0,n);
        string res = str.substr(n,str.size()-n) + s;
        return res;
    }
};
```

03 找出数组中重复的数字。


在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。

示例 1：

输入：
[2, 3, 1, 0, 2, 5, 3]
输出：2 或 3 


限制：

2 <= n <= 100000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    int findRepeatNumber(vector<int>& nums) {
        int n = nums.size();
        unordered_map<int, int> map;
        for(int i = 0; i < n; i++){
            map[nums[i]]++;
            if(map[nums[i]] > 1) return nums[i];
        }
        return 0;

    }
};
```

53 统计一个数字在排序数组中出现的次数。

 

示例 1:

输入: nums = [5,7,7,8,8,10], target = 8
输出: 2
示例 2:

输入: nums = [5,7,7,8,8,10], target = 6
输出: 0


提示：

0 <= nums.length <= 105
-109 <= nums[i] <= 109
nums 是一个非递减数组
-109 <= target <= 109

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/zai-pai-xu-shu-zu-zhong-cha-zhao-shu-zi-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int res = 0;
        if(nums.size() == 0) return 0;
        for(int i = 0; i < nums.size(); i++){
            if(nums[i] == target){
                res++;
                i++;
                while(i < nums.size() && nums[i] == target ){
                    res++;
                    i++;
                }
                return res;
            }
        }
        return res;
    }
};
```

53 一个长度为n-1的递增排序数组中的所有数字都是唯一的，并且每个数字都在范围0～n-1之内。在范围0～n-1内的n个数字中有且只有一个数字不在该数组中，请找出这个数字。

 

示例 1:

输入: [0,1,3]
输出: 2
示例 2:

输入: [0,1,2,3,4,5,6,7,9]
输出: 8


限制：

1 <= 数组长度 <= 10000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/que-shi-de-shu-zi-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    int missingNumber(vector<int>& nums) {
        int n = nums.size();
        for(int i = 0; i < n; i++){
            if(i != nums[i]) return i;
        }
        return nums[n - 1] + 1;
    }
};
```

04 在一个 n * m 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个高效的函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

 

示例:

现有矩阵 matrix 如下：

[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
给定 target = 5，返回 true。

给定 target = 20，返回 false。

 

限制：

0 <= n <= 1000

0 <= m <= 1000

 

注意：本题与主站 240 题相同：https://leetcode-cn.com/problems/search-a-2d-matrix-ii/

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/er-wei-shu-zu-zhong-de-cha-zhao-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    bool findNumberIn2DArray(vector<vector<int>>& matrix, int target) {
        int m = matrix.size();
        if(m == 0) return false;
        int n = matrix[0].size();
        if(n == 0) return false;
        for(int i = 0; i < m; i++)
            for(int j = 0; j < n; j++)
                if(matrix[i][j] == target)
                    return true;
        return false;
    }
};


class Solution {
public:
    bool findNumberIn2DArray(vector<vector<int>>& matrix, int target) {
        int m = matrix.size();
        if(m == 0) return false;
        int n = matrix[0].size();
        if(n == 0) return false;
        int row = 0, col = n - 1;
        while(row < m && col >= 0){
            if(matrix[row][col] == target) return true;
            else if(matrix[row][col] > target) col--;
            else if(matrix[row][col] < target) row++;
        }
        return false;
    }
};
```

50 在字符串 s 中找出第一个只出现一次的字符。如果没有，返回一个单空格。 s 只包含小写字母。

示例 1:

输入：s = "abaccdeff"
输出：'b'
示例 2:

输入：s = "" 
输出：' '


限制：

0 <= s 的长度 <= 50000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/di-yi-ge-zhi-chu-xian-yi-ci-de-zi-fu-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    char firstUniqChar(string s) {
        if(s.size() == 0) return ' ';
        unordered_map<char, int> map;
        for(char c: s) map[c]++;
        for(char c: s) if(map[c] == 1) return c;
        return ' ';
    }
};

class Solution {
public:
    char firstUniqChar(string s) {
        int table[30] = {0};
        for(int i = 0;i < s.size();i++)
            table[(int)(s[i] - 'a')]++;
        for(int i = 0;i < s.size();i++)
        {
            if(table[(int)(s[i] - 'a')] > 1)
                continue;
            else
                return s[i];
        }
        return ' ';
    }
};
```

32 从上到下打印出二叉树的每个节点，同一层的节点按照从左到右的顺序打印。

 

例如:
给定二叉树: [3,9,20,null,null,15,7],

    3
   / \
  9  20
    /  \
   15   7
返回：

[3,9,20,15,7]


提示：

节点总数 <= 1000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    vector<int> levelOrder(TreeNode* root) {
        if(root == NULL) return {};
        vector<int> res;
        queue<TreeNode*> q;
        q.push(root);
        while(!q.empty()){
            int sz = q.size();
            for(int i = 0; i < sz; i++){
                TreeNode* node = q.front();
                q.pop();
                res.push_back(node->val);
                if(node->left != NULL) q.push(node->left);
                if(node->right != NULL) q.push(node->right);
            }
        }
        return res;
    }
};
```

32从上到下按层打印二叉树，同一层的节点按从左到右的顺序打印，每一层打印到一行。

 

例如:
给定二叉树: [3,9,20,null,null,15,7],

    3
   / \
  9  20
    /  \
   15   7
返回其层次遍历结果：

[
  [3],
  [9,20],
  [15,7]
]


提示：

节点总数 <= 1000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-ii-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        if(root == NULL) return {};
        vector<vector<int>> res;
        queue<TreeNode*> q;
        q.push(root);
        while(!q.empty()){
            int sz = q.size();
            vector<int> row;
            for(int i = 0; i < sz; i++){
                TreeNode* node = q.front();
                q.pop();
                row.push_back(node->val);
                if(node->left != NULL) q.push(node->left);
                if(node->right != NULL) q.push(node->right);
            }
            res.push_back(row);
        }
        return res;
    }
};
```

32 请实现一个函数按照之字形顺序打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右到左的顺序打印，第三行再按照从左到右的顺序打印，其他行以此类推。

 

例如:
给定二叉树: [3,9,20,null,null,15,7],

    3
   / \
  9  20
    /  \
   15   7
返回其层次遍历结果：

[
  [3],
  [20,9],
  [15,7]
]


提示：

节点总数 <= 1000
通过次数176,074提交次数299,806

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-iii-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        if(root == NULL) return {};
        vector<vector<int>> res;
        queue<TreeNode*> q;
        bool right = false;
        q.push(root);
        while(!q.empty()){
            int sz = q.size();
            vector<int> row;
            for(int i = 0; i < sz; i++){
                TreeNode* node = q.front();
                q.pop();
                row.push_back(node->val);
                if(node->left != NULL) q.push(node->left);
                if(node->right != NULL) q.push(node->right);
            }
            if(right){                
                reverse(row.begin(), row.end());
                right = false;
            }
            else right = true;
            res.push_back(row);
        }
        return res;
    }
};
```

26 输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)

B是A的子结构， 即 A中有出现和B相同的结构和节点值。

例如:
给定的树 A:

     3
    / \
   4   5
  / \
 1   2
给定的树 B：

   4 
  /
 1
返回 true，因为 B 与 A 的一个子树拥有相同的结构和节点值。

示例 1：

输入：A = [1,2,3], B = [3,1]
输出：false
示例 2：

输入：A = [3,4,5,1,2], B = [4,1]
输出：true
限制：

0 <= 节点个数 <= 10000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/shu-de-zi-jie-gou-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    bool subStructure(TreeNode* A, TreeNode* B){
        if(B == NULL) return true; //B空返回true
        if(A == NULL) return false; 
        if(A->val != B->val) return false;
        else return subStructure(A->left, B->left) && subStructure(A->right, B->right);
    }
    bool isSubStructure(TreeNode* A, TreeNode* B) {
        // 如果B空，返回false
        //在A中寻找B的根节点，如果A中找到一个节点等于B的根节点，那么比较A的左子树和右子树
        if(A == NULL) return false;
        if(B == NULL) return false;
        if(A->val == B->val) 
            return (subStructure(A->left, B->left) && subStructure(A->right, B->right)) ||
            (isSubStructure(A->left, B) || isSubStructure(A->right, B));
        else return isSubStructure(A->left, B) || isSubStructure(A->right, B);

    }
};
```

27 请完成一个函数，输入一个二叉树，该函数输出它的镜像。

例如输入：

     4
   /   \
  2     7
 / \   / \
1   3 6   9
镜像输出：

     4
   /   \
  7     2
 / \   / \
9   6 3   1

 

示例 1：

输入：root = [4,2,7,1,3,6,9]
输出：[4,7,2,9,6,3,1]


限制：

0 <= 节点个数 <= 1000

注意：本题与主站 226 题相同：https://leetcode-cn.com/problems/invert-binary-tree/

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/er-cha-shu-de-jing-xiang-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    TreeNode* mirrorTree(TreeNode* root) {
        if(root == NULL) return NULL;
        TreeNode* res = new TreeNode(root->val);
        res->left = mirrorTree(root->right);
        res->right = mirrorTree(root->left);
        return res;
    }
};
```

28 请实现一个函数，用来判断一棵二叉树是不是对称的。如果一棵二叉树和它的镜像一样，那么它是对称的。

例如，二叉树 [1,2,2,3,4,4,3] 是对称的。

    1
   / \
  2   2
 / \ / \
3  4 4  3
但是下面这个 [1,2,2,null,3,null,3] 则不是镜像对称的:

    1
   / \
  2   2
   \   \
   3    3

 

示例 1：

输入：root = [1,2,2,3,4,4,3]
输出：true
示例 2：

输入：root = [1,2,2,null,3,null,3]
输出：false


限制：

0 <= 节点个数 <= 1000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/dui-cheng-de-er-cha-shu-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    bool isSymmetric(TreeNode* left, TreeNode* right){
        if(left == NULL && right == NULL) return true;
        if(left != NULL && right == NULL) return false;
        if(left == NULL && right != NULL) return false;
        if(left->val != right->val) return false;
        else return isSymmetric(left->left, right->right) && isSymmetric(left->right, right->left);
    }
    bool isSymmetric(TreeNode* root) {
        //如果左子树和右子树相同，那么就是对称的
        //不是相同
        //空树是对称的
        if(root == NULL) return true;
        return isSymmetric(root->left, root->right);
    }
};
```

10 写一个函数，输入 n ，求斐波那契（Fibonacci）数列的第 n 项（即 F(N)）。斐波那契数列的定义如下：

F(0) = 0,   F(1) = 1
F(N) = F(N - 1) + F(N - 2), 其中 N > 1.
斐波那契数列由 0 和 1 开始，之后的斐波那契数就是由之前的两数相加而得出。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

 

示例 1：

输入：n = 2
输出：1
示例 2：

输入：n = 5
输出：5


提示：

0 <= n <= 100

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    int fib(int n) {
        if(n == 0) return 0;
        if(n == 1) return 1;
        int n_1 = 1, n_2 = 0, cur = 0;//n - 1, n - 2
        int k = 1e9 + 7;
        for(int i = 2; i <= n; i++){
            cur = (n_1+ n_2) % k; //fn = f(n - 1) + f(n - 2)
            // (a + b) % k = (Ak + c + Bk + d) % k = ((Ak + Bk) + (c + d)) % k = c + d = a % k + b % k
            n_2 = n_1;
            n_1 = cur;
        }
        return cur;
    }
};
```

10-2 一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个 n 级的台阶总共有多少种跳法。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

示例 1：

输入：n = 2
输出：2
示例 2：

输入：n = 7
输出：21
示例 3：

输入：n = 0
输出：1
提示：

0 <= n <= 100
注意：本题与主站 70 题相同：https://leetcode-cn.com/problems/climbing-stairs/

 

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/qing-wa-tiao-tai-jie-wen-ti-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    int numWays(int n) {
        if(n == 0) return 1;
        if(n == 1) return 1;
        if(n == 2) return 2;
        
        int k = 1e9 + 7, n_1 = 2, n_2 = 1, cur = 0;
        for(int i = 3; i <= n; i++){
            cur = (n_1 + n_2) % k;
            n_2 = n_1;
            n_1 = cur;
        }
        return cur;
    }
};
```

63 假设把某股票的价格按照时间先后顺序存储在数组中，请问买卖该股票一次可能获得的最大利润是多少？

 

示例 1:

输入: [7,1,5,3,6,4]
输出: 5
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格。
示例 2:

输入: [7,6,4,3,1]
输出: 0
解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。


限制：

0 <= 数组长度 <= 10^5

 

注意：本题与主站 121 题相同：https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/gu-piao-de-zui-da-li-run-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        if(n == 0) return 0;
        if(n == 1) return 0;
        //在第i天卖出的利润最大值 = prices[i] - min_pre,
        int min_pre = prices[0], res = 0;
        for(int i = 1; i < n; i++){
            res = res > prices[i] - min_pre ? res : prices[i] - min_pre;
            min_pre = min_pre < prices[i] ? min_pre : prices[i];
        }
        return res;
    }
};
```

42 输入一个整型数组，数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。

要求时间复杂度为O(n)。

 

示例1:

输入: nums = [-2,1,-3,4,-1,2,1,-5,4]
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。


提示：

1 <= arr.length <= 10^5
-100 <= arr[i] <= 100

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/lian-xu-zi-shu-zu-de-zui-da-he-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int n = nums.size();
        int res = nums[0];
        //dp[i] nums[i] 结尾的子数组的最大值，如果dp[i - 1] + nums[i] > nums[i],赋值
        //如果dp[i - 1] + nums[i] < nums[i]，说明之前的值都比现在的小，当前的就是最大值
        vector<int> dp(n);
        dp[0] = nums[0];
        for(int i = 1; i < n; i++){
            dp[i] = dp[i - 1] + nums[i] > nums[i] ? dp[i - 1] + nums[i] : nums[i];
            res = dp[i] > res ? dp[i] : res;
        }
        return res;
    }
};


//只需要dp[i] 和 dp[i - 1]
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int n = nums.size();
        int res = nums[0];
        int pre = nums[0];
        for(int i = 1; i < n; i++){
            pre = pre + nums[i] > nums[i] ? pre + nums[i] : nums[i];
            res = pre > res ? pre : res;
        }
        return res;
    }
};
```

47 在一个 m*n 的棋盘的每一格都放有一个礼物，每个礼物都有一定的价值（价值大于 0）。你可以从棋盘的左上角开始拿格子里的礼物，并每次向右或者向下移动一格、直到到达棋盘的右下角。给定一个棋盘及其上面的礼物的价值，请计算你最多能拿到多少价值的礼物？

 

示例 1:

输入: 
[
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
输出: 12
解释: 路径 1→3→5→2→1 可以拿到最多价值的礼物


提示：

0 < grid.length <= 200
0 < grid[0].length <= 200

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/li-wu-de-zui-da-jie-zhi-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    int maxValue(vector<vector<int>>& grid) {
        //dp(i,j)：到位置ij拿到的礼物的最大价值
        //dp(0, 0) = grid[0][0]
        //dp(i,j) = max(dp(i - 1, j) + grid[i][j], dp(i, j - 1) + grid[i][j])
        int m = grid.size();
        int n = grid[0].size();
        vector<vector<int>> dp(m, vector<int>(n));
        dp[0][0] = grid[0][0];
        for(int i = 1; i < m; i++)
            dp[i][0] = dp[i - 1][0] + grid[i][0];
        for(int i = 1; i < n; i++)
            dp[0][i] = dp[0][i - 1] + grid[0][i];
        for(int i = 1; i < m; i++){
            for(int j = 1; j < n; j++){
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
            }
        }
        return dp[m - 1][n - 1];
    }  
};


class Solution {
public:
    int maxValue(vector<vector<int>>& grid) {
        //dp(i,j)：到位置ij拿到的礼物的最大价值
        //dp(0, 0) = grid[0][0]
        //dp(i,j) = max(dp(i - 1, j) + grid[i][j], dp(i, j - 1) + grid[i][j])
        int m = grid.size();
        int n = grid[0].size();
        vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
        
        for(int i = 1; i <= m; i++){
            for(int j = 1; j <= n; j++){
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]) + grid[i - 1][j - 1];
            }
        }
        return dp[m][n];
    }  
};
```

46 给定一个数字，我们按照如下规则把它翻译为字符串：0 翻译成 “a” ，1 翻译成 “b”，……，11 翻译成 “l”，……，25 翻译成 “z”。一个数字可能有多个翻译。请编程实现一个函数，用来计算一个数字有多少种不同的翻译方法。

 

示例 1:

输入: 12258
输出: 5
解释: 12258有5种不同的翻译，分别是"bccfi", "bwfi", "bczi", "mcfi"和"mzi"


提示：

0 <= num < 231

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/ba-shu-zi-fan-yi-cheng-zi-fu-chuan-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    int translateNum(int num) {
        //假设数字最后一位为i，那么i的翻译只和它和前一位数有关。
        //转成数组
        vector<int> nums;
        while(num >= 10){
            nums.push_back(num % 10);
            num = num / 10;
        }
        nums.push_back(num);
        reverse(nums.begin(), nums.end());

        int n = nums.size();
        vector<int> dp(n, 0);
        dp[0] = 1;
        if(n >= 2)
            if(nums[1] + (nums[0] * 10) < 26 && nums[0] != 0) dp[1] = 2;
            else dp[1] = 1;
        for(int i = 2; i < n; i++){
            
            if(nums[i] + (nums[i - 1] * 10) < 26 && nums[i - 1] != 0) dp[i] = dp[i - 1] + dp[i - 2];
            else dp[i] = dp[i - 1];
        }
        return dp[n - 1];
    }
};
```

48 请从字符串中找出一个最长的不包含重复字符的子字符串，计算该最长子字符串的长度。

 

示例 1:

输入: "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
示例 2:

输入: "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
示例 3:

输入: "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。


提示：

s.length <= 40000
注意：本题与主站 3 题相同：https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/zui-chang-bu-han-zhong-fu-zi-fu-de-zi-zi-fu-chuan-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        //左右指针，left, right, right后移，构造一个窗口字典，如果right所指
        //字符在窗口中存在，left指针移向该字符下一位，字典应该保存字符和其对应的位置
        if(s.size() == 0) return 0;
        if(s.size() == 1) return 1;
        int left = 0, right = 1;
        int res = 1;
        unordered_map<char, int> window;
        window[s[0]] = 0;
        int window_size = 0;
        while(right < s.size()){
            char c = s[right];
            //cout << c << right << left << endl;
            if(window.count(c) && window[c] >= left){
                left = window[c] + 1;
            }
            window[c] = right;
            window_size = right - left + 1;
            res = res > window_size ? res : window_size;
            right++;
            
        }
        return res;
    }
};
```

18 给定单向链表的头指针和一个要删除的节点的值，定义一个函数删除该节点。

返回删除后的链表的头节点。

注意：此题对比原题有改动

示例 1:

输入: head = [4,5,1,9], val = 5
输出: [4,1,9]
解释: 给定你链表中值为 5 的第二个节点，那么在调用了你的函数之后，该链表应变为 4 -> 1 -> 9.
示例 2:

输入: head = [4,5,1,9], val = 1
输出: [4,5,9]
解释: 给定你链表中值为 1 的第三个节点，那么在调用了你的函数之后，该链表应变为 4 -> 5 -> 9.


说明：

题目保证链表中节点的值互不相同
若使用 C 或 C++ 语言，你不需要 free 或 delete 被删除的节点

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/shan-chu-lian-biao-de-jie-dian-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    ListNode* deleteNode(ListNode* head, int val) {
        //遍历链表找到要删除的节点node，node的前一个节点指向node的后一个节点
        if(head == NULL) return NULL;
        if(head->next == NULL) return NULL;
        if(head->val == val) return head->next;
        ListNode* cur_node = head;
        ListNode* pre_node = head;
        while(cur_node != NULL){
            if(cur_node->val == val){
                pre_node->next = cur_node->next;
                return head;
            }
            else{
                pre_node = cur_node;
                cur_node = cur_node->next;
            }
        }
        return head;
    }
};
```

22 输入一个链表，输出该链表中倒数第k个节点。为了符合大多数人的习惯，本题从1开始计数，即链表的尾节点是倒数第1个节点。

例如，一个链表有 6 个节点，从头节点开始，它们的值依次是 1、2、3、4、5、6。这个链表的倒数第 3 个节点是值为 4 的节点。

 

示例：

给定一个链表: 1->2->3->4->5, 和 k = 2.

返回链表 4->5.

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    ListNode* getKthFromEnd(ListNode* head, int k) {
        //遍历链表，统计节点数
        //再遍历链表，返回倒数第k个
        //1->2->3->4->5，k=2,倒数k=2个的头节点是第四个，n - k + 1
        int n = 0;
        ListNode* node = head;
        while(node != NULL){
            n++;
            node = node->next;
        }
        node = head;
        int i = 1;
        while(i < n - k + 1){
            node = node->next;
            i++;
        }
        return node;
        
    }
};

class Solution {
public:
    ListNode* getKthFromEnd(ListNode* head, int k) {
        //遍历链表，统计节点数
        //返回倒数第k个
        //1->2->3->4->5，k=2,倒数k=2个的头节点是第四个，n - k + 1
        int n = 0;
        vector<ListNode*> list;
        ListNode* node = head;
        while(node != NULL){
            n++;
            list.push_back(node);
            node = node->next;
        }
        return list[n - k];
        
    }
};
```

25 输入两个递增排序的链表，合并这两个链表并使新链表中的节点仍然是递增排序的。

示例1：

输入：1->2->4, 1->3->4
输出：1->1->2->3->4->4
限制：

0 <= 链表长度 <= 1000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/he-bing-liang-ge-pai-xu-de-lian-biao-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        if(!l1) return l2;
        if(!l2) return l1;
        ListNode* root = new ListNode(0);
        ListNode* node = root;
        while(l1 && l2){
            if(l1->val < l2->val){
                node->next = l1;
                node = node->next;
                l1 = l1->next;
            }
            else{
                node->next = l2;
                node = node->next;
                l2 = l2->next;
            }
        }
        if(l1) node->next = l1;
        if(l2) node->next = l2;
        return root->next;
    }
};
```

52 输入两个链表，找出它们的第一个公共节点。

如下面的两个链表：



在节点 c1 开始相交。

 

示例 1：



输入：intersectVal = 8, listA = [4,1,8,4,5], listB = [5,0,1,8,4,5], skipA = 2, skipB = 3
输出：Reference of the node with value = 8
输入解释：相交节点的值为 8 （注意，如果两个列表相交则不能为 0）。从各自的表头开始算起，链表 A 为 [4,1,8,4,5]，链表 B 为 [5,0,1,8,4,5]。在 A 中，相交节点前有 2 个节点；在 B 中，相交节点前有 3 个节点。


示例 2：



输入：intersectVal = 2, listA = [0,9,1,2,4], listB = [3,2,4], skipA = 3, skipB = 1
输出：Reference of the node with value = 2
输入解释：相交节点的值为 2 （注意，如果两个列表相交则不能为 0）。从各自的表头开始算起，链表 A 为 [0,9,1,2,4]，链表 B 为 [3,2,4]。在 A 中，相交节点前有 3 个节点；在 B 中，相交节点前有 1 个节点。


示例 3：



输入：intersectVal = 0, listA = [2,6,4], listB = [1,5], skipA = 3, skipB = 2
输出：null
输入解释：从各自的表头开始算起，链表 A 为 [2,6,4]，链表 B 为 [1,5]。由于这两个链表不相交，所以 intersectVal 必须为 0，而 skipA 和 skipB 可以是任意值。
解释：这两个链表不相交，因此返回 null。


注意：

如果两个链表没有交点，返回 null.
在返回结果后，两个链表仍须保持原有的结构。
可假定整个链表结构中没有循环。
程序尽量满足 O(n) 时间复杂度，且仅用 O(1) 内存。
本题与主站 160 题相同：https://leetcode-cn.com/problems/intersection-of-two-linked-lists/

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/liang-ge-lian-biao-de-di-yi-ge-gong-gong-jie-dian-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        ListNode* A = headA;
        ListNode* B = headB;
        unordered_map<ListNode*, int> nodes;
        while(A){
            nodes[A]++;
            A = A->next;
        }
        while(B){
            if(nodes.count(B)) return B;
            B = B->next;
        }
        return NULL;
    }
};
```

21 输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有奇数在数组的前半部分，所有偶数在数组的后半部分。

 

示例：

输入：nums = [1,2,3,4]
输出：[1,3,2,4] 
注：[3,1,2,4] 也是正确的答案之一。


提示：

0 <= nums.length <= 50000
0 <= nums[i] <= 10000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    vector<int> exchange(vector<int>& nums) {
        //使用一个新的数组保存结果
        //遍历原数组，如果是奇数，则放在前面，如果是偶数则放在后面
        //一个指针指向前面的位置，一个指针指向后面的位置
        int n = nums.size();
        if(n == 0) return {};
        vector<int> res(n, 0);
        int left = 0, right = n - 1;
        //遍历数组
        for(int i = 0; i < n; i++){
            //奇数从左边开始存，偶数从右边开始存
            if(nums[i] % 2 == 0)
                res[right--] = nums[i];
            else 
                res[left++] = nums[i];
        }
        return res;
    }
};
```

57 输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。如果有多对数字的和等于s，则输出任意一对即可。

 

示例 1：

输入：nums = [2,7,11,15], target = 9
输出：[2,7] 或者 [7,2]
示例 2：

输入：nums = [10,26,30,31,47,60], target = 40
输出：[10,30] 或者 [30,10]


限制：

1 <= nums.length <= 10^5
1 <= nums[i] <= 10^6

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/he-wei-sde-liang-ge-shu-zi-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        //遍历数组，找到两个数之和等于target输出这两数
        //顺序递增，从数组两端开始遍历，nums[left] + nums[right] = sum
        // if sum == target ,输出这两个数
        // if sum < target, nums[right] 是右边最大的，所以左边的指针应该右移
        // if sum > target, 说明和过大，右指针左移
        int n = nums.size();
        if(n == 1) return {};
        int left = 0, right = n - 1;
        while(left < right){//相等时不满足条件，只是一个数不是两个数
            if(nums[left] + nums[right] == target) return vector<int>{nums[left], nums[right]};
            else if(nums[left] + nums[right] < target) left++;
            else if(nums[left] + nums[right] > target) right--;
        }
        return {};
    }
};
```

58 输入一个英文句子，翻转句子中单词的顺序，但单词内字符的顺序不变。为简单起见，标点符号和普通字母一样处理。例如输入字符串"I am a student. "，则输出"student. a am I"。

 

示例 1：

输入: "the sky is blue"
输出: "blue is sky the"
示例 2：

输入: "  hello world!  "
输出: "world! hello"
解释: 输入字符串可以在前面或者后面包含多余的空格，但是反转后的字符不能包括。
示例 3：

输入: "a good   example"
输出: "example good a"
解释: 如果两个单词间有多余的空格，将反转后单词间的空格减少到只含一个。


说明：

无空格字符构成一个单词。
输入字符串可以在前面或者后面包含多余的空格，但是反转后的字符不能包括。
如果两个单词间有多余的空格，将反转后单词间的空格减少到只含一个。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/fan-zhuan-dan-ci-shun-xu-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    string reverseWords(string s) {
        //遍历字符串，分词，遇空格为一个词
        //使用一个数组存储， 反向连接
        //用两个指针，fast, slow, fast一直往前走，slow保存单词的起始位置
        //遇到空格，只保留一个
        int n = s.size();
        if(n == 0) return "";
        string res = "";
        int slow = 0, fast = 0;
        //字符串前面可能会有空格，需要跳过
        while(s[fast] == ' ') fast++;
        slow = fast;
        vector<string> words;
        while(fast < n){
            if(s[fast] != ' ') fast++;
            else if(s[fast] == ' '){//遇到空格
                words.push_back(s.substr(slow, fast - slow));
                //字符串中间可能会有空格，此时需要跳过
                while(fast < n && s[fast] == ' ') fast++;
                slow = fast;
            }
        }
        if(fast > slow) words.push_back(s.substr(slow, fast - slow));
        for(int i = words.size() - 1; i >= 0; i--)
            res += words[i] + " ";
        //去掉最后一个空格
        return res.substr(0, res.size() - 1);
    }
};
```

12 给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。

单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

 

例如，在下面的 3×4 的矩阵中包含单词 "ABCCED"（单词中的字母已标出）。



 

示例 1：

输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
输出：true
示例 2：

输入：board = [["a","b"],["c","d"]], word = "abcd"
输出：false


提示：

1 <= board.length <= 200
1 <= board[i].length <= 200
board 和 word 仅由大小写英文字母组成


注意：本题与主站 79 题相同：https://leetcode-cn.com/problems/word-search/

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/ju-zhen-zhong-de-lu-jing-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    bool exist(vector<vector<char>>& board, string word) {
        rows = board.size();
        cols = board[0].size();
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                if(dfs(board, word, i, j, 0)) return true;
            }
        }
        return false;
    }
private:
    int rows, cols;
    bool dfs(vector<vector<char>>& board, string word, int i, int j, int k) {
        if(i >= rows || i < 0 || j >= cols || j < 0 || board[i][j] != word[k]) return false;
        if(k == word.size() - 1) return true;
        board[i][j] = '\0';
        bool res = dfs(board, word, i + 1, j, k + 1) || dfs(board, word, i - 1, j, k + 1) || 
                      dfs(board, word, i, j + 1, k + 1) || dfs(board, word, i , j - 1, k + 1);
        board[i][j] = word[k];
        return res;
    }
};

```

13 地上有一个m行n列的方格，从坐标 [0,0] 到坐标 [m-1,n-1] 。一个机器人从坐标 [0, 0] 的格子开始移动，它每次可以向左、右、上、下移动一格（不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。例如，当k为18时，机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。但它不能进入方格 [35, 38]，因为3+5+3+8=19。请问该机器人能够到达多少个格子？

 

示例 1：

输入：m = 2, n = 3, k = 1
输出：3
示例 2：

输入：m = 3, n = 1, k = 0
输出：1
提示：

1 <= n,m <= 100
0 <= k <= 20

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/ji-qi-ren-de-yun-dong-fan-wei-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    int movingCount(int m, int n, int k) {
        // 假设当前位置为i, j, 下一步可以是上（i - 1, j）下(i + 1, j)左(i, j - 1)右(i, j + 1)
        // 如果下一步的位置越界，退回
        // 如果下一步的数位之和大于k，退回
        // 如果下一步已经走过，退回
        // 否则，下一步可以走，结果+1
        // 越界： i < 0 || j < 0 || i >= m || j >= n
        // 数位之和：任意一个整数，模10取得个位，除以10取得除个位之外的。如果除以10结果为0，说明只有一位数了。while(num)： 个位 = num % 10, 其它 = num / 10，num = 其它
        // 下一步是否走过？用一个m * n的矩阵保存
        vector<vector<bool>> visited(m, vector<bool>(n, false));
        return search(0, 0, m, n, k, visited);

    }
    int search(int i, int j, int m, int n, int k, vector<vector<bool>>& visited){
        if(i < 0 || j < 0 || i >= m || j >= n) return 0;
        int sum = 0;
        int num = i;
        while(num){
            sum += num % 10;
            num = num / 10;
        }
        num = j;
        while(num){
            sum += num % 10;
            num = num / 10;
        }
        if(sum > k) return 0;
        if(visited[i][j]) return 0;
        visited[i][j] = true;
        return 1 + search(i - 1, j, m, n, k, visited)
                 + search(i + 1, j, m, n, k, visited)
                 + search(i, j - 1, m, n, k, visited)
                 + search(i, j + 1, m, n, k, visited);
    }
};
```

34 给你二叉树的根节点 root 和一个整数目标和 targetSum ，找出所有 从根节点到叶子节点 路径总和等于给定目标和的路径。

叶子节点 是指没有子节点的节点。

 

示例 1：



输入：root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
输出：[[5,4,11,2],[5,8,4,5]]
示例 2：



输入：root = [1,2,3], targetSum = 5
输出：[]
示例 3：

输入：root = [1,2], targetSum = 0
输出：[]

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/er-cha-shu-zhong-he-wei-mou-yi-zhi-de-lu-jing-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    vector<vector<int>> pathSum(TreeNode* root, int target) {
        // 从根节点开始遍历，累加节点值，将节点值保存到数组中，如果到叶子节点和等于目标值，保存该数组
        // 假设当前节点为node，路径为track，如果sum(track) + node->val = target && node->left = null && node->right = null, val加入到track中，track保存到结果中，回退
        // 如果sum(track) + node->val < target, val 加入到track中，遍历node的左右子树
        // 如果sum(track) + node->val > target, 说明这条路径不满足，退回到上一个节点
        // 有负数，上面两个条件合二为一，sum(track) + node->val ！= target val 加入到track中，遍历node的左右子树
        vector<vector<int>> res;
        vector<int> track;
        int track_sum = 0;
        trackBack(root, target, track_sum, track, res);
        return res;
    }
    void trackBack(TreeNode* node, int target, int track_sum, vector<int>& track,          vector<vector<int>>& res){
        if(node == NULL) return;
        int node_val = node->val;
        track_sum += node_val;
        if(track_sum == target && node->left == NULL && node->right == NULL){
            track.push_back(node_val);
            res.push_back(track);
            //回退
            //sum为值传递，不用管
            //track回退
            track.pop_back();
            return;
        }
        else {
            //sum < target
            //val 加入到track
            //sum += val
            //遍历左右子树
            track.push_back(node_val);
            trackBack(node->left, target, track_sum, track, res);
            trackBack(node->right, target, track_sum, track, res);
            //回退
            track.pop_back();
            return;
        }
    }
};
```

36 输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的循环双向链表。要求不能创建任何新的节点，只能调整树中节点指针的指向。

 

为了让您更好地理解问题，以下面的二叉搜索树为例：

 



 

我们希望将这个二叉搜索树转化为双向循环链表。链表中的每个节点都有一个前驱和后继指针。对于双向循环链表，第一个节点的前驱是最后一个节点，最后一个节点的后继是第一个节点。

下图展示了上面的二叉搜索树转化成的链表。“head” 表示指向链表中有最小元素的节点。

 



 

特别地，我们希望可以就地完成转换操作。当转化完成以后，树中节点的左指针需要指向前驱，树中节点的右指针需要指向后继。还需要返回链表中的第一个节点的指针。

 

注意：本题与主站 426 题相同：https://leetcode-cn.com/problems/convert-binary-search-tree-to-sorted-doubly-linked-list/

注意：此题对比原题有改动。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
/*
// Definition for a Node.
class Node {
public:
    int val;
    Node* left;
    Node* right;

    Node() {}

    Node(int _val) {
        val = _val;
        left = NULL;
        right = NULL;
    }

    Node(int _val, Node* _left, Node* _right) {
        val = _val;
        left = _left;
        right = _right;
    }
};
*/
class Solution {
public:
    Node* treeToDoublyList(Node* root) {
        //二叉搜索树任意一个节点其左子树上所有节点小于该节点，右子树上所有节点大于该节点，对其进行中序遍历能够获得完整的数组，且是由小到大
        //非原地操作可以使用一个额外的容器保存节点，但是原地操作怎么搞？
        if(root == NULL) return NULL;
        vector<Node*> nodes;
        inorder(root, nodes);
        Node* head = nodes[0];
        Node* pre = head;
        for(int i = 1; i < nodes.size(); ++i){
            Node* cur = nodes[i];
            pre->right = cur;
            cur->left = pre;
            pre = cur;
        }
        pre->right = head;
        head->left = pre;
        return head;
    }
    void inorder(Node* root, vector<Node*>& nodes){
        if(root == NULL) return;
        inorder(root->left, nodes);
        nodes.push_back(root);
        inorder(root->right, nodes);
        return;
    }
};

/*
// Definition for a Node.
class Node {
public:
    int val;
    Node* left;
    Node* right;

    Node() {}

    Node(int _val) {
        val = _val;
        left = NULL;
        right = NULL;
    }

    Node(int _val, Node* _left, Node* _right) {
        val = _val;
        left = _left;
        right = _right;
    }
};
*/
class Solution {
public:
    Node* pre = NULL, *head = NULL;
    Node* treeToDoublyList(Node* root) {
        //二叉搜索树任意一个节点其左子树上所有节点小于该节点，右子树上所有节点大于该节点，对其进行中序遍历能够获得完整的数组，且是由小到大
        //非原地操作可以使用一个额外的容器保存节点，但是原地操作怎么搞？
        if(root == NULL) return NULL;
        inorder(root);
        head->left = pre;
        pre->right = head;
        return head;
        
    }
    void inorder(Node* root){
        if(root == NULL) return;
        inorder(root->left);
        if(pre) pre->right = root;
        else head = root;
        root->left = pre;
        pre = root;
        inorder(root->right);
        return;
    }
};
```

54 给定一棵二叉搜索树，请找出其中第 k 大的节点的值。

 

示例 1:

输入: root = [3,1,4,null,2], k = 1
   3
  / \
 1   4
  \
   2
输出: 4
示例 2:

输入: root = [5,3,6,2,4,null,null,1], k = 3
       5
      / \
     3   6
    / \
   2   4
  /
 1
输出: 4


限制：

1 ≤ k ≤ 二叉搜索树元素个数

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    int flag = 0, val = 0;
    int kthLargest(TreeNode* root, int k) {
        //中序遍历，每遍历一个节点，flag + 1,直到flag = k
        inoder(root, k);
        return val;

    }
    void inoder(TreeNode* root, int k){
        if(root == NULL) return;
        inoder(root->right, k);
        flag++;
        if(flag == k) {
            val = root->val;
            return;
        }
        inoder(root->left, k);
    }
};
```

45 输入一个非负整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。

 

示例 1:

输入: [10,2]
输出: "102"
示例 2:

输入: [3,30,34,5,9]
输出: "3033459"


提示:

0 < nums.length <= 100
说明:

输出结果可能非常大，所以你需要返回一个字符串而不是整数
拼接起来的数字可能会有前导 0，最后结果不需要去掉前导 0

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    string minNumber(vector<int>& nums) {
        vector<string> strs;
        string res;
        for(int i = 0; i < nums.size(); i++)
            strs.push_back(to_string(nums[i]));
        sort(strs.begin(), strs.end(), [](string& x, string& y){ return x + y < y + x; });
        for(int i = 0; i < strs.size(); i++)
            res.append(strs[i]);
        return res;
    }
};


class Solution {
public:
    string minNumber(vector<int>& nums) {
        // 快速排序
        int len = nums.size();
        vector<string> ans;
        for (auto num : nums) ans.push_back(to_string(num));
        sort(ans, 0, len - 1);
        string ret = "";
        for (auto str : ans) ret += str;
        return ret;
    }

    void sort(vector<string>& ans, int left, int right){
        if (right <= left) return;
        int j = partition(ans, left, right);
        sort(ans, left, j - 1);
        sort(ans, j + 1, right);
    }
    
    // 切分
    int partition(vector<string>& ans, int left, int right){
        int i = left, j = right;
        string pivot = ans[left];
        while (true){
            while (ans[i] + pivot <= pivot + ans[i]){
                if (++i > j) break;
            }
            while (pivot + ans[j] < ans[j] + pivot){
                if (--j < i) break;
            }
            if (i >= j) break;
            string tmp = ans[i];
            ans[i] = ans[j];
            ans[j] = tmp;
        }
        string tmp = ans[left];
        ans[left] = ans[j];
        ans[j] = tmp;
        return j;
    }
};

```

61 从若干副扑克牌中随机抽 5 张牌，判断是不是一个顺子，即这5张牌是不是连续的。2～10为数字本身，A为1，J为11，Q为12，K为13，而大、小王为 0 ，可以看成任意数字。A 不能视为 14。

 

示例 1:

输入: [1,2,3,4,5]
输出: True


示例 2:

输入: [0,0,1,2,5]
输出: True


限制：

数组长度为 5 

数组的数取值为 [0, 13] .

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/bu-ke-pai-zhong-de-shun-zi-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    bool isStraight(vector<int>& nums) {
        //1先从小到大排序
        //2统计0的个数
        //3从第一个不为零的开始，假设为i, i 和 i + 1连续，说明nums[i] + 1 = nums[i + 1]
        //4如果nums[i + 1] - nums[i] - 1 <= count_0,说明可以用0代替中间数进行拼接。此时，0的可用数量要减去nums[i + 1] - nums[i] - 1
        //5如果nums[i + 1] = nums[i] ！= 0如[0,0,2,2,5]则不是一个顺子
        //3和4可以合在一起

        sort(nums.begin(), nums.end());
        int count_0 = 0;
        for(int i = 0; i < 5; ++i)
            if(nums[i] == 0) count_0++;
        //第一个不为0的数从i = count_0 开始
        for(int i = count_0; i < 4; ++i){
            if(nums[i] == nums[i + 1]) return false;
            else {
                int temp = nums[i + 1] - nums[i] - 1;
                if(temp <= count_0){
                    count_0 -= temp;
                    continue;
                }
                else return false;
            }
        }
        return true;
    }
};
```

40 输入整数数组 arr ，找出其中最小的 k 个数。例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4。

 

示例 1：

输入：arr = [3,2,1], k = 2
输出：[1,2] 或者 [2,1]
示例 2：

输入：arr = [0,1,2,1], k = 1
输出：[0]


限制：

0 <= k <= arr.length <= 10000
0 <= arr[i] <= 10000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    vector<int> getLeastNumbers(vector<int>& arr, int k) {
        //先排序，再取最小的k个
        sort(arr.begin(), arr.end());
        vector<int> res;
        for(int i = 0; i < k; ++i)
            res.push_back(arr[i]);
        return res;
    }
};
```

55 输入一棵二叉树的根节点，求该树的深度。从根节点到叶节点依次经过的节点（含根、叶节点）形成树的一条路径，最长路径的长度为树的深度。

例如：

给定二叉树 [3,9,20,null,null,15,7]，

    3
   / \
  9  20
    /  \
   15   7
返回它的最大深度 3 。

 

提示：

节点总数 <= 10000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/er-cha-shu-de-shen-du-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    int maxDepth(TreeNode* root) {
        //对于根节点root，其左子树和右子树的最大深度+1就是总的最大深度
        //如果root为空，返回0
        //如果root不为空，返回root 左右子树的最大值+1
        //if(root == NULL) return 0;
        //else return 1 + max(maxDepth(root->left), maxDepth(root->right));
        //广度优先搜索，直到队列为空
        if(root == NULL) return 0;
        queue<TreeNode*> q;
        q.push(root);
        int res = 0;
        while(!q.empty()){
            int sz = q.size();
            for(int i = 0; i < sz; ++i){
                TreeNode* node = q.front();
                q.pop();
                if(node->left) q.push(node->left);
                if(node->right) q.push(node->right);
            }
            res++;
        }
        return res;
    }
};


/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    int maxDepth(TreeNode* root) {
        //对于根节点root，其左子树和右子树的最大深度+1就是总的最大深度
        //如果root为空，返回0
        //如果root不为空，返回root 左右子树的最大值+1
        if(root == NULL) return 0;
        else return 1 + max(maxDepth(root->left), maxDepth(root->right));
    }
};
```

55 输入一棵二叉树的根节点，判断该树是不是平衡二叉树。如果某二叉树中任意节点的左右子树的深度相差不超过1，那么它就是一棵平衡二叉树。

 

示例 1:

给定二叉树 [3,9,20,null,null,15,7]

    3
   / \
  9  20
    /  \
   15   7
返回 true 。

示例 2:

给定二叉树 [1,2,2,3,3,null,null,4,4]

       1
      / \
     2   2
    / \
   3   3
  / \
 4   4
返回 false 。

 

限制：

0 <= 树的结点个数 <= 10000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/ping-heng-er-cha-shu-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    bool isBalanced(TreeNode* root) {
        //空节点是平衡二叉树吗？假设空节点是平衡二叉树
        //如果root为空返回true
        //如果root不为空：
        //1 获得左右子树的深度，计算左右子树的深度差
        // 2 如果深度差 <= 1 并且左右子树都满足平衡二叉树的条件，那么这棵树就是平衡二叉树
        // 如果根节点的左右子树深度差超过1，或者左子树或者右子树不是平衡二叉树，那么这棵树就不是平衡二叉树
        if(root == NULL) return true;
        bool left = isBalanced(root->left);
        bool right = isBalanced(root->right);
        if(!left || !right) return false;
        int left_deepth = deepth(root->left);
        int right_deepth = deepth(root->right);
        if(left_deepth - right_deepth <= 1 && left_deepth - right_deepth >= -1) return true;
        else return false;
    }
    int deepth(TreeNode* root){
        if(root == NULL) return 0;
        else return 1 + max(deepth(root->left), deepth(root->right));
    }
};
```

64 求 1+2+...+n ，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。

 

示例 1：

输入: n = 3
输出: 6
示例 2：

输入: n = 9
输出: 45


限制：

1 <= n <= 10000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/qiu-12n-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

$$1+2+...+n=\frac{n\times(n+1)}{2}$$

```c++
class Solution {
public:
    int sumNums(int n) {
        bool a[n][n + 1];
        return sizeof(a) >> 1;
    }
};
```

68 给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

例如，给定如下二叉搜索树:  root = [6,2,8,0,4,7,9,null,null,3,5]



 

示例 1:

输入: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
输出: 6 
解释: 节点 2 和节点 8 的最近公共祖先是 6。
示例 2:

输入: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4
输出: 2
解释: 节点 2 和节点 4 的最近公共祖先是 2, 因为根据定义最近公共祖先节点可以为节点本身。


说明:

所有节点的值都是唯一的。
p、q 为不同节点且均存在于给定的二叉搜索树中。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-zui-jin-gong-gong-zu-xian-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:    
    bool find_p = false, find_q = false;
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        //看父节点是不是同一个节点，或者一个节点是不是另一个节点的父节点
        //如果一个节点是另一个节点的父节点，那么这个节点就是返回值
        //如果一个节点不是另一个节点的父节点，那就看他们的父节点满不满足关系。
        //首先是要找到这两个节点，然后是他们的父节点
        //可以使用两个数组保存搜索路径,搜索到两个节点后，找到最后面的那一个公共节点就可以了
        vector<TreeNode*> track_p;
        vector<TreeNode*> track_q;
        find(root, p, track_p);
        find(root, q, track_q);
        int n = min(track_q.size(), track_p.size());
        int i = 0;
        for(; i < n; ++i){
            if(track_p[i] != track_q[i]) break;
        }
        return track_p[i - 1];

    }
    void find(TreeNode* root, TreeNode* node, vector<TreeNode*>& track){
        if(root == NULL) return;
        track.push_back(root);
        if(node->val == root->val) return;
        if(node->val > root->val) find(root->right, node, track);
        if(node->val < root->val) find(root->left, node, track);
        return;
    }
   
};


/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:    
    bool find_p = false, find_q = false;
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        //二叉搜索树，所有值唯一
        //如果节点root的值>p && > q,说明p,q都在root的左子树，root < p, q说明p, q都在root的右子树，否则，说明一个比root大一个比root小，root就是公共祖先。
        while(true){
            if(root->val > p->val && root->val > q->val) root = root->left;
            else if(root->val < p->val && root->val < q->val) root = root->right;
            else return root;
        }
        return NULL;
        

    }
   
};
```

68 给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

例如，给定如下二叉树:  root = [3,5,1,6,2,0,8,null,null,7,4]



 

示例 1:

输入: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
输出: 3
解释: 节点 5 和节点 1 的最近公共祖先是节点 3。
示例 2:

输入: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
输出: 5
解释: 节点 5 和节点 4 的最近公共祖先是节点 5。因为根据定义最近公共祖先节点可以为节点本身。


说明:

所有节点的值都是唯一的。
p、q 为不同节点且均存在于给定的二叉树中。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/er-cha-shu-de-zui-jin-gong-gong-zu-xian-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    bool flag = false;
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        //看父节点是不是同一个节点，或者一个节点是不是另一个节点的父节点
        //如果一个节点是另一个节点的父节点，那么这个节点就是返回值
        //如果一个节点不是另一个节点的父节点，那就看他们的父节点满不满足关系。
        //首先是要找到这两个节点，然后是他们的父节点
        //可以使用两个数组保存搜索路径,搜索到两个节点后，找到最后面的那一个公共节点就可以了
        vector<TreeNode*> track1;
        vector<vector<TreeNode*>> track_p;
        vector<vector<TreeNode*>> track_q;
        find(root, p, track1, track_p);
        vector<TreeNode*> track2;
        flag = false;
        find(root, q, track2, track_q);
        int n = min(track_q[0].size(), track_p[0].size());
        int i = 0;
        for(; i < n; ++i){
            if(track_p[0][i] != track_q[0][i]) break;
        }
        return track_p[0][i - 1];
    }
    void find(TreeNode* root, TreeNode* node, vector<TreeNode*>& track, vector<vector<TreeNode*>>& res){
        if(root == NULL) return;
        track.push_back(root);
        if(node->val == root->val) {
            res.push_back(track);
            flag = true;
            return;

        };
        find(root->left, node, track, res);
        if(flag) return;
        find(root->right, node, track, res);
        if(flag) return;
        track.pop_back();
        return;
    }
};
```

7 输入某二叉树的前序遍历和中序遍历的结果，请构建该二叉树并返回其根节点。

假设输入的前序遍历和中序遍历的结果中都不含重复的数字。

 

示例 1:


Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
Output: [3,9,20,null,null,15,7]
示例 2:

Input: preorder = [-1], inorder = [-1]
Output: [-1]


限制：

0 <= 节点个数 <= 5000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/zhong-jian-er-cha-shu-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
private:
    unordered_map<int, int> index;

public:
    TreeNode* myBuildTree(const vector<int>& preorder, const vector<int>& inorder,int preorder_left, int preorder_right, int inorder_left, int inorder_right){
        if(preorder_left > preorder_right) return nullptr;
        int inorder_root = index[preorder[preorder_left]];
        TreeNode* root = new TreeNode(preorder[preorder_left]);
        int size_left_subtree = inorder_root - inorder_left;
        root->left = myBuildTree(preorder, inorder, preorder_left + 1, preorder_left + size_left_subtree, inorder_left, inorder_root - 1);
        root->right = myBuildTree(preorder, inorder, preorder_left + size_left_subtree + 1, preorder_right, inorder_root + 1, inorder_right);
        return root;


    }
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        int n = preorder.size();
        for(int i = 0; i < n; ++i)
            index[inorder[i]] = i;
        return myBuildTree(preorder, inorder, 0, n - 1, 0, n - 1);
    }
};
```

16 实现 pow(x, n) ，即计算 x 的 n 次幂函数（即，xn）。不得使用库函数，同时不需要考虑大数问题。

 

示例 1：

输入：x = 2.00000, n = 10
输出：1024.00000
示例 2：

输入：x = 2.10000, n = 3
输出：9.26100
示例 3：

输入：x = 2.00000, n = -2
输出：0.25000
解释：2-2 = 1/22 = 1/4 = 0.25


提示：

-100.0 < x < 100.0
-231 <= n <= 231-1
-104 <= xn <= 104

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/shu-zhi-de-zheng-shu-ci-fang-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    double myPow(double x, int n) {
        long long N = n;
        return N > 0 ? mul(x, N) : 1 / mul(x, -N);
    }
    double mul(double x, long long n){
        if(n == 0) return 1;
        if(n == 1) return x;
        double res = 1;
        res =  myPow(x, n / 2);
        return n % 2 == 0 ? res * res : res * res * x;
    }
};
```

33 输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。如果是则返回 true，否则返回 false。假设输入的数组的任意两个数字都互不相同。

 

参考以下这颗二叉搜索树：

     5
    / \
   2   6
  / \
 1   3
示例 1：

输入: [1,6,3,2,5]
输出: false
示例 2：

输入: [1,3,2,6,5]
输出: true


提示：

数组长度 <= 1000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    bool verifyPostorder(vector<int>& postorder) {
        //二叉搜索树的后续遍历
        return check(postorder, 0, postorder.size() - 1);
    }
    bool check(vector<int>& postorder, int left, int right){
        if(left >= right) return true;
        //根节点
        int root = postorder[right];
        int left_end = left; //找到左子树的结束位置
        while(postorder[left_end] < root) left_end++;
        //循环停止条件为postorder[left_end] >= root, left_end - 1才是左子树的最后一个取值
        //判断左子树中是不是所有的值都小于root,并且右子树中是不是所有的值都大于root
        //while(postorder[left_end++] < root);保证left_end左边都小于root，只需要判断右边
        int right_start = left_end;
        while(postorder[right_start] > root) right_start++;
        //循环停止条件为postorder[left_end] <= root, 此时right_start指向root
        return right_start == right 
               && check(postorder, left, left_end - 1) 
               && check(postorder, left_end, right - 1);
    }
};
```

15 编写一个函数，输入是一个无符号整数（以二进制串的形式），返回其二进制表达式中数字位数为 '1' 的个数（也被称为 汉明重量).）。

 

提示：

请注意，在某些语言（如 Java）中，没有无符号整数类型。在这种情况下，输入和输出都将被指定为有符号整数类型，并且不应影响您的实现，因为无论整数是有符号的还是无符号的，其内部的二进制表示形式都是相同的。
在 Java 中，编译器使用 二进制补码 记法来表示有符号整数。因此，在上面的 示例 3 中，输入表示有符号整数 -3。


示例 1：

输入：n = 11 (控制台输入 00000000000000000000000000001011)
输出：3
解释：输入的二进制串 00000000000000000000000000001011 中，共有三位为 '1'。
示例 2：

输入：n = 128 (控制台输入 00000000000000000000000010000000)
输出：1
解释：输入的二进制串 00000000000000000000000010000000 中，共有一位为 '1'。
示例 3：

输入：n = 4294967293 (控制台输入 11111111111111111111111111111101，部分语言中 n = -3）
输出：31
解释：输入的二进制串 11111111111111111111111111111101 中，共有 31 位为 '1'。


提示：

输入必须是长度为 32 的 二进制串 。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/er-jin-zhi-zhong-1de-ge-shu-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    int hammingWeight(uint32_t n) {
        int res = 0;
        while(n){
            res += n & 1;
            n = n >> 1;
        }
        return res;
    }
};
```

65 写一个函数，求两个整数之和，要求在函数体内不得使用 “+”、“-”、“*”、“/” 四则运算符号。

 

示例:

输入: a = 1, b = 1
输出: 2


提示：

a, b 均可能是负数或 0
结果不会溢出 32 位整数

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/bu-yong-jia-jian-cheng-chu-zuo-jia-fa-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    int add(int a, int b) {
        while (b) {
            int carry = a & b; // 计算 进位
            a = a ^ b; // 计算 本位
            b = (unsigned)carry << 1;
        }
        return a;
    }
};

```

56 一个整型数组 nums 里除两个数字之外，其他数字都出现了两次。请写程序找出这两个只出现一次的数字。要求时间复杂度是O(n)，空间复杂度是O(1)。

 

示例 1：

输入：nums = [4,1,4,6]
输出：[1,6] 或 [6,1]
示例 2：

输入：nums = [1,2,10,4,1,4,3,3]
输出：[2,10] 或 [10,2]


限制：

2 <= nums.length <= 10000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    vector<int> singleNumbers(vector<int>& nums) {
        //可以对每个数计数，然后输出计数值为1的那两个数。
        unordered_map<int, int> count;
        int n = nums.size();
        for(int i = 0; i < n; ++i){
            count[nums[i]]++;
        }
        vector<int> res;
        for(int i = 0; i < n; ++i){
            if(count[nums[i]] == 1) res.push_back(nums[i]);
        }
        return res;
    }
};

//看不懂的解法
class Solution {
public:
    vector<int> singleNumbers(vector<int>& nums) {
        int ret = 0;
        for (int n : nums)
            ret ^= n;
        int div = 1;
        while ((div & ret) == 0)
            div <<= 1;
        int a = 0, b = 0;
        for (int n : nums)
            if (div & n)
                a ^= n;
            else
                b ^= n;
        return vector<int>{a, b};
    }
};

```

56 在一个数组 nums 中除一个数字只出现一次之外，其他数字都出现了三次。请找出那个只出现一次的数字。

 

示例 1：

输入：nums = [3,4,3,3]
输出：4
示例 2：

输入：nums = [9,1,7,9,7,9,7]
输出：1


限制：

1 <= nums.length <= 10000
1 <= nums[i] < 2^31

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-ii-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        //和之前那个题一样，使用一个字典来计数，最后输出值为1的那一个
        unordered_map<int, int> count;
        for(int i = 0; i < nums.size(); ++i)
            count[nums[i]]++;
        for(auto iter = count.begin(); iter != count.end(); ++iter)
            if(iter->second == 1) return iter->first;
        return 0;
    }
};
```

39 数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。

 

你可以假设数组是非空的，并且给定的数组总是存在多数元素。

 

示例 1:

输入: [1, 2, 3, 2, 2, 2, 5, 4, 2]
输出: 2


限制：

1 <= 数组长度 <= 50000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/shu-zu-zhong-chu-xian-ci-shu-chao-guo-yi-ban-de-shu-zi-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        //使用一个map计数，然后找到那个超过数组一半的元素
        unordered_map<int, int> count;
        int n = nums.size();
        for(int i = 0; i < n; ++i){
            count[nums[i]]++;
            if(count[nums[i]] > n / 2) return nums[i];
        }
        return 0;
    }
};
```

66 给定一个数组 A[0,1,…,n-1]，请构建一个数组 B[0,1,…,n-1]，其中 B[i] 的值是数组 A 中除了下标 i 以外的元素的积, 即 B[i]=A[0]×A[1]×…×A[i-1]×A[i+1]×…×A[n-1]。不能使用除法。

 

示例:

输入: [1,2,3,4,5]
输出: [120,60,40,30,24]


提示：

所有元素乘积之和不会溢出 32 位整数
a.length <= 100000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/gou-jian-cheng-ji-shu-zu-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
//在做除法时应该慎重考虑是否会有除以0的情况！！！
class Solution {
public:
    vector<int> constructArr(vector<int>& a) {
        //如果可以使用除法，那么先求所有乘积，然后除以每一个A[i],将其存到数组B中。
        //但是会遇到0.如果遇到零，那么除了这个位置，其它位置都为0。
        //如果有两个或两个以上的0，则全部为0
        int mul = 1;
        int n = a.size();
        vector<int> zero;
        for(int i = 0; i < n; ++i){
            if(a[i] == 0){
                zero.push_back(i);
                continue;
            }
            mul *= a[i];
        }

        vector<int> res(n, 0);
        if(zero.size() == 1){
            for(int i = 0; i < zero.size(); ++i)
                res[zero[i]] = mul;
            return res;
            }
        if(zero.size() > 1) return res;
        for(int i = 0; i < n; ++i)
            res[i] = mul / a[i];
        return res; 
    }
};
```

```c++
class Solution {
public:
    vector<int> constructArr(vector<int>& a) {
        //但是不可以使用除法，那就只能左边乘以右边。
        //可以用额外的两个数组保存左边和右边的积L[i] = a[0] * a[1] ... * a[i - 1]  
        // R[i] = a[i + 1] * ... * a[n - 1]
        //然后B[i] = L[i] * R[i]
        // a[0]的左边是1
        // a[n - 1]的右边是1
        // 也就是L[0] = 1, R[n - 1] = 1 
        int n = a.size();
        vector<int> left(n, 1);
        vector<int> right(n, 1);
        for(int i = 1; i < n; ++i){
            left[i] = left[i - 1] * a[i - 1];
        }
        for(int i = n - 2; i >= 0; --i){
            right[i] = right[i + 1] * a[i + 1];
        }
        vector<int> res(n, 0);
        for(int i = 0; i < n; ++i)
            res[i] = left[i] * right[i];
        return res;

    }
};
```

14 给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m-1] 。请问 k[0]*k[1]*...*k[m-1] 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

示例 1：

输入: 2
输出: 1
解释: 2 = 1 + 1, 1 × 1 = 1
示例 2:

输入: 10
输出: 36
解释: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36
提示：

2 <= n <= 58

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/jian-sheng-zi-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    int cuttingRope(int n) {
        // 1 绳子的长度为1， 积为1
        // 2 长为2的绳子分为m段的最大值积为1
        // 3 长为3的绳子分为m段的最大积为2
        // 4 长为4的绳子分为m段的最大积为max(1 * (3), 2 * (2), 3 * (1))
        // 5 长为5的绳子分为m段的最大积为max(1 * (4), 2 * (3), 3 * (2), 4 * (1))
        // 6 长为6的绳子分为m段的最大积为max(1 * (5), 2 * (4), 3 * (3), 4 * (2), 5 * (1))
        // n 长为n的绳子分为m段的最大积为max(1 * (n - 1), 2 * (n - 2),..., (n - 1) * 1)
        // 用 dp[n] 表示长为n的绳子分为m段的最大积，那么dp[n] = max(1 * dp[n - 1], 2 * dp[n - 2],...,(n - 1) * dp[1])
        //如果第一段已经确定了，那么后面的可以分也可以不分，所以dp[n] = max(1 *  max (dp[n - 1], n - 1), 2 * max(dp[n - 2], n - 2),...,(n - 1) * max(dp[1], 1))
        vector<int> dp(n + 1, 0);
        dp[1] = 1;
        dp[2] = 1;
        for(int i = 3; i <= n; ++i){
            int max = 0;
            for(int j = 1; j < i; ++j){
                int mul = j * dp[i - j];
                mul = mul > j * (i - j) ? mul : j * (i - j);
                max = mul > max ? mul : max;
            }
            dp[i] = max;
        }
        return dp[n];
    }
};
```

57 输入一个正整数 target ，输出所有和为 target 的连续正整数序列（至少含有两个数）。

序列内的数字由小到大排列，不同序列按照首个数字从小到大排列。

 

示例 1：

输入：target = 9
输出：[[2,3,4],[4,5]]
示例 2：

输入：target = 15
输出：[[1,2,3,4,5],[4,5,6],[7,8]]


限制：

1 <= target <= 10^5

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    vector<vector<int>> findContinuousSequence(int target) {
        // 双指针，滑动窗口。左右指针指向开始，右指针往右滑动，累加和
        // 如果和值小于目标值，右指针继续右移
        // 如果和值等于目标值，将子数组压入结果后，右指针右移
        // 如果和值大于目标值，右指针不懂，左指针右移，和值减去移出的值，直到和值等于或小于目标值
        // 右指针移到 target / 2 向上取整结束
        
        int left = 1, right = 0;
        int sum = 0;
        vector<vector<int>> res;
        while(right <= (target / 2 + 1)){
            if(sum == target){
                vector<int> array(right - left + 1, 0);
                for(int i = 0; i < right - left + 1; ++i)
                   array[i] = left + i;
                res.push_back(array);
                right++;
                sum += right;
            }
            if(sum < target){
                right++;
                sum += right;
            };
            if(sum > target){
                sum -= left;
                left++;
            }
        }
        return res;
    }
};
```

62 0,1,···,n-1这n个数字排成一个圆圈，从数字0开始，每次从这个圆圈里删除第m个数字（删除后从下一个数字开始计数）。求出这个圆圈里剩下的最后一个数字。

例如，0、1、2、3、4这5个数字组成一个圆圈，从数字0开始每次删除第3个数字，则删除的前4个数字依次是2、0、4、1，因此最后剩下的数字是3。

 

示例 1：

输入: n = 5, m = 3
输出: 3
示例 2：

输入: n = 10, m = 17
输出: 2


限制：

1 <= n <= 10^5
1 <= m <= 10^6

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    int lastRemaining(int n, int m) {
        int pos = 0; // 最终活下来那个人的初始位置
        for(int i = 2; i <= n; i++){
            pos = (pos + m) % i;  // 每次循环右移
        }
        return pos;
    }
};

```

31 输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如，序列 {1,2,3,4,5} 是某栈的压栈序列，序列 {4,5,3,2,1} 是该压栈序列对应的一个弹出序列，但 {4,3,5,1,2} 就不可能是该压栈序列的弹出序列。

 

示例 1：

输入：pushed = [1,2,3,4,5], popped = [4,5,3,2,1]
输出：true
解释：我们可以按以下顺序执行：
push(1), push(2), push(3), push(4), pop() -> 4,
push(5), pop() -> 5, pop() -> 3, pop() -> 2, pop() -> 1
示例 2：

输入：pushed = [1,2,3,4,5], popped = [4,3,5,1,2]
输出：false
解释：1 不能在 2 之前弹出。


提示：

0 <= pushed.length == popped.length <= 1000
0 <= pushed[i], popped[i] < 1000
pushed 是 popped 的排列。
注意：本题与主站 946 题相同：https://leetcode-

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/zhan-de-ya-ru-dan-chu-xu-lie-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    bool validateStackSequences(vector<int>& pushed, vector<int>& popped) {
        //使用一个辅助栈压入数组pushed
        // 每压入一个元素，都判断栈顶元素和下一个弹出元素是否相等
        // 如果相等，弹出元素，继续判断，直到栈顶元素不等于弹出元素或栈空
        // 使用一个指针指向下一个弹出位置
        // 如果弹出元素剩余，则说明popped不是弹出序列
        stack<int> aux;
        int flag = 0;
        for(int i = 0; i < pushed.size(); ++i){
            // 元素入栈
            aux.push(pushed[i]);
            // 判断栈顶元素与下一个弹出元素是否相等,相等则出栈
            while(!aux.empty() && aux.top() == popped[flag]){
                // 弹出栈顶元素
                aux.pop();
                // 弹出元素后移
                flag++;
            }
        }
        return flag == popped.size();
    }
};
```

20 请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。

数值（按顺序）可以分成以下几个部分：

若干空格
一个 小数 或者 整数
（可选）一个 'e' 或 'E' ，后面跟着一个 整数
若干空格
小数（按顺序）可以分成以下几个部分：

（可选）一个符号字符（'+' 或 '-'）
下述格式之一：
至少一位数字，后面跟着一个点 '.'
至少一位数字，后面跟着一个点 '.' ，后面再跟着至少一位数字
一个点 '.' ，后面跟着至少一位数字
整数（按顺序）可以分成以下几个部分：

（可选）一个符号字符（'+' 或 '-'）
至少一位数字
部分数值列举如下：

["+100", "5e2", "-123", "3.1416", "-1E-16", "0123"]
部分非数值列举如下：

["12e", "1a3.14", "1.2.3", "+-5", "12e+5.4"]


示例 1：

输入：s = "0"
输出：true
示例 2：

输入：s = "e"
输出：false
示例 3：

输入：s = "."
输出：false
示例 4：

输入：s = "    .1  "
输出：true


提示：

1 <= s.length <= 20
s 仅含英文字母（大写和小写），数字（0-9），加号 '+' ，减号 '-' ，空格 ' ' 或者点 '.' 。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/biao-shi-shu-zhi-de-zi-fu-chuan-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    bool isNumber(string s) {
        // 如果前面可能有空格，可以去掉空格
        // 如果去掉空格后，第一个非数字字符不是 (. + - )，那么就不是数值
        // 如果第一个符号是(.)，说明是小数，之后如果再出现其它非数字符号，就不是数值，空格不算
        // 如果第一个符号不是点，说明后面可以出现点
        // 点的后面必须跟数字，否则不是数值
        // e的后面必须跟数值
        int n = s.size();
        // 去除空格
        int start = 0, end = n - 1;
        while(start < n && s[start] == ' ') start++;
        while(end >= 0 && s[end] == ' ') end--;
        if(start == n || end == -1) return false;
        int left = start, right = end;
        bool have_dot = false, have_sum = false, have_sub = false, have_e = false, have_num = false;
        int index_dot = -1, index_e = -1;
        if(left == right) if(s[left] - '0' > 9 || s[left] - '0' < 0) return false;     
        while(left <= right){
            if(s[left] - '0' <= 9 && s[left] - '0' >= 0){
                have_num = true;
                left++;
            }
            else if(s[left] == '+'){
                // + 可以出现在开始或e后面
                if((left != start && s[left - 1] != 'e' && s[left - 1] != 'E') || left == right) return false;
                else left++;
            }
            // - 也是一样
            else if(s[left] == '-'){
                // + 可以出现在开始或e后面
                if((left != start && s[left - 1] != 'e' && s[left - 1] != 'E') || left == right) return false;
                else left++;
            }
            else if(s[left] == '.'){
                // 点不可以出现在e的后面
                if(have_e || have_dot) return false;
                else{
                    have_dot = true;
                    left++;
                }
            }
            else if(s[left] == 'e' || s[left] == 'E'){
                // e的前面必须有数，并且不能有e
                if(have_e || !have_num || left == right) return false;
                else{
                    have_e = true;
                    left++;
                }
            }
            else return false;
        }
        if(!have_num) return false;
        else return true;
    }
};
```

67 写一个函数 StrToInt，实现把字符串转换成整数这个功能。不能使用 atoi 或者其他类似的库函数。

 

首先，该函数会根据需要丢弃无用的开头空格字符，直到寻找到第一个非空格的字符为止。

当我们寻找到的第一个非空字符为正或者负号时，则将该符号与之后面尽可能多的连续数字组合起来，作为该整数的正负号；假如第一个非空字符是数字，则直接将其与之后连续的数字字符组合起来，形成整数。

该字符串除了有效的整数部分之后也可能会存在多余的字符，这些字符可以被忽略，它们对于函数不应该造成影响。

注意：假如该字符串中的第一个非空格字符不是一个有效整数字符、字符串为空或字符串仅包含空白字符时，则你的函数不需要进行转换。

在任何情况下，若函数不能进行有效的转换时，请返回 0。

说明：

假设我们的环境只能存储 32 位大小的有符号整数，那么其数值范围为 [−231,  231 − 1]。如果数值超过这个范围，请返回  INT_MAX (231 − 1) 或 INT_MIN (−231) 。

示例 1:

输入: "42"
输出: 42
示例 2:

输入: "   -42"
输出: -42
解释: 第一个非空白字符为 '-', 它是一个负号。
     我们尽可能将负号与后面所有连续出现的数字组合起来，最后得到 -42 。
示例 3:

输入: "4193 with words"
输出: 4193
解释: 转换截止于数字 '3' ，因为它的下一个字符不为数字。
示例 4:

输入: "words and 987"
输出: 0
解释: 第一个非空字符是 'w', 但它不是数字或正、负号。
     因此无法执行有效的转换。
示例 5:

输入: "-91283472332"
输出: -2147483648
解释: 数字 "-91283472332" 超过 32 位有符号整数范围。 
     因此返回 INT_MIN (−231) 。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/ba-zi-fu-chuan-zhuan-huan-cheng-zheng-shu-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    int strToInt(string str) {
        // 从第一个非空格开始遍历字符串
        // 如果第一个字符不是+-数字，返回
        // 2^31 = 2147483648, 2^31 - 1 = 2147483647，用一个标签保存位数
        // 用一个标签保存符号位
        // 如果符号位为+且第10位大于7，返回INT_MAX
        // 如果符号位位-且第10位大于8，返回INT_MIN
        int n = str.size();
        int left = 0;
        while(str[left] == ' ') left++;
       
        if(left == n) return 0;
        // 第一个非空字符如果不是数字、+、-返回
        if((str[left] - '0' < 0 || str[left] - '0' > 9) && str[left] != '+' && str[left] != '-') return 0;
        char sign = '+';
        if(str[left] == '+' || str[left] == '-'){
            sign = str[left];
            left++;
        }
         // 去除前面全是0的
        while(str[left] == '0') left++;
        int res = 0, flag_num = 1;
        for(int i = left; i < n; ++i){
            //遇到非数字类型返回
            if(str[i] - '0' < 0 || str[i] - '0' > 9) break;
            if(flag_num == 10){
                if(res == 214748364){
                    //已经有9位数了
                    if(sign == '+' && str[i] - '0' > 7) return INT_MAX;
                    if(sign == '-' && str[i] - '0' >= 8) return INT_MIN;
                    else {
                        res = 10 * res + (str[i] - '0');
                        flag_num++;
                    }
                }
                else if(res > 214748364) {
                    if(sign == '+') return INT_MAX;
                    if(sign == '-') return INT_MIN;
                }
                else {
                        res = 10 * res + (str[i] - '0');
                        flag_num++;
                    }
            }
            else if(flag_num > 10){
                //已经有10位数了
                if(sign == '+') return INT_MAX;
                if(sign == '-') return INT_MIN;
            }
            else {
                res = 10 * res + (str[i] - '0');
                flag_num++;
            }
        }
        if(sign == '+') return res;
        else return -res;
    }
};
```

59 给定一个数组 nums 和滑动窗口的大小 k，请找出所有滑动窗口里的最大值。

示例:

输入: nums = [1,3,-1,-3,5,3,6,7], 和 k = 3
输出: [3,3,5,5,6,7] 
解释: 

  滑动窗口的位置                最大值
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7


提示：

你可以假设 k 总是有效的，在输入数组不为空的情况下，1 ≤ k ≤ 输入数组的大小。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/hua-dong-chuang-kou-de-zui-da-zhi-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        //从左向右移动窗口，然后找窗口里面的最大值
        vector<int> res;
        int left = 0, right = k - 1;
        while(right < nums.size()){
            int max_num = INT_MIN;
            for(int i = left; i <= right; ++i)
                max_num = max_num > nums[i] ? max_num : nums[i];
            res.push_back(max_num);
            right++;
            left++;
        }
        return res;
    }
};



class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        // 从左向右移动窗口，然后找窗口里面的最大值，时间花费高
        // 使用双端队列
        // 首先将第一个窗口中的最大值加入队列
        // 当k=1时，最大值是1，当k=2时，最大值是三，此时1不可能是之后窗口的最大值，将其出队
        
        vector<int> res;
        deque<int> window_max;
        for(int i = 0; i < nums.size(); ++i){
            // 队列中的最后一个元素比nums[i]还要小，说明这个元素不可能是窗口的最大值
            while(!window_max.empty() && nums[window_max.back()] < nums[i]) 
                window_max.pop_back();
            // 窗口右移，如果队首元素是未移动的窗口的左界，则移动后队首元素不在窗口中
            while(!window_max.empty() && i - window_max.front() >= k)
                window_max.pop_front();
            // 窗口右界加到队列中
            window_max.push_back(i);
            //当前窗口的最大值就是队首元素
            if(i >= k - 1)
               res.push_back(nums[window_max.front()]);
        }
        return res;
    }
};
```

59 请定义一个队列并实现函数 max_value 得到队列里的最大值，要求函数max_value、push_back 和 pop_front 的均摊时间复杂度都是O(1)。

若队列为空，pop_front 和 max_value 需要返回 -1

示例 1：

输入: 
["MaxQueue","push_back","push_back","max_value","pop_front","max_value"]
[[],[1],[2],[],[],[]]
输出: [null,null,null,2,1,2]
示例 2：

输入: 
["MaxQueue","pop_front","max_value"]
[[],[],[]]
输出: [null,-1,-1]


限制：

1 <= push_back,pop_front,max_value的总操作数 <= 10000
1 <= value <= 10^5
通过次数125,637提交次数263,254

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/dui-lie-de-zui-da-zhi-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class MaxQueue {
    // 1 3 5 4 2 6
    // 1入队最大值是1，3入队最大值是3，1不是最大值，5入队最大值是5，3不是最大值，4入队5是最大值，但不确定4是不是之后序列的最大值，2同理，6入队后确定4，2不是最大值
public:
    queue<int> my_queue;
    deque<int> max_queue;
    MaxQueue() {
    }
    
    int max_value() {
        if(!max_queue.empty()) return max_queue.front();
        else return -1;
    }
    
    void push_back(int value) {
        my_queue.push(value);
        while(!max_queue.empty() && max_queue.back() < value)
            max_queue.pop_back();
        max_queue.push_back(value);
    }
    
    int pop_front() {
        if(!my_queue.empty()){
            int value = my_queue.front();
            if(value == max_queue.front())
                max_queue.pop_front();
            my_queue.pop();
            return value;
        }
        else return -1;
    }
};

/**
 * Your MaxQueue object will be instantiated and called as such:
 * MaxQueue* obj = new MaxQueue();
 * int param_1 = obj->max_value();
 * obj->push_back(value);
 * int param_3 = obj->pop_front();
 */
```

37 请实现两个函数，分别用来序列化和反序列化二叉树。

你需要设计一个算法来实现二叉树的序列化与反序列化。这里不限定你的序列 / 反序列化算法执行逻辑，你只需要保证一个二叉树可以被序列化为一个字符串并且将这个字符串反序列化为原始的树结构。

提示：输入输出格式与 LeetCode 目前使用的方式一致，详情请参阅 LeetCode 序列化二叉树的格式。你并非必须采取这种方式，你也可以采用其他的方法解决这个问题。

 

示例：


输入：root = [1,2,3,null,null,4,5]
输出：[1,2,3,null,null,4,5]

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/xu-lie-hua-er-cha-shu-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Codec 
{
public:

    // Encodes a tree to a single string.
    string serialize(TreeNode* root) 
    {
        string res = "";
        if (root == NULL)
            return "";
        queue<TreeNode*> Q;
        Q.push(root);
        while (!Q.empty())
        {
            TreeNode* p = Q.front();    Q.pop();
            if (p == NULL)
                res += "NULL,";
            else
            {
                res += to_string(p->val);   res += ',';
                Q.push(p->left);
                Q.push(p->right);
            }
        }

        return res;
    }

    // Decodes your encoded data to tree.
    TreeNode* deserialize(string data) 
    {
        vector<string> rec;

        if (data.size() == 0)
            return NULL;
        int i = 0,  j = 0;
        while (i < data.size())
        {
            if (data[i] != ',')
                ++ i;
            else
            {
                string tmp = data.substr(j, i - j);
                rec.push_back(tmp);
                ++ i;
                j = i;
            }
        }
        if (j != i)
        {
            string tmp = data.substr(j, i - j);
            rec.push_back(tmp);
        }

        TreeNode * root = new TreeNode(stoi(rec[0]));
        queue<TreeNode*> Q;
        Q.push(root);
        i = 1;
        while (!Q.empty())
        {
            TreeNode* p = Q.front();    Q.pop();
            if (rec[i] != "NULL")
            {
                p->left = new TreeNode(stoi(rec[i]));
                Q.push(p->left);
            }
            ++ i;
            if (rec[i] != "NULL")
            {
                p->right = new TreeNode(stoi(rec[i]));
                Q.push(p->right);
            }
            ++ i;
        }
        return root;
    }
};

```

38 输入一个字符串，打印出该字符串中字符的所有排列。

 

你可以以任意顺序返回这个字符串数组，但里面不能有重复元素。

 

示例:

输入：s = "abc"
输出：["abc","acb","bac","bca","cab","cba"]


限制：

1 <= s 的长度 <= 8

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/zi-fu-chuan-de-pai-lie-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```c++
class Solution {
public:
    vector<string> permutation(string s) {
        //next_permutation 函数可以获取序列的下一个升序排列
        sort(s.begin(), s.end());
        vector<string> res;
        do{
            res.push_back(s);
        } while(next_permutation(s.begin(), s.end()));
        return res;
    }
};
```

