import java.util.*;

/**
 * Created by zplchn on 10/7/16.
 */
public class Solution {
    class ListNode{
        int val;
        ListNode next;

        int getVal(){
            return val;
        }

        ListNode(int x){val = x;}
    }

    class TreeNode{
        int val;
        TreeNode left, right;
        TreeNode(int x){val = x;}
    }

    class TrieNode {
        TrieNode[] children; //here cannot use private as the Trie class will directly call it.
        boolean isWord;
        // Initialize your data structure here.
        public TrieNode() {
            children = new TrieNode[26];
        }
    }

    class Interval{
        int start, end;
        public Interval(){start = end = 0;}
        public Interval(int x, int y){
            start = x;
            end = y;
        }
    }

    public static void main(String[] args){
        Solution st = new Solution();
        double x = 0;
        int v = 3;
        x += v;
        int n = -3;
        n %= 10;
        System.out.println(n);
        System.out.println("3".substring(0, 0)); //empty string
        System.out.println(x);
        String s = "12,3,#,5,#,";
        String[] tokens = s.split(",");
        for (String t : tokens)
            System.out.println("[" + t + "]");
        /*
        [12] split will check till last char but will ignore after last one!
        [3]
        [#]
        [5]
        [#]
         */



    }

    //1
    public int[] twoSum1(int[] nums, int target) {
        int[] res = {-1, -1};
        if (nums == null || nums.length < 2)
            return res;
        Map<Integer, Integer> hm = new HashMap<>();
        for (int i = 0; i < nums.length; ++i){
            if (hm.containsKey(target - nums[i])){
                res[0] = hm.get(target - nums[i]);
                res[1] = i;
                break;
            }
            hm.put(nums[i], i);
        }
        return res;
    }

    //2
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        if (l1 == null)
            return l2;
        if (l2 == null)
            return l1;
        int sum = 0, carry = 0;
        ListNode dummy = new ListNode(0);
        ListNode pre = dummy;
        while (l1 != null || l2 != null || carry != 0){
            sum = (l1!= null? l1.val: 0) + (l2!= null? l2.val: 0) + carry;
            pre.next = new ListNode(sum % 10);
            pre = pre.next;
            carry = sum / 10;
            l1 = l1!=null? l1.next: l1;
            l2 = l2!=null? l2.next: l2;
        }
        pre.next = null;
        return dummy.next;
    }

    //5
    public String longestPalindrome(String s) {
        if (s == null || s.length() == 0)
            return "";
        boolean [][] dp = new boolean[s.length()][s.length()];
        String res = "";
        int max = 0, l = 0, r = 0;
        for (int i = s.length() - 1; i >= 0; --i){
            for (int j = i; j < s.length(); ++j){
                if (s.charAt(i) == s.charAt(j) && (j - i <= 2 || dp[i+1][j-1])){
                    dp[i][j] = true;
                    if (j - i + 1 > max){
                        l = i;
                        r = j + 1;
                        max = j - i + 1;
                    }
                }
            }
        }
        return s.substring(l, r);
    }

    //7
    public int reverse(int x) {
        long res = 0;
        while (x != 0){
            res = res * 10 + x % 10; //note -3 % 10 = -3
            x /= 10;
            if (res > Integer.MAX_VALUE || res < Integer.MIN_VALUE)
                return 0;
        }
        return (int)res;
    }

    //9
    public boolean isPalindrome(int x) {
        if (x < 0)
            return false;
        int div = 1;
        while (x / div >= 10)
            div *= 10;
        while (x != 0){
            if (x / div != x % 10)
                return false;
            x = x % div / 10;
            div /= 100;
        }
        return true;
    }

    //11
    public int maxArea(int[] height) {
        if (height == null || height.length < 2)
            return 0;
        int l = 0, r = height.length - 1, max = 0;
        while (l < r){
            int min = Math.min(height[l], height[r]);
            if (min == height[l]){
                max = Math.max(max, min * (r - l));
                ++l;
            }
            else if (min == height[r]){
                max = Math.max(max, min * (r - l));
                --r;
            }
        }
        return max;
    }

    //13
    public int romanToInt(String s) {
        if (s == null || s.length() == 0)
            return 0;
        Map<Character, Integer> hm = new HashMap<>();
        hm.put('I', 1);
        hm.put('V', 5);
        hm.put('X', 10);
        hm.put('L', 50);
        hm.put('C', 100);
        hm.put('D', 500);
        hm.put('M', 1000);
        int res = hm.get(s.length() - 1);
        for (int i = s.length() -2; i>= 0; --i){
            if (hm.get(s.charAt(i)) < hm.get(s.charAt(i+1)))
                res -= hm.get(s.charAt(i));
            else
                res += hm.get(s.charAt(i));
        }
        return res;
    }

    //14
    public String longestCommonPrefix(String[] strs) {
        if (strs == null || strs.length == 0)
            return "";
        String pre = strs[0];
        int min = pre.length();
        for (int i = 1; i < strs.length; ++i){
            int j = 0;
            for (; j < Math.min(min, strs[i].length()); ++j){
                if (pre.charAt(j) != strs[i].charAt(j))
                    break;
            }
            min = Math.min(min, j);
        }
        return pre.substring(0, min);
    }

    //15
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        if (nums == null || nums.length < 3)
            return res;
        Arrays.sort(nums);
        for (int i = 0; i < nums.length -2; ++i){
            if (i > 0 && nums[i] == nums[i-1])
                continue;
            int l = i + 1, r = nums.length - 1;
            while (l < r){
                int sum = nums[i] + nums[l] + nums[r];

                if (sum < 0)
                    ++l;
                else if(sum > 0)
                    --r;
                else {
                    res.add(Arrays.asList(nums[i], nums[l++], nums[r--]));
                    while (l < r && nums[l] == nums[l - 1])
                        ++l;
                    while (l < r && nums[r] == nums[r + 1])
                        --r;
                }
            }
        }
        return res;
    }

    //16
    public int threeSumClosest(int[] nums, int target) {
        if (nums == null || nums.length < 3)
            return 0;
        Arrays.sort(nums);
        int diff = Integer.MAX_VALUE, res = 0;
        for (int i = 0; i < nums.length - 2; ++i){
            if (i > 0 && nums[i] == nums[i-1])
                continue;
            int l = i + 1, r = nums.length - 1;
            while (l < r){
                int sum = nums[i] + nums[l] + nums[r];
                if (Math.abs(target - sum) < diff){
                    diff = Math.abs(target - sum);
                    res = sum;
                }
                if (sum < target)
                    ++l;
                else if (sum > target)
                    --r;
                else
                    break;
            }
        }
        return res;
    }

    //17
    private static final String[] phone = {"", "", "abc", "def","ghi","jkl","mno", "pqrs","tuv","wxyz"};

    public List<String> letterCombinations(String digits) {
        List<String> res = new ArrayList<>();
        if (digits == null || digits.length() == 0)
            return res;
        letterCombiHelper(digits, 0, new StringBuilder(),res);
        return res;
    }

    private void letterCombiHelper(String digits, int i, StringBuilder sb, List<String> res){
        if (i == digits.length()){
            res.add(sb.toString());
            return;
        }
        String s = phone[digits.charAt(i) - '0'];
        for (int k = 0; k < s.length(); ++k){
            sb.append(s.charAt(k));
            letterCombiHelper(digits, i + 1, sb, res);
            sb.deleteCharAt(sb.length() - 1);
        }
    }

    //18
    public List<List<Integer>> fourSum(int[] nums, int target) {
        List<List<Integer>> res = new ArrayList<>();
        if (nums == null || nums.length < 4)
            return res;
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 3; ++i){
            if (i > 0 && nums[i] == nums[i-1])
                continue;
            for (int j = i + 1; j < nums.length - 2; ++j){
                if (j > i + 1 && nums[j] == nums[j - 1])
                    continue;
                int l = j + 1, r = nums.length - 1;
                while (l < r){
                    int sum = nums[i] + nums[j] + nums[l] + nums[r];
                    if (sum < target)
                        ++l;
                    else if (sum > target)
                        --r;
                    else {
                        res.add(Arrays.asList(nums[i],nums[j],nums[l++],nums[r--]));
                        while (l < r && nums[l] == nums[l - 1])
                            ++l;
                        while (l < r && nums[r] == nums[r + 1])
                            --r;
                    }
                }
            }
        }
        return res;
    }

    //20
    public boolean isValid(String s) {
        if (s == null || s.length() == 0)
            return false;
        Deque<Character> st = new ArrayDeque<>();
        for (int i = 0; i < s.length(); ++i){
            switch(s.charAt(i)){
                case '(':
                case '{':
                case '[':
                    st.push(s.charAt(i));
                    break;
                case ')':
                    if (st.isEmpty() || st.pop() != '(')
                        return false;
                    break;
                case ']':
                    if (st.isEmpty() || st.pop() != '(')
                        return false;
                    break;
                case '}':
                    if (st.isEmpty() || st.pop() != '(')
                        return false;
                    break;
                default:
                    break;
            }
        }
        return st.isEmpty();
    }

    //21
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null)
            return l2;
        if (l2 == null)
            return l1;
        ListNode dummy = new ListNode(0), pre = dummy;
        while (l1 != null && l2 != null){
            if (l1.val < l2.val){
                pre.next = l1;
                l1 = l1.next;
            }
            else {
                pre.next = l2;
                l2 = l2.next;
            }
            pre = pre.next;
        }
        pre.next = l1 != null ? l1 : l2;
        return dummy.next;
    }

    //22
    public List<String> generateParenthesis(int n) {
        List<String> res = new ArrayList<>();
        if (n <= 0)
            return res;
        generateParHelper(n, 0, 0, "", res);
        return res;
    }

    private void generateParHelper(int n, int l, int r, String pre, List<String> res){
        if(r == n){
            res.add(pre);
            return;
        }
        if (l < n){
            generateParHelper(n, l + 1, r, pre + "(", res);
        }
        if (r < l){
            generateParHelper(n, l, r + 1, pre + ")", res);
        }
    }

    //23
    public ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length == 0)
            return null;
        Queue<ListNode> pq = new PriorityQueue<>((n1, n2)->n1.val - n2.val);
        ListNode dummy = new ListNode(0);
        ListNode cur = dummy;

        for (ListNode ln: lists){
            if (ln != null)
                pq.offer(ln);
        }

        while (!pq.isEmpty()){
            ListNode ln = pq.poll();
            cur.next = ln;
            cur = cur.next;
            if (ln.next != null)
                pq.offer(ln.next);
        }
        cur.next = null;
        return dummy.next;
    }

    //26
    public int removeDuplicates(int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;
        int l = 0, r = 1;
        while (r < nums.length){
            if (nums[l] != nums[r])
                nums[++l] = nums[r];
            ++r;
        }
        return l + 1;
    }

    //27
    public int removeElement(int[] nums, int val) {
        if (nums == null || nums.length == 0)
            return 0;
        int l = 0, r = 0;
        while (r < nums.length){
            if (nums[r] != val)
                nums[l++] = nums[r];
            ++r;
        }
        return l;
    }

    //31
    public void nextPermutation(int[] nums) {
        if (nums == null || nums.length == 0)
            return;
        int i = nums.length -2;
        while (i >= 0 && nums[i] >= nums[i+1])
            --i;
        if (i < 0) {
            reverse(nums, 0, nums.length - 1);
            return;
        }
        int j = nums.length - 1;
        while (j > i && nums[j] <= nums[i])
            --j;
        swap(nums, i, j);
        reverse(nums, i + 1, nums.length - 1);
    }

    private void reverse(int[] nums, int i, int j){
        while (i < j){
            swap(nums, i++, j--);
        }
    }

    private void swap(int[] nums, int i, int j){
        int t = nums[i];
        nums[i] = nums[j];
        nums[j] = t;
    }

    //33
    public int search(int[] nums, int target) {
        if (nums == null || nums.length == 0)
            return -1;
        int l = 0, r = nums.length - 1, m;
        while (l <= r){
            m = l + ((r - l) >> 1);
            if (nums[m] == target)
                return m;
            else if (nums[m] < nums[r]){
                if (target > nums[m] && target <= nums[r])
                    l = m + 1;
                else
                    r = m - 1;
            }
            else {
                if (target >= nums[l] && target < nums[m])
                    r = m - 1;
                else
                    l = m + 1;
            }
        }
        return -1;
    }

    //34
    public int[] searchRange(int[] nums, int target) {
        int[] res = {-1, -1};
        if (nums == null || nums.length == 0)
            return res;
        int l = 0, r = nums.length - 1, m;
        while (l <= r){
            m = l + ((r - l) >> 1);
            if (nums[m] <= target)
                l = m + 1;
            else
                r = m - 1;
        }
        int rr = r;
        l = 0;
        r = nums.length - 1;
        while (l <= r){
            m = l + ((r - l) >> 1);
            if (nums[m] >= target)
                r = m - 1;
            else
                l = m + 1;
        }
        if (l > rr)
            return res;
        res[0] = l;
        res[1] = rr;
        return res;
    }

    //35
    public int searchInsert(int[] nums, int target) {
        if (nums == null || nums.length == 0)
            return 0;
        int l = 0, r = nums.length - 1, m;
        while (l <= r){
            m = l + ((r - l) >> 1);
            if (target > nums[m])
                l = m + 1;
            else if (target < nums[m])
                r = m - 1;
            else
                return m;
        }
        return l;
    }

    //36
    public boolean isValidSudoku(char[][] board) {
        if (board == null || board.length != 9 || board[0].length != 9)
            return false;

        for (int i = 0; i < board.length; ++i){
            Set<Character> hs = new HashSet<>();
            for (int j = 0; j < board[0].length; ++j){
                if (board[i][j] != '.'){
                    if (hs.contains(board[i][j]))
                        return false;
                    hs.add(board[i][j]);
                }
            }
        }
        for (int j = 0; j < board[0].length; ++j){
            Set<Character> hs = new HashSet<>();
            for (int i = 0; i < board.length; ++i){
                if (board[i][j] != '.'){
                    if (hs.contains(board[i][j]))
                        return false;
                    hs.add(board[i][j]);
                }
            }
        }
        for (int m = 0; m < 9; ++m){
            Set<Character> hs = new HashSet<>();
            for (int i = m / 3 * 3; i < m / 3 * 3 + 3; ++i){
                for (int j = m % 3 * 3; j < m % 3 * 3 + 3; ++j){
                    if (board[i][j] != '.'){
                        if (hs.contains(board[i][j]))
                            return false;
                        hs.add(board[i][j]);
                    }
                }
            }
        }
        return true;
    }

    //37
    public void solveSudoku(char[][] board) {
        if (board == null || board.length != 9 || board[0].length != 9)
            return;
        solveSudokuHelper(board, 0, 0);
    }

    private boolean solveSudokuHelper(char[][] board, int i, int j){
        if (i == 9)
            return true;
        if (j == 9)
            return solveSudokuHelper(board, i + 1, 0);

        if (board[i][j] == '.'){
            for (char k = '1'; k <= '9'; ++k){
                board[i][j] = k;
                if (isValidSudoku2(board, i, j)){
                    if (solveSudokuHelper(board, i, j+1))
                        return true;
                }
            }
            board[i][j] = '.'; //still need backtracking
        }
        else
            return solveSudokuHelper(board, i, j + 1);
        return false;
    }

    private boolean isValidSudoku2(char[][] board, int i, int j){
        for (int jj = 0; jj < board[0].length; ++jj){
            if (board[i][jj] == board[i][j] && jj != j)
                return false;
        }
        for (int ii = 0; ii < board.length; ++ii){
            if (board[i][j] == board[ii][j] && ii != i)
                return false;
        }
        for (int ii = i /3*3; ii < i/3*3 + 3; ++ii){
            for (int jj = j/3*3; jj < j/3*3+3; ++jj){
                if (board[i][j] == board[ii][jj] && !(i == ii && j == jj))
                    return false;
            }
        }
        return true;
    }

    //39
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        if (candidates == null || candidates.length == 0)
            return res;
        Arrays.sort(candidates);
        combinationSumHelper(candidates, target, 0, 0, new ArrayList<Integer>(), res);
        return res;
    }

    private void combinationSumHelper(int[] candidates, int target, int i, int sum, List<Integer> combi, List<List<Integer>> res){
        if (sum == target){
            res.add(new ArrayList<>(combi));
            return;
        }
        for (int k = i; k < candidates.length; ++k){
            if (k > i && candidates[k] == candidates[k-1])
                continue;
            if (sum + candidates[k] <= target){
                combi.add(candidates[k]);
                combinationSumHelper(candidates, target, k, sum + candidates[k], combi, res);
                combi.remove(combi.size() - 1);
            }
        }
    }

    //40
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        if (candidates == null || candidates.length == 0)
            return res;
        Arrays.sort(candidates);
        combinationSum2Helper(candidates, target, 0, 0, new ArrayList<Integer>(), res);
        return res;
    }

    private void combinationSum2Helper(int[] candidates, int target, int i, int sum, List<Integer> combi, List<List<Integer>> res){
        if (sum == target){
            res.add(new ArrayList<>(combi));
            return;
        }
        for (int k = i; k < candidates.length; ++k){
            if (k > i && candidates[k] == candidates[k - 1])
                continue;
            if (sum +candidates[k] <= target){
                combi.add(candidates[k]);
                combinationSum2Helper(candidates, target, k + 1, sum + candidates[k], combi, res);
                combi.remove(combi.size() - 1);
            }
        }
    }

    //41
    public int firstMissingPositive(int[] nums) {
        if (nums == null || nums.length == 0)
            return 1;
        for (int i = 0; i < nums.length; ++i){
            if (i + 1 != nums[i] && nums[i] >= 1 && nums[i] <= nums.length && nums[nums[i] - 1] != nums[i]){
                int t = nums[nums[i] - 1];
                nums[nums[i] - 1] = nums[i];
                nums[i] = t;
                --i;
            }
        }
        for (int i = 0; i < nums.length; ++i){
            if (i + 1 != nums[i])
                return i + 1;
        }
        return nums.length + 1; //[1] return 2.
    }

    //42
    public int trap(int[] height) {
        if (height == null || height.length <= 2)
            return 0;
        int l = 0, r = height.length - 1, res = 0;
        while (l < r){
            int min = Math.min(height[l], height[r]);
            if (min == height[l]){
                while (l < r && height[l] <= min){
                    res += min - height[l++];
                }
            }
            else {
                while (l < r && height[r] <= min)
                    res += min - height[r--];
            }
        }
        return res;
    }

    //45
    public int jump(int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;
        int lastReach = 0, max = 0, step = 0;
        for (int i = 0; i <= max && i < nums.length; ++i){
            if (i > lastReach){
                ++step;
                lastReach = max;
            }
            max = Math.max(max, i + nums[i]); //dont stop in teh middle very hard to follow
        }
        return max >= nums.length - 1? step: -1;
    }

    //46
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        if (nums == null || nums.length == 0)
            return res;
        permuteHelper(nums, 0, new boolean[nums.length], new ArrayList<Integer>(), res);
        return res;
    }

    private void permuteHelper(int[] nums, int i, boolean[] used, List<Integer> combi, List<List<Integer>> res){
        if (i == nums.length){
            res.add(new ArrayList<>(combi));
            return;
        }
        for (int k = 0; k < nums.length; ++k){
            if (!used[k]){
                used[k] = true;
                combi.add(nums[k]);
                permuteHelper(nums, i + 1, used, combi, res);
                combi.remove(combi.size() - 1);
                used[k] = false;
            }
        }
    }

    //47
    public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        if (nums == null || nums.length == 0)
            return res;
        Arrays.sort(nums);
        permuteUniqueHelper(nums, 0, new boolean[nums.length], new ArrayList<Integer>(), res);
        return res;
    }

    private void permuteUniqueHelper(int[] nums, int i, boolean[] used, List<Integer> combi, List<List<Integer>> res){
        if (i == nums.length){
            res.add(new ArrayList<>(combi));
            return;
        }
        for (int k = 0; k < nums.length; ++k){
            if (used[k] || (k > 0 && nums[k] == nums[k-1] && !used[k-1])) //when first dup is used we skip, otherwise first used, second also use
                continue;
            used[k] = true;
            combi.add(nums[k]);
            permuteUniqueHelper(nums, i + 1, used, combi, res);
            combi.remove(combi.size() - 1);
            used[k] = false;
        }
    }

    //49
    public List<List<String>> groupAnagrams(String[] strs) {
        List<List<String>> res = new ArrayList<>();
        if (strs == null || strs.length == 0)
            return res;
        Map<String, List<String>> hm = new HashMap<>();
        for (String s : strs){
            char[] ss = s.toCharArray();
            Arrays.sort(ss);
            String ns = new String(ss);
            if (!hm.containsKey(ns))
                hm.put(ns, new ArrayList<>());
            hm.get(ns).add(s);
        }
        res.addAll(hm.values());
        return res;
    }

    //50
    public double myPow(double x, int n) {
        if (n == 0)
            return 1;
        if (n == 1)
            return x;
        double t = myPow(x, n/2);
        if (n % 2 == 0)
            return t * t;
        else if (n < 0) //it's n < 0 not x!!!
             return t * t / x;
        else
            return t * t * x;
    }

    //53
    public int maxSubArray(int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;
        int lmax = nums[0], max = nums[0];
        for (int i = 1; i < nums.length; ++i){
            lmax = Math.max(lmax + nums[i], nums[i]);
            max = Math.max(max, lmax);
        }
        return max;
    }

    //55
    public boolean canJump(int[] nums) {
        if (nums == null || nums.length == 0)
            return true;
        int max = 0, lastReach = 0;
        for (int i = 0; i <= max && i < nums.length; ++i){
            if (i > lastReach)
                lastReach = max;
            max = Math.max(max, i + nums[i]);
            if (max >= nums.length - 1)
                return true;
        }
        return false;
    }

    //56
    public List<Interval> merge(List<Interval> intervals) {
        List<Interval> res = new ArrayList<>();
        if (intervals == null || intervals.size() == 0)
            return res;
        intervals.sort((i1, i2)->i1.start - i2.start);
        res.add(intervals.get(0));
        for (int i = 1; i < intervals.size(); ++i){
            if (intervals.get(i).start <= res.get(res.size()-1).end)
                res.get(res.size()-1).end = Math.max(intervals.get(i).end, res.get(res.size()-1).end);
            else
                res.add(intervals.get(i));
        }
        return res;
    }

    //57
    public List<Interval> insert(List<Interval> intervals, Interval newInterval) {
        List<Interval> res = new ArrayList<>();
        if (intervals == null || newInterval == null) //intervals.size() ==0 is ok
            return intervals;
        intervals.sort((i1, i2)->i1.start - i2.start);
        int i = 0;
        while (i < intervals.size() && newInterval.start > intervals.get(i).end)
            res.add(intervals.get(i++));
        while (i < intervals.size() && newInterval.end >= intervals.get(i).start){
            newInterval.start = Math.min(newInterval.start, intervals.get(i).start);
            newInterval.end = Math.max(newInterval.end, intervals.get(i).end);
            ++i;
        }
        res.add(newInterval);
        while (i < intervals.size())
            res.add(intervals.get(i++));
        return res;
    }

    //58
    public int lengthOfLastWord(String s) {
        if (s == null)
            return 0;
        s = s.trim();
        if (s.length() == 0)
            return 0;
        String[] tokens = s.split("\\s+");
        return tokens[tokens.length - 1].length();
    }

    //60
    public String getPermutation(int n, int k) {
        if (n <= 0 || k <= 0)
            return "";
        k -= 1; //back to 0-based index
        List<Integer> nums = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        int fac = 1;
        for (int i = 1; i <= n; ++i) {
            nums.add(i);
            fac *= i;
        }

        for (int i = 0; i < n; ++i){
            fac /= (n - i);
            int index = k / fac;
            k %= fac;
            sb.append(nums.get(index));
            nums.remove(index);
        }
        return sb.toString();
    }

    //62
    public int uniquePaths(int m, int n) {
        if (m <= 0 || n <= 0)
            return 0;
        int[][] dp = new int[m][n];
        dp[0][0] = 1;

        for (int i = 0; i < dp.length; ++i){
            for (int j = 0; j < dp[0].length; ++j){
                dp[i][j] += (i > 0? dp[i-1][j]: 0) + (j > 0? dp[i][j-1]: 0);
            }
        }
        return dp[dp.length - 1][dp[0].length - 1];
    }

    //63
    public int uniquePathsWithObstacles(int[][] og) {
        if (og == null || og.length == 0 || og[0].length == 0)
            return 0;
        if (og[0][0] == 1)
            return 0;
        og[0][0] = 1; //need to set start to 1!
        for (int i = 0; i < og.length; ++i){
            for (int j = 0; j < og[0].length; ++j){
                if (og[i][j] == 1 && !(i == 0 && j == 0))
                    og[i][j] = 0;
                else
                    og[i][j] += (i > 0? og[i-1][j]: 0) + (j > 0?og[i][j-1]: 0);
            }
        }
        return og[og.length - 1][og[0].length - 1];
    }

    //64
    public int minPathSum(int[][] grid) {
        if (grid == null || grid.length == 0 || grid[0].length == 0)
            return 0;
        for (int i = 0; i < grid.length; ++i){
            for (int j = 0; j < grid[0].length; ++j){
                if (i == 0 && j == 0)
                    continue;
                else if (i == 0)
                    grid[i][j] += grid[i][j-1];
                else if (j == 0)
                    grid[i][j] += grid[i-1][j];
                else
                    grid[i][j] += Math.min(grid[i-1][j], grid[i][j-1]);
            }
        }
        return grid[grid.length - 1][grid[0].length - 1];
    }

    //66
    public int[] plusOne(int[] digits) {
        if (digits == null || digits.length == 0)
            return digits;
        int i = digits.length - 1;
        while (i >= 0 && digits[i] == 9)
            digits[i--] = 0;
        if (i < 0){
            int[] res = new int[digits.length + 1];
            res[0] = 1;
            return res;
        }
        ++digits[i];
        return digits;
    }

    //67
    public String addBinary(String a, String b) {
        if (a == null || a.length() == 0)
            return b;
        if (b == null || b.length() == 0)
            return a;
        StringBuilder sb = new StringBuilder();
        int i = a.length() - 1, j = b.length() - 1, carry = 0;
        while (i >= 0 || j >= 0 || carry != 0){
            int sum = (i >= 0? a.charAt(i--) - '0': 0) + (j >= 0? b.charAt(j--) - '0': 0) + carry;
            sb.append(sum % 2);
            carry = sum / 2;
        }
        return sb.reverse().toString();
    }

    //70
    public int climbStairs(int n) {
        if (n < 0)
            return 0;
        int n0 = 1, n1 = 1, nn = 1;
        while (n-- >= 2){
            nn = n0 + n1;
            n0 = n1;
            n1 = nn;
        }
        return nn;
    }

    //71
    public String simplifyPath(String path) {
        if (path == null || path.length() == 0)
            return path;
        String[] tokens = path.split("/");
        Deque<String> st = new ArrayDeque<>();
        StringBuilder sb = new StringBuilder();

        for (String s : tokens){
            if (s.equals("") || s.equals("."))
                continue;
            else if (s.equals("..")){
                if (!st.isEmpty())
                    st.pop();
            }
            else {
                st.push(s);
            }
        }
        if (st.isEmpty())
            return "/";
        while (!st.isEmpty()){
            sb.insert(0, "/" + st.pop());
        }
        return sb.toString();
    }

    //73
    public void setZeroes(int[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0)
            return;
        boolean firstRow = false, firstCol = false;
        for (int j = 0; j < matrix[0].length; ++j) {
            if (matrix[0][j] == 0) {
                firstRow = true;
                break;
            }
        }
        for (int i = 0; i < matrix.length; ++i){
            if (matrix[i][0] == 0) {
                firstCol = true;
                break;
            }
        }
        for (int i = 0; i < matrix.length; ++i){
            for (int j = 0; j < matrix[0].length; ++j){
                if (matrix[i][j] == 0){
                    matrix[i][0] = 0;
                    matrix[0][j] = 0;
                }
            }
        }
        for (int i = 1; i < matrix.length; ++i){
            for (int j = 1; j < matrix[0].length; ++j){
                if (matrix[i][0] == 0 || matrix[0][j] == 0)
                    matrix[i][j] = 0;
            }
        }
        if (firstRow){
            for (int j = 0; j < matrix[0].length; ++j) {
                matrix[0][j] = 0;
            }
        }
        if (firstCol){
            for (int i = 0; i < matrix.length; ++i) {
                matrix[i][0] = 0;
            }
        }
    }

    //74
    public boolean searchMatrix(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0)
            return false;
        int l = 0, r = matrix.length - 1, m;
        while (l < r){
            m = l + ((r - l) >> 1);
            if (target > matrix[m][matrix[0].length - 1])
                l = m + 1;
            else if (target < matrix[m][matrix[0].length - 1])
                r = m;
            else
                return true;
        }
        int row = l;
        r = matrix[row].length - 1;
        l = 0;
        while (l <= r){
            m = l + ((r - l) >> 1);
            if (target > matrix[row][m])
                l = m + 1;
            else if (target < matrix[row][m])
                r = m - 1;
            else
                return true;
        }
        return false;
    }

    //75
    public void sortColors(int[] nums) {
        if (nums == null || nums.length == 0)
            return;
        int i0 = 0, i1 = 0, i2 = nums.length - 1;
        while (i1 <= i2){
            if (nums[i1] == 0){
                int t = nums[i0];
                nums[i0] = nums[i1];
                nums[i1] = t;
                ++i0;
                ++i1;
            }
            else if (nums[i1] == 2){
                nums[i1] = nums[i2];
                nums[i2--] = 2;
            }
            else
                ++i1;
        }
    }

    //78
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        if (nums == null)
            return res;
        res.add(new ArrayList<>());
        for (int i : nums){
            //for (List<Integer> l : res){ foreach loop cannot modify underlyint list ->java.util.ConcurrentModificationException
            int size = res.size();
            for (int j = 0; j < size; ++j){
                List<Integer> nl = new ArrayList<>(res.get(j));
                nl.add(i);
                res.add(nl);
            }
        }
        return res;
    }

    //79
    public boolean exist(char[][] board, String word) {
        if (board == null || board.length == 0 || board[0].length == 0 || word.length() == 0)
            return false;
        for (int i = 0; i < board.length; ++i){
            for (int j = 0; j < board[0].length; ++j){
                if (existHelper(board, i, j, word, 0))
                    return true;
            }
        }
        return false;
    }

    private boolean existHelper(char[][] board, int i, int j, String word, int k){
        if (k == word.length())
            return true;
        if (i < 0 || i >= board.length || j < 0 || j >= board[0].length || (board[i][j] & 256) != 0 || board[i][j] != word.charAt(k))
            return false;

        board[i][j] ^= 256;
        boolean ex = existHelper(board, i - 1, j, word, k + 1)
                || existHelper(board, i + 1, j, word, k + 1)
                || existHelper(board, i, j - 1, word, k + 1)
                || existHelper(board, i, j + 1, word, k + 1);
        board[i][j] ^= 256;
        return ex;
    }

    //80
    public int removeDuplicates2(int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;
        boolean dup = false;
        int l = 0, r = 1;
        while (r < nums.length){
            if (nums[l] != nums[r]) {
                nums[++l] = nums[r];
                dup = false;
            }
            else if (!dup) {
                dup = true;
                nums[++l] = nums[r];
            }
            ++r;
        }
        return l + 1;
    }

    //81
    public boolean search2(int[] nums, int target) {
        if (nums == null || nums.length == 0)
            return false;
        int l = 0, r = nums.length - 1, m;
        while (l <= r){
            m = l + ((r - l) >> 1);
            if (nums[m] == target)
                return true;
            else if (nums[m] < nums[r]){
                if (target > nums[m] && target <= nums[r])
                    l = m + 1;
                else
                    r = m - 1;
            }
            else if (nums[m] > nums[r]){
                if (target >= nums[l] && target < nums[m])
                    r = m - 1;
                else
                    l = m + 1;
            }
            else
                --r;
        }
        return false;
    }

    //82
    public ListNode deleteDuplicates2(ListNode head) {
        if (head == null || head.next == null)
            return head;
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode pre = dummy, cur = head;
        while (cur != null){
            while (cur.next != null && cur.val == cur.next.val)
                cur = cur.next;
            if (pre.next != cur)
                pre.next = cur.next;
            else
                pre = pre.next;
            cur = cur.next;
        }
        return dummy.next;
    }

    //83
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null)
            return head;
        ListNode cur = head;
        while (cur != null && cur.next != null){
            if (cur.val == cur.next.val)
                cur.next = cur.next.next;
            else
                cur = cur.next;
        }
        return head;
    }

    //88
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        if (nums1 == null || nums2 == null || m < 0 || n < 0) //m == 0 is ok as long as n has data
            return;
        int im = m - 1, in = n - 1, i = m + n - 1;
        while (im >= 0 || in >= 0){
            if (im >= 0 && in >= 0){
                nums1[i--] = nums1[im] > nums2[in]? nums1[im--]: nums2[in--];
            }
            else if (in >= 0){
                nums1[i--] = nums2[in--];
            }
            else
                break;
        }
    }


    //89
    public List<Integer> grayCode(int n) {
        List<Integer> res = new ArrayList<>();
        if (n < 0)
            return res;
        for (int i = 0; i < (1<<n); ++i){
            res.add(i ^ (i >> 1));
        }
        return res;
    }

    //90
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        if (nums == null)
            return res;
        res.add(new ArrayList<>());
        Arrays.sort(nums);
        int size = 1;
        for (int i = 0; i < nums.length; ++i){
            int start = i > 0 && nums[i] == nums[i-1] ? size: 0;
            size = res.size();
            for (int j = start; j < size; ++j){
                List<Integer> nl = new ArrayList<>(res.get(j));
                nl.add(nums[i]);
                res.add(nl);
            }
        }
        return res;
    }

    //94
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null)
            return res;
        Deque<TreeNode> st = new ArrayDeque<>();
        while (root != null || !st.isEmpty()){
            if (root != null){
                st.push(root);
                root = root.left;
            }
            else {
                TreeNode tn = st.pop();
                res.add(tn.val);
                root = tn.right;
            }
        }
        return res;
    }

    //95
    public List<TreeNode> generateTrees(int n) {
        if (n < 1)
            return new ArrayList<TreeNode>();
        return generateTrees(1, n);
    }

    private List<TreeNode> generateTrees(int l, int r){
        List<TreeNode> res = new ArrayList<>();
        if (l > r){
            res.add(null);
            return res;
        }
        for (int i = l; i <= r; ++i){
            List<TreeNode> ll = generateTrees(l, i - 1);
            List<TreeNode> rl = generateTrees(i + 1, r);

            for (int j = 0; j < ll.size(); ++j){
                for (int k = 0; k < rl.size(); ++k){
                    TreeNode root = new TreeNode(i); //for every combination, create new root and pick left/right
                    root.left = ll.get(j);
                    root.right = rl.get(k);
                    res.add(root);
                }
            }
        }
        return res;
    }

    //96
    public int numTrees(int n) {
        if (n < 1)
            return 0;
        int[] dp = new int[n + 1];
        dp[0] = dp[1] = 1;
        for (int i = 2; i <= n; ++i){
            for (int l = 0; l < i; ++l)
                dp[i] += dp[l]*dp[i-1-l];
        }
        return dp[n];
    }

    //98
    public boolean isValidBST(TreeNode root) {
        return isValidBSTHelper(root, Long.MIN_VALUE, Long.MAX_VALUE);
    }

    private boolean isValidBSTHelper(TreeNode root, long min, long max){
        if (root == null)
            return true;
        if (root.val <=min || root.val >= max)
            return false;
        return isValidBSTHelper(root.left, min, root.val) && isValidBSTHelper(root.right, root.val, max);
    }

    //102
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null)
            return res;
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        int cur = 1, next = 0;
        List<Integer> lvl = new ArrayList<>();

        while (!q.isEmpty()){
            TreeNode tn = q.poll();
            lvl.add(tn.val);
            if (tn.left != null){
                q.offer(tn.left);
                ++next;
            }
            if (tn.right != null){
                q.offer(tn.right);
                ++next;
            }
            if (--cur == 0){
                res.add(lvl);
                lvl = new ArrayList<>();
                cur = next;
                next = 0;
            }
        }
        return res;
    }

    //103
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null)
            return res;
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        int cur = 1, next = 0;
        boolean rev = false;
        List<Integer> lvl = new ArrayList<>();
        while (!q.isEmpty()){
            TreeNode tn = q.poll();
            lvl.add(tn.val);
            if (tn.left != null){
                q.offer(tn.left);
                ++next;
            }
            if (tn.right != null){
                q.offer(tn.right);
                ++next;
            }
            if (--cur == 0){
                if (rev)
                    Collections.reverse(lvl);
                rev = !rev;
                res.add(lvl);
                lvl = new ArrayList<>();
                cur = next;
                next = 0;
            }
        }
        return res;
    }

    //104
    public int maxDepth(TreeNode root) {
        if (root == null)
            return 0;
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;

    }

    //107
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null)
            return res;
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        int cur = 1, next = 0;
        List<Integer> lvl = new ArrayList<>();

        while (!q.isEmpty()){
            TreeNode tn = q.poll();
            lvl.add(tn.val);
            if (tn.left != null){
                q.offer(tn.left);
                ++next;
            }
            if (tn.right != null){
                q.offer(tn.right);
                ++next;
            }
            if (--cur == 0){
                res.add(lvl);
                lvl = new ArrayList<>();
                cur = next;
                next = 0;
            }
        }
        Collections.reverse(res);
        return res;
    }

    //108
    public TreeNode sortedArrayToBST(int[] nums) {
        if (nums == null || nums.length == 0)
            return null;
        return sortedArrayToBSTHelper(nums, 0, nums.length - 1);
    }

    private TreeNode sortedArrayToBSTHelper(int[] nums, int l, int r){
        if (l > r)
            return null;
        int m = l + ((r - l) >> 1);
        TreeNode root = new TreeNode(nums[m]);
        root.left = sortedArrayToBSTHelper(nums, l, m - 1);
        root.right = sortedArrayToBSTHelper(nums, m + 1, r);
        return root;
    }

    //109
    private ListNode inListH;
    public TreeNode sortedListToBST(ListNode head) {
        if (head == null)
            return null;
        ListNode cur = head;
        int len = 0;
        inListH = head;
        while (cur != null){
            ++len;
            cur = cur.next;
        }
        return sortedListToBSTHelper(0, len - 1);
    }

    private TreeNode sortedListToBSTHelper(int l, int r){
        if (l > r)
            return null;
        int m = l + ((r - l) >> 1);
        TreeNode lc = sortedListToBSTHelper(l, m - 1);
        TreeNode root = new TreeNode(inListH.val);
        inListH = inListH.next;
        root.left = lc;
        root.right = sortedListToBSTHelper(m + 1, r);
        return root;
    }

    //110
    public boolean isBalanced(TreeNode root) {
        if (root == null)
            return true;
        return isBalancedHelper(root) != -1;
    }

    private int isBalancedHelper(TreeNode root){
        if (root == null)
            return 0;
        int lh = isBalancedHelper(root.left);
        if (lh == -1)
            return -1;
        int rh = isBalancedHelper(root.right);
        if (rh == -1)
            return -1;
        if (Math.abs(lh - rh) > 1)
            return -1;
        return Math.max(lh, rh) + 1;
    }

    //111
    public int minDepth(TreeNode root) {
        if (root == null)
            return 0;
        if (root.left == null)
            return minDepth(root.right) + 1;
        else if (root.right == null)
            return minDepth(root.left) + 1;
        else
            return Math.min(minDepth(root.left), minDepth(root.right)) + 1;
    }

    //112
    public boolean hasPathSum(TreeNode root, int sum) {
        if (root == null)
            return false;
        if (root.left == null && root.right == null)
            return root.val == sum;
        if (hasPathSum(root.left, sum - root.val) || hasPathSum(root.right, sum - root.val))
            return true;
        return false;
    }

    //113
    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null)
            return res;
        pathSumHelper(root, sum, new ArrayList<Integer>(), res);
        return res;
    }

    private void pathSumHelper(TreeNode root, int sum, List<Integer> combi, List<List<Integer>> res){
        if (root == null)
            return;
        if (root.left == null && root.right == null){
            if (sum == root.val){
                List<Integer> l = new ArrayList<>(combi);
                l.add(root.val);
                res.add(l);
            }
            return;
        }
        combi.add(root.val);
        pathSumHelper(root.left, sum - root.val, combi, res);
        pathSumHelper(root.right, sum - root.val, combi, res);
        combi.remove(combi.size() - 1);
    }

    //114
    TreeNode pre;
    public void flatten(TreeNode root) {
        if (root == null)
            return;
        if (pre != null) {
            pre.left = null;
            pre.right = root;
        }
        pre = root;
        TreeNode t = root.right;
        flatten(root.left);
        flatten(t);
    }

    //118
    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> res = new ArrayList<>();
        if (numRows < 1)
            return res;
        res.add(Arrays.asList(1));
        for (int i = 2; i <= numRows; ++i){
            List<Integer> combi = new ArrayList<>();
            combi.add(1);
            for (int j = 0; j < res.get(i-2).size() - 1; ++j){ //1-based, so i - 1 - 1 is the last row
                combi.add(res.get(i-2).get(j) + res.get(i-2).get(j+1));
            }
            combi.add(1);
            res.add(combi);
        }
        return res;
    }

    //121
    public int maxProfit(int[] prices) {
        if (prices == null || prices.length < 2)
            return 0;
        int min = prices[0], max = 0;
        for (int i = 1; i < prices.length; ++i){
            max = Math.max(max, prices[i] - min);
            min = Math.min(min, prices[i]);
        }
        return max;
    }

    //116
    public void connect(TreeLinkNode root) {
        if (root == null || (root.left == null && root.right == null))
            return;

        root.left.next = root.right;
        root.right.next = root.next!= null? root.next.left: null;
        connect(root.left);
        connect(root.right);
    }

    //117
    class TreeLinkNode {
        int val;
        TreeLinkNode left, right, next;
        public TreeLinkNode(int x){val = x;}
    }

    public void connect2(TreeLinkNode root) {
        if (root == null)
            return;
        //find next valid
        TreeLinkNode nv = root.next;
        while (nv != null && nv.left == null && nv.right == null){
            nv = nv.next;
        }
        if (nv != null)
            nv = nv.left != null? nv.left: nv.right;
        if (root.right != null)
            root.right.next = nv;
        if (root.left != null)
            root.left.next = root.right != null? root.right: nv;
        connect(root.right);
        connect(root.left);
    }

    //122
    public int maxProfit2(int[] prices) {
        if (prices == null || prices.length < 2)
            return 0;
        int res = 0;
        for (int i = 1; i < prices.length; ++i){
            if (prices[i] > prices[i-1])
                res += prices[i] - prices[i-1];
        }
        return res;
    }

    //124
    private int maxPath;
    public int maxPathSum(TreeNode root) {
        if (root == null)
            return 0;
        maxPath = root.val;
        maxPathSumHelper(root);
        return this.maxPath;
    }

    private int maxPathSumHelper(TreeNode root){
        if (root == null)
            return 0;
        int lsum = Math.max(maxPathSumHelper(root.left), 0);
        int rsum = Math.max(maxPathSumHelper(root.right), 0);
        if (lsum + rsum + root.val > maxPath)
            maxPath = lsum + rsum + root.val;
        return root.val + Math.max(lsum, rsum);
    }

    //125
    public boolean isPalindrome(String s) {
        if (s == null || s.length() == 0)
            return true;
        int l = 0, r = s.length() - 1;
        while (l < r){
            if (!Character.isLetterOrDigit(s.charAt(l)))
                ++l;
            else if (!Character.isLetterOrDigit(s.charAt(r)))
                --r;
            else if(Character.toLowerCase(s.charAt(l++)) != Character.toLowerCase(s.charAt(r--)))
                return false;
        }
        return true;
    }

    //128
    public int longestConsecutive(int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;
        //need o(n) so map to a space where needs o(1) and treat each number as isolated islands of graph
        Set<Integer> hs = new HashSet<>();
        int res = 1;
        for (int i : nums)
            hs.add(i);
        //dfs
        Iterator<Integer> iter = hs.iterator();
        while (iter.hasNext()){
            int x = iter.next();
            hs.remove(x);
            int len = 1;
            int left = x - 1, right = x + 1;
            while (hs.contains(left)){
                hs.remove(left--);
                ++len;
            }
            while (hs.contains(right)){
                hs.remove(right++);
                ++len;
            }
            res = Math.max(res, len);
            iter = hs.iterator();
        }
        return res;
    }

    //129
    private int sumNumbers;
    public int sumNumbers(TreeNode root) {
        if (root == null)
            return 0;
        sumNumbersHelper(root, 0);
        return this.sumNumbers;
    }

    private void sumNumbersHelper(TreeNode root, int pre){
        if (root == null)
            return;
        if (root.left == null && root.right == null){
            this.sumNumbers += pre + root.val;
            return;
        }
        pre = 10 * (pre + root.val);
        sumNumbersHelper(root.left, pre);
        sumNumbersHelper(root.right, pre);
    }

    //130
    public void solve(char[][] board) {
        if (board == null || board.length == 0 || board[0].length == 0)
            return;
        for (int j = 0; j < board[0].length; ++j) {
            if (board[0][j] == 'O')
                floodFill(board, 0, j);
            if (board[board.length - 1][j] == 'O')
                floodFill(board, board.length - 1, j);
        }
        for (int i = 0; i < board.length; ++i) {
            if (board[i][0] == 'O')
                floodFill(board, i, 0);
            if (board[i][board[0].length - 1] == 'O')
                floodFill(board, i, board[0].length - 1);
        }
        for (int i = 0; i < board.length; ++i){
            for (int j = 0; j < board[0].length; ++j){
                if (board[i][j] == 'O')
                    board[i][j] = 'X';
                if (board[i][j] == '#')
                    board[i][j] = 'O';
            }
        }
    }

    private void floodFill(char[][] board, int i, int j){
        Queue<Integer> q = new LinkedList<>();
        board[i][j] = '#';
        q.offer(i * board[0].length + j);
        int[] xo = {-1, 1,  0, 0};
        int[] yo = {0,  0, -1, 1};
        while (!q.isEmpty()) {
            int n = q.poll();
            int nx = n / board[0].length, ny = n % board[0].length;
            for (int k = 0; k < xo.length; ++k) {
                int x = nx + xo[k], y = ny + yo[k];
                if (x >= 0 && x < board.length && y >= 0 && y < board[0].length && board[x][y] == 'O'){
                    board[x][y] = '#';
                    q.offer(x * board[0].length + y);
                }
            }
        }
    }

    //131
    public List<List<String>> partition(String s) {
        List<List<String>> res = new ArrayList<>();
        if (s == null || s.length() == 0)
            return res;
        boolean[][] dp = new boolean[s.length()][s.length()];

        for (int i = s.length() - 1; i >= 0; --i){
            for (int j = i; j < s.length(); ++j){
                if (s.charAt(i) == s.charAt(j) && (j - i <= 2 || dp[i+1][j-1]))
                    dp[i][j] = true;
            }
        }

        partitionHelper(s, 0, dp, new ArrayList<String>(), res);
        return res;
    }

    private void partitionHelper(String s, int i, boolean[][] dp, List<String> combi, List<List<String>> res){
        if (i == s.length()){
            res.add(new ArrayList<>(combi));
            return;
        }
        for (int k = i; k < s.length(); ++k){
            if (dp[i][k]){
                combi.add(s.substring(i, k + 1));
                partitionHelper(s, k + 1, dp, combi, res);
                combi.remove(combi.size() - 1);
            }
        }
    }

    //132
    public int minCut(String s) {
        if (s == null || s.length() == 0)
            return 0;
        boolean[][] dp = new boolean[s.length()][s.length()];
        for (int i = s.length() - 1; i >= 0; --i){
            for (int j = i; j < s.length(); ++j){
                if (s.charAt(i) == s.charAt(j) && (j - i <= 2 || dp[i+1][j-1]))
                    dp[i][j] = true;
            }
        }
        int[] dp1 = new int[s.length() + 1];
        Arrays.fill(dp1, Integer.MAX_VALUE);
        dp1[0] = 0;
        for (int i = 0; i < s.length(); ++i){
            for (int j = i; j < s.length(); ++j){
                if (dp[i][j])
                    dp1[j+1] = Math.min(dp1[j+1], dp1[i] + 1);
            }
        }
        return dp1[dp1.length - 1] - 1;
    }

    //135
    public int candy(int[] ratings) {
        if (ratings == null || ratings.length == 0)
            return 0;
        int[] candy = new int[ratings.length];

        Arrays.fill(candy, 1);
        for (int i = 1; i < candy.length; ++i){
            if (ratings[i] > ratings[i-1])
                candy[i] = candy[i-1] + 1;
        }
        int res = candy[candy.length - 1];
        for (int i = candy.length - 2; i >= 0; --i){
            if (ratings[i] > ratings[i+1] && candy[i] <= candy[i+1])
                candy[i] = candy[i+1] + 1;
            res += candy[i];
        }
        return res;
    }

    //136
    public int singleNumber(int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;
        int res = 0;
        for (int i : nums)
            res ^= i;
        return res;
    }

    //138
    class RandomListNode{
        int label;
        RandomListNode next, random;
        RandomListNode(int val){label = val;}
    }

    public RandomListNode copyRandomList(RandomListNode head) {
        if (head == null)
            return head;
        RandomListNode cur = head;
        while (cur != null){
            RandomListNode next = cur.next;
            cur.next = new RandomListNode(cur.label);
            cur.next.next = next;
            cur = cur.next.next;
        }
        cur = head;
        while (cur != null){
            cur.next.random = cur.random == null? null : cur.random.next; //random could be null!!!
            cur = cur.next.next;
        }
        RandomListNode dummy = new RandomListNode(0), pre = dummy;
        cur = head;
        while (cur != null){
            pre.next = cur.next;
            pre = pre.next;
            cur.next = cur.next.next;
            cur = cur.next;
        }
        return dummy.next;
    }

    //139
    public boolean wordBreak(String s, Set<String> wordDict) {
        if (s == null || wordDict == null || wordDict.size() == 0)
            return false;
        boolean[] dp = new boolean[s.length()+1];
        dp[0] = true;
        for (int i = 0; i < s.length(); ++i){
            if (dp[i]){ //need before it just a word
                for (int j = i+1; j <= s.length(); ++j){
                    if (wordDict.contains(s.substring(i,j)))
                        dp[j] = true;
                }
            }
        }
        return dp[dp.length - 1];
    }

    //141
    public boolean hasCycle(ListNode head) {
        if (head == null)
            return false;
        ListNode slow, fast;
        slow = fast = head;
        while (fast != null && fast.next != null){
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast)
                return true;
        }
        return false;
    }

    //142
    public ListNode detectCycle(ListNode head) {
        if (head == null)
            return head;
        ListNode slow, fast;
        slow = fast = head;
        boolean hasCycle = false;
        while (fast != null && fast.next != null){
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast){
                hasCycle = true;
                break;
            }
        }
        if (!hasCycle)
            return null;
        slow = head;
        while (slow != fast){
            slow = slow.next;
            fast = fast.next;
        }
        return fast;
    }


    //144
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null)
            return res;
        Deque<TreeNode> st = new ArrayDeque<>();
        while (root != null || !st.isEmpty()){
            if (root != null){
                res.add(root.val);
                if (root.right != null)
                    st.push(root.right);
                root = root.left;
            }
            else {
                root = st.pop();
            }
        }
        return res;
    }

    //150
    public int evalRPN(String[] tokens) {
        if (tokens == null || tokens.length == 0)
            return 0;
        Deque<Integer> st = new ArrayDeque<>();
        String op = "+-*/";

        for (String s : tokens){
            if (!op.contains(s))
                st.push(Integer.parseInt(s));
            else {
                if (st.size() < 2)
                    return -1;
                int y = st.pop();
                int x = st.pop();
                switch(s){
                    case "+":
                        st.push(x + y);
                        break;
                    case "-":
                        st.push(x - y);
                        break;
                    case "*":
                        st.push(x * y);
                        break;
                    case "/":
                        st.push(x / y);
                        break;
                    default:
                        break;
                }
            }
        }
        return st.pop();
    }

    //151
    public String reverseWords(String s) {
        if (s == null)
            return s;
        s = s.trim();
        if (s.length() == 0)
            return s;
        String[] tokens = s.split("\\s+");
        StringBuilder sb = new StringBuilder();
        sb.append(tokens[tokens.length - 1]);
        for (int i = tokens.length - 2; i >= 0; --i) { //java array does not have reverse(). Collections has.
            sb.append(" ");
            sb.append(tokens[i]);
        }
        return sb.toString();
    }

    //152
    public int maxProduct(int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;
        int min = nums[0], max = nums[0], res = nums[0];
        for (int i = 1; i < nums.length; ++i){
            int t = max;
            max = Math.max(nums[i], Math.max(nums[i] * max, nums[i] * min));
            min = Math.min(nums[i], Math.min(nums[i] * t, nums[i] * min ));
            res = Math.max(res, max);
        }
        return res;
    }

    //153
    public int findMin(int[] nums) {
        if (nums == null || nums.length == 0)
            return -1;
        int l = 0, r = nums.length - 1, m;
        while (l < r){
            m = l + ((r - l) >> 1);
            if (nums[m] > nums[r])
                l = m + 1;
            else
                r = m;
        }
        return nums[l];
    }

    //154
    public int findMin2(int[] nums) {
        if (nums == null || nums.length == 0)
            return -1;
        int l = 0, r = nums.length - 1, m;
        while (l < r){
            m = l + ((r - l) >> 1);
            if (nums[m] > nums[r])
                l = m + 1;
            else if (nums[m] < nums[r])
                r = m;
            else
                --r;
        }
        return nums[l];
    }

    //155
    public class MinStack {
        Deque<Integer> st;
        Deque<Integer> mt;
        /** initialize your data structure here. */
        public MinStack() {
            st = new ArrayDeque<>();
            mt = new ArrayDeque<>();
        }

        public void push(int x) {
            st.push(x);
            if (mt.isEmpty() || x <= mt.peek())
                mt.push(x);
        }

        public void pop() {
            int x = st.pop();
            if (x == mt.peek())
                mt.pop();
        }

        public int top() {
            return st.peek();
        }

        public int getMin() {
            return mt.peek();
        }
    }

    //160
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null)
            return null;
        int n1 = 0, n2 = 0;
        ListNode ha = headA, hb = headB;
        while (ha != null){
            ++n1;
            ha = ha.next;
        }
        while (hb != null){
            ++n2;
            hb = hb.next;
        }
        ha = headA;
        hb = headB;
        while (n1 - n2 > 0){
            ha = ha.next;
            --n1;
        }
        while (n2 - n1 > 0){
            hb = hb.next;
            --n2;
        }
        while (ha != null && ha != hb){
            ha = ha.next;
            hb = hb.next;
        }
        return ha;
    }

    //161
    public boolean isOneEditDistance(String s, String t) {
        if (s == null || t == null)
            return false;
        if (s.length() > t.length())
            return isOneEditDistance(t, s);
        int diff = t.length() - s.length();
        if (diff > 1)
            return false;
        int i = 0;
        while (i < s.length() && s.charAt(i) == t.charAt(i))
            ++i;
        if (i == s.length())
            return diff == 1;
        if (diff == 0)
            ++i;
        while (i < s.length() && s.charAt(i) == t.charAt(i + diff))
            ++i;
        return i == s.length();
    }

    //162
    public int findPeakElement(int[] nums) {
        if (nums == null || nums.length == 0)
            return -1;
        int l = 0, r = nums.length - 1, m;
        while (l <= r){
            m = l + ((r - l) >> 1);
            if ((m == 0 || nums[m] > nums[m - 1]) && (m == nums.length - 1 || nums[m] > nums[m + 1]))
                return m;
            else if (m != nums.length - 1 && nums[m] < nums[m + 1]) //peek will be either right
                l = m + 1;
            else if (m != 0 && nums[m] < nums[m - 1]) //or left
                r = m - 1;
        }
        return l;
    }

    //163
    public List<String> findMissingRanges(int[] nums, int lower, int upper) {
        List<String> res = new ArrayList<>();
        if (nums == null || nums.length == 0){
            res.add(findMissingRangesHelper(lower, upper)); //first need summary full if input is empty
            return res;
        }
        if (nums[0] > lower)
            res.add(findMissingRangesHelper(lower, nums[0]-1)); //note helper function should be inclusive, as lower and uuper are inclusive
        for (int i = 1; i < nums.length; ++i){
            if (nums[i] > nums[i-1] + 1) //cannot use nums[i] - nums[i-1] see int.max - (-int.min) will overflow
                res.add(findMissingRangesHelper(nums[i-1]+1, nums[i]-1));
        }
        if (upper > nums[nums.length - 1])
            res.add(findMissingRangesHelper(nums[nums.length - 1]+1, upper)); //check int.max
        return res;
    }

    private String findMissingRangesHelper(int l, int r){ //should take care of only missing ranges
        if (r == l)
            return Integer.toString(l);
        else
            return l + "->" + r;
    }

    //165
    public int compareVersion(String version1, String version2) {
        if (version1 == null || version2 == null)
            return 0;
        String[] v1 = version1.split("\\.");
        String[] v2 = version2.split("\\.");

        for (int i = 0; i < Math.max(v1.length, v2.length); ++i){
            int diff = (i < v1.length? Integer.parseInt(v1[i]): 0) - (i < v2.length? Integer.parseInt(v2[i]): 0);
            if (diff != 0)
                return diff > 0? 1: -1;
        }
        return 0;
    }

    //167
    public int[] twoSum(int[] numbers, int target) {
        int[] res = {-1, -1};
        if (numbers == null || numbers.length < 2)
            return res;
        int l = 0, r = numbers.length - 1;
        while (l < r){
            int sum = numbers[l] + numbers[r];
            if (sum < target)
                ++l;
            else if (sum > target)
                --r;
            else {
                res[0] = l + 1;
                res[1] = r + 1;
                break;
            }
        }
        return res;
    }

    //169
    public int majorityElement(int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;
        int cnt = 1, major = nums[0];
        for (int i = 1; i < nums.length; ++i){
            if (nums[i] == major)
                ++cnt;
            else if (--cnt == 0){
                cnt = 1;
                major = nums[i];
            }
        }
        return major;
    }

    //170
    public class TwoSum {
        Map<Integer, Integer> hm = new HashMap<>();

        // Add the number to an internal data structure.
        public void add(int number) {
            hm.put(number, hm.containsKey(number)? 2: 1);
        }

        // Find if there exists any pair of numbers which sum is equal to the value.
        public boolean find(int value) {
            for (int x : hm.keySet()){
                int y = value - x;
                if (hm.containsKey(y) && (x != y || hm.get(x) > 1))
                    return true;
            }
            return false;
        }
    }

    //171
    public int titleToNumber(String s) {
        if (s == null || s.length() == 0)
            return 0;
        int res = 0;
        for (int i = 0; i < s.length(); ++i){
            res = res * 26 + s.charAt(i) - 'A' + 1;
        }
        return res;
    }

    //173
    public class BSTIterator {
        Deque<TreeNode> stack;
        public BSTIterator(TreeNode root) {
            stack = new ArrayDeque<>();
            pushLeft(root);
        }

        private void pushLeft(TreeNode root){
            while (root != null) {
                stack.push(root);
                root = root.left;
            }
        }

        /** @return whether we have a next smallest number */
        public boolean hasNext() {
            return !stack.isEmpty();
        }

        /** @return the next smallest number */
        public int next() {
            TreeNode tn = stack.pop();
            pushLeft(tn.right);
            return tn.val;
        }
    }

    //179
    public String largestNumber(int[] nums) {
        if (nums == null || nums.length == 0)
            return "";
        String[] strs = new String[nums.length];
        for (int i = 0; i < nums.length; ++i)
            strs[i] = String.valueOf(nums[i]);
        Arrays.sort(strs, (s1, s2)-> (s2 + s1).compareTo(s1 + s2));
        if (strs[0].equals("0")) // prevent nums have all 0
            return "0";
        String res = "";
        for (String s : strs)
            res += s;
        return res;
    }

    //186
    public void reverseWords(char[] s) {
        if (s == null || s.length == 0)
            return;
        reverse(s, 0, s.length - 1);
        int l = 0, r = 0;
        while (r < s.length){
            if (Character.isSpaceChar(s[r])){ //Character.isSpaceChar(char c) is the new API
                reverse(s, l, r - 1);
                l = r + 1;
            }
            ++r;
        }
        reverse(s, l, r - 1);
    }

    private void reverse(char[] s, int l, int r){
        while (l < r)
            swap(s, l++, r--);
    }

    private void swap(char[] s, int l, int r){
        char t = s[l];
        s[l] = s[r];
        s[r] = t;
    }

    //190
    public int reverseBits(int n) {
        int res = 0;
        for (int i = 0; i < 32; ++i){
            res = (res << 1) | (n & 1);
            n >>>= 1;
        }
        return res;
    }

    //199
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null)
            return res;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        int cur = 1, next = 0;

        while (!queue.isEmpty()){
            TreeNode tn = queue.poll();
            if(tn.left != null){
                queue.offer(tn.left);
                ++next;
            }
            if(tn.right != null){
                queue.offer(tn.right);
                ++next;
            }
            if (--cur == 0){
                res.add(tn.val);
                cur = next;
                next = 0;
            }
        }
        return res;
    }

    //200
    public int numIslands(char[][] grid) {
        if (grid == null || grid.length == 0 || grid[0].length == 0)
            return 0;
        int res = 0;
        for (int i = 0; i < grid.length; ++i){
            for (int j = 0; j < grid[0].length; ++j){
                if (grid[i][j] == '1') {
                    numIslandsHelper(grid, i, j);
                    ++res;
                }
            }
        }
        for (int i = 0; i < grid.length; ++i){
            for (int j = 0; j < grid[0].length; ++j){
                grid[i][j] ^= 256;
            }
        }
        return res;
    }

    private void numIslandsHelper(char[][] grid, int i, int j){
        if (i < 0 || i >= grid.length || j < 0 || j >= grid[0].length || grid[i][j] != '1')
            return;
        grid[i][j] ^= 256;
        numIslandsHelper(grid, i - 1, j);
        numIslandsHelper(grid, i + 1, j);
        numIslandsHelper(grid, i, j - 1);
        numIslandsHelper(grid, i, j + 1);
    }

    //203
    public ListNode removeElements(ListNode head, int val) {
        if (head == null)
            return null;
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode pre = dummy;
        while (pre != null && pre.next != null){
            if (pre.next.val == val)
                pre.next = pre.next.next;
            else
                pre = pre.next;
        }
        return dummy.next;
    }

    //205
    public boolean isIsomorphic(String s, String t) {
        if (s == null || t == null || s.length() != t.length())
            return false;
        Map<Character, Character> hm = new HashMap<>();
        for (int i = 0; i < t.length(); ++i){
            if (hm.containsKey(t.charAt(i))){
                if (hm.get(t.charAt(i)) != s.charAt(i))
                    return false;
            }
            else if (hm.containsValue(s.charAt(i)))
                return false;
            else
                hm.put(t.charAt(i), s.charAt(i));
        }
        return true;
    }

    //208
    public class Trie {
        private TrieNode root;

        public Trie() {
            root = new TrieNode();
        }

        // Inserts a word into the trie.
        public void insert(String word) {
            if (word == null || word.length() == 0)
                return;
            TrieNode tr = root;
            for (int i = 0; i < word.length(); ++i) {
                int off = word.charAt(i) - 'a';
                if (tr.children[off] == null)
                    tr.children[off] = new TrieNode();
                tr = tr.children[off];
            }
            tr.isWord = true;
        }

        // Returns if the word is in the trie.
        public boolean search(String word) {
            if (word == null)
                return false;
            TrieNode tr = root;
            for (int i = 0; i < word.length(); ++i){
                int off = word.charAt(i) - 'a';
                if (tr.children[off] == null)
                    return false;
                tr = tr.children[off];
            }
            return tr.isWord;
        }

        // Returns if there is any word in the trie
        // that starts with the given prefix.
        public boolean startsWith(String prefix) {
            if (prefix ==null)
                return false;
            TrieNode tr = root;
            for (int i = 0; i < prefix.length(); ++i){
                int off = prefix.charAt(i) - 'a';
                if (tr.children[off] == null)
                    return false;
                tr = tr.children[off];
            }
            return true;
        }
    }

    //209
    public int minSubArrayLen(int s, int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;
        int l = 0, r= 0, sum = 0, min = nums.length +1;

        while (r < nums.length){
            sum += nums[r];
            while (l <= r && sum >= s){
                    min = Math.min(min, r - l + 1);
                    sum -= nums[l++]; //all the way until sum < s cauze when >= s cannot be even shorter
            }
            ++r; //only add here after try shrinking the left!
        }
        return min > nums.length ? 0 : min;
    }

    //211
    public class WordDictionary {
        TrieNode root = new TrieNode();

        // Adds a word into the data structure.
        public void addWord(String word) {
            if (word == null)
                return;
            TrieNode tn = root;
            for (int i = 0; i < word.length(); ++i){
                char c = word.charAt(i);
                if (tn.children[c - 'a'] == null)
                    tn.children[c - 'a'] = new TrieNode();
                tn = tn.children[c - 'a'];
            }
            tn.isWord = true;
        }

        // Returns if the word is in the data structure. A word could
        // contain the dot character '.' to represent any one letter.
        public boolean search(String word) {
            if (word == null)
                return false;
            return searchHelper(word, 0, root);
        }

        private boolean searchHelper(String word, int i, TrieNode root){
            if (root == null)
                return false;
            if (i == word.length())
                return root.isWord;
            char c = word.charAt(i);
            if (c != '.')
                return searchHelper(word, i + 1, root.children[c - 'a']);
            for (TrieNode tn : root.children){
                if (tn != null && searchHelper(word, i + 1, tn))
                    return true;
            }
            return false;
        }
    }

    //212
    public List<String> findWords(char[][] board, String[] words) {
        List<String> res = new ArrayList<>();
        if (board == null || words == null || words.length == 0)
            return res;
        TrieNode troot = constructTrie(words);
        for (int i = 0; i < board.length; ++i){
            for (int j = 0; j < board[0].length; ++j){
                findWordsHelper(board, i, j, troot, new StringBuilder(), res);
            }
        }
        Set<String> hs = new HashSet<>(); //because possiblly dup
        hs.addAll(res);
        return new ArrayList<String>(hs);
    }

    private TrieNode constructTrie(String[] words){
        TrieNode troot = new TrieNode();
        TrieNode root;
        for (String s : words){
            root = troot;
            for (int i = 0; i < s.length(); ++i){
                if (root.children[s.charAt(i) - 'a'] == null)
                    root.children[s.charAt(i) - 'a'] = new TrieNode();
                root = root.children[s.charAt(i) - 'a'];
            }
            root.isWord = true;
        }
        return troot;
    }

    private void findWordsHelper(char[][] board, int i, int j, TrieNode root, StringBuilder sb, List<String> res){
        if (root.isWord == true){
            res.add(sb.toString());
            //return; shouldnt return here!!!!!
        }
        if (i < 0 || i >= board.length || j < 0 || j >= board[0].length
                || (board[i][j] & 256) != 0 ||root.children[board[i][j] - 'a'] == null)
            return; //check it's not a marked point!!!
        sb.append(board[i][j]);
        root = root.children[board[i][j] - 'a'];
        board[i][j] ^= 256;
        findWordsHelper(board, i - 1, j, root, sb, res);
        findWordsHelper(board, i + 1, j, root, sb, res);
        findWordsHelper(board, i, j - 1, root, sb, res);
        findWordsHelper(board, i, j + 1, root, sb, res);
        board[i][j] ^= 256;
        sb.deleteCharAt(sb.length() - 1);
    }

    //216
    public List<List<Integer>> combinationSum3(int k, int n) {
        List<List<Integer>> res = new ArrayList<>();
        if (k <= 0 || n <= 0)
            return res;
        combinationSum3Helper(n, k, 1, 0, 0, new ArrayList<Integer>(), res);
        return res;
    }

    private void combinationSum3Helper(int n, int k, int s, int i, int sum, List<Integer> combi, List<List<Integer>> res){
        if (i == k){
            if (sum == n){
                res.add(new ArrayList<>(combi));
            }
            return;
        }
        for (int j = s; j <= 9; ++j){
            if (sum + j <= n){
                combi.add(j);
                combinationSum3Helper(n, k, j + 1, i + 1, sum + j, combi, res);
                combi.remove(combi.size() - 1);
            }
        }
    }

    //217
    public boolean containsDuplicate(int[] nums) {
        if (nums == null || nums.length <= 1)
            return false;
        Set<Integer> hs = new HashSet<>();
        for (int i : nums){
            if (hs.contains(i))
                return true;
            hs.add(i);
        }
        return false;
    }

    //219
    public boolean containsNearbyDuplicate(int[] nums, int k) {
        if (nums == null || nums.length <= 1)
            return false;
        Map<Integer, Integer> hm = new HashMap<>();
        for (int i = 0; i < nums.length; ++i){
            if (hm.containsKey(nums[i]) && (i - hm.get(nums[i]) <= k))
                return true;
            hm.put(nums[i], i);
        }
        return false;
    }

    //222
    public int countNodes(TreeNode root) {
        if (root == null)
            return 0;
        int lh = 1, rh = 1;
        TreeNode cur = root;
        while (cur.left != null){
            cur = cur.left;
            ++lh;
        }
        cur = root;
        while (cur.right != null){
            cur = cur.right;
            ++rh;
        }
        if (lh == rh)
            return (1 << lh) - 1;
        else
            return countNodes(root.left) + countNodes(root.right) + 1;
    }

    //226
    public TreeNode invertTree(TreeNode root) {
        if (root == null)
            return null;
        TreeNode t = root.left;
        root.left = root.right;
        root.right = t;
        invertTree(root.left);
        invertTree(root.right);
        return root;
    }

    //228
    public List<String> summaryRanges(int[] nums) {
        List<String> res = new ArrayList<>();
        if (nums == null || nums.length == 0)
            return res;
        int l = 0, r = 1;
        while (r < nums.length){
            while (r < nums.length && nums[r] == nums[r-1]+1)
                ++r;
            if (r == nums.length)
                break;
            res.add(summaryRangesHelper(nums[l], nums[r-1]));
            l = r;
            r+=1;
        }
        res.add(summaryRangesHelper(nums[l], nums[r-1]));
        return res;
    }

    private String summaryRangesHelper(int l, int r){
        if (l == r)
            return Integer.toString(l);
        else
            return l + "->" + r;
    }

    //230
    private int kth;
    private TreeNode kthNode;
    public int kthSmallest(TreeNode root, int k) {
        if (root == null || k <1)
            return -1;
        kthSmallestHelper(root, k);
        return kthNode.val;
    }

    private void kthSmallestHelper(TreeNode root, int k){
        if (root == null)
            return;
        if (kthNode == null)
            kthSmallestHelper(root.left, k);
        if (++kth == k)
            kthNode = root;
        if (kthNode == null)
            kthSmallestHelper(root.right, k);
    }

    //231
    public boolean isPowerOfTwo(int n) {
        return n > 0 && (n & (n - 1)) == 0;
    }

    //235
    public TreeNode lowestCommonAncestorBST(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || p == null || q == null)
            return null;
        while (root != null){
            if (p.val < root.val && q.val < root.val)
                root = root.left;
            else if (p.val > root.val && q.val > root.val)
                root = root.right;
            else
                break;
        }
        return root;
    }

    //236
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || p == null || q == null)
            return null;
        if (root == p || root == q)
            return root;
        TreeNode tl = lowestCommonAncestor(root.left, p, q);
        TreeNode tr = lowestCommonAncestor(root.right, p, q);
        if (tl != null && tr != null)
            return root;
        return tl != null? tl : tr;
    }

    //237
    public void deleteNode(ListNode node) {
        if (node == null)
            return;
        node.val = node.next.val;
        node.next = node.next.next;
    }

    //242
    public boolean isAnagram(String s, String t) {
        if (s == null)
            return t == null;
        if (t == null)
            return false;
        if (s.length() != t.length())
            return false;
        char[] ss = s.toCharArray();
        char[] ts = t.toCharArray();
        Arrays.sort(ss);
        Arrays.sort(ts);
        return new String(ss).equals(new String(ts));
    }

    //246
    public boolean isStrobogrammatic(String num) {
        if (num == null || num.length() == 0)
            return true;
        Map<Character, Character> hm = new HashMap<>();
        hm.put('0', '0');
        hm.put('1', '1');
        hm.put('6', '9');
        hm.put('8', '8');
        hm.put('9', '6');

        int l = 0, r = num.length() - 1;
        while (l <= r){ //need equal like 2
            if (!hm.containsKey(num.charAt(l)) || hm.get(num.charAt(l)) != num.charAt(r)) //onlu use hm get one side
                return false;
            ++l;
            --r;
        }
        return true;
    }

    //247
    public List<String> findStrobogrammatic(int n) {
        List<String> res = new ArrayList<>();
        if (n <= 0)
            return res;
        Map<Character, Character> hm = new HashMap<>();
        hm.put('0', '0');
        hm.put('1', '1');
        hm.put('6', '9');
        hm.put('8', '8');
        hm.put('9', '6');

        findStroHelper(new char[n], 0, n, res, hm);
        return res;
    }

    private void findStroHelper(char[] combi, int i, int n, List<String> res, Map<Character, Character> hm){
        if (i == n/2){
            if (n % 2 == 1){
                char[] single = {'0', '1', '8'};
                for (char s : single){
                    combi[i] = s;
                    res.add(new String(combi));
                }
            }
            else
                res.add(new String(combi));
            return;
        }
        for (char x : hm.keySet()){
            if (i == 0 && x == '0') //'0' cannot be the first one!
                continue;
            combi[i] = x;
            combi[n - i - 1] = hm.get(x);
            findStroHelper(combi, i + 1, n, res, hm);
        }
    }

    //249
    public List<List<String>> groupStrings(String[] strings) {
        List<List<String>> res = new ArrayList<>();
        if (strings == null || strings.length == 0)
            return res;
        Map<String, List<String>> hm = new HashMap<>();
        for (String s : strings){
            if (s == null)
                continue;
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < s.length(); ++i){
                sb.append((s.charAt(i) - s.charAt(0) + 26) % 26); //this is delta no need to adjust for 0-based
                sb.append(" "); //must have deliminitor, like "abc" -> 012 will be ths same as "am"->012
            }
            String key = sb.toString();
            if (!hm.containsKey(key))
                hm.put(key, new ArrayList<>());
            hm.get(key).add(s);
        }
        res.addAll(hm.values()); //List.addAll(Collection)!!!
        return res;
    }

    //251
    public class Vector2D implements Iterator<Integer> {
        List<Iterator<Integer>> iters;
        int index;

        public Vector2D(List<List<Integer>> vec2d) {
            iters = new ArrayList<>();
            for (List l : vec2d){
                if (l.iterator().hasNext())
                    iters.add(l.iterator());
            }
        }

        @Override
        public Integer next() {
            return iters.get(index).next();
        }

        @Override
        public boolean hasNext() {
            if (iters.isEmpty())
                return false;
            if (iters.get(index).hasNext())
                return true;
            if (++index == iters.size())
                return false;
            return true;
        }
    }

    //252
    public boolean canAttendMeetings(Interval[] intervals) {
        if (intervals == null || intervals.length <= 1)
            return true;
        Arrays.sort(intervals, (i1, i2)-> i1.start == i2.start ? i1.end - i2.end : i1.start - i2.start);
        for (int i = 1; i < intervals.length; ++i){
            if (intervals[i].start < intervals[i-1].end)
                return false;
        }
        return true;
    }

    //253
    public int minMeetingRooms(Interval[] intervals) {
        //greedy, sort by start time and then if a new start is after a finish, can add to the same row, otherwise open a new row
        if (intervals == null || intervals.length == 0)
            return 0;
        Arrays.sort(intervals, (i1, i2) -> i1.start == i2.start ? i1.end - i2.end : i1.start - i2.start);
        //use pq to each time grab the earliest available row(last end ended)
        Queue<Integer> pq = new PriorityQueue<>();
        pq.offer(intervals[0].end);

        for (int i = 1; i < intervals.length; ++i){
            if (intervals[i].start >= pq.peek())
                pq.poll();
            pq.offer(intervals[i].end);
        }
        return pq.size();
    }

    //257
    public List<String> binaryTreePaths(TreeNode root) {
        List<String> res = new ArrayList<>();
        if (root == null)
            return res;
        binaryTreePathsHelper(root, "", res);
        return res;
    }

    private void binaryTreePathsHelper(TreeNode root, String pre, List<String> res){
        if (root == null)
            return;
        if (root.left == null && root.right == null){
            res.add(pre + root.val);
            return;
        }
        binaryTreePathsHelper(root.left, pre + root.val + "->", res);
        binaryTreePathsHelper(root.right, pre + root.val + "->", res);
    }

    //259
    public int threeSumSmaller(int[] nums, int target) {
        if (nums == null || nums.length < 3)
            return 0;
        int res = 0;
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 2; ++i){
            int l = i + 1, r = nums.length - 1;
            while (l < r){
                int sum = nums[i] + nums[l] + nums[r];
                if (sum < target){
                    res += r - l;
                    ++l;
                }
                else
                    --r;
            }
        }
        return res;
    }

    //263
    public boolean isUgly(int num) {
        if (num <= 0)
            return false;
        while (num % 2 == 0)
            num /= 2;
        while (num % 3 == 0)
            num /= 3;
        while (num % 5 == 0)
            num /= 5;
        return num == 1;
    }

    //264
    public int nthUglyNumber(int n) {
        if (n < 1)
            return - 1;
        //dp every ugly = a previous ugly *2/3/5, 3 list merge sort
        int[] dp = new int[n];
        dp[0] = 1;
        int i2 = 0, i3 = 0, i5 = 0;
        for (int i = 1; i < n; ++i){
            int m2 = dp[i2] * 2;
            int m3 = dp[i3] * 3;
            int m5 = dp[i5] * 5;

            dp[i] = Math.min(m2, Math.min(m3, m5));
            if (dp[i] == m2)
                ++i2;
            if (dp[i] == m3) //note here cannot use elseif: 2 * 3 = 6, 3 * 2 = 6. there will be dup so we need at both if pass the one.
                ++i3;
            if (dp[i] == m5)
                ++i5;
        }
        return dp[dp.length - 1];
    }

    //266
    public boolean canPermutePalindrome(String s) {
        if (s == null || s.length() <= 1)
            return true;
        Map<Character, Integer> hm = new HashMap<>();
        for (int i = 0; i < s.length(); ++i){
            hm.put(s.charAt(i), hm.containsKey(s.charAt(i))? hm.get(s.charAt(i)) + 1 : 1);
        }
        boolean hasOdd = false;
        for (int v : hm.values()){
            if (v % 2 == 1){
                if (hasOdd)
                    return false;
                hasOdd = true;
            }
        }
        return true;
    }

    //267
//    public List<String> generatePalindromes(String s) {
//        List<String> res = new ArrayList<>();
//        if (s == null || s.length() == 0)
//            return res;
//        //first count odd and even
//
//        Map<Character, Integer> tm = new TreeMap<>();
//        for (int i = 0; i < s.length(); ++i){
//            char c = s.charAt(i);
//            tm.put(c, tm.containsKey(c) ? tm.get(c) + 1 : 1);
//        }
//        char c = 0;
//        StringBuilder sb = new StringBuilder();
//
//        for (Map.Entry<Character, Integer> e : tm.entrySet()){
//
//            if (e.getValue() % 2 == 1){
//                if (c != 0)
//                    return res;
//                c = e.getKey();
//            }
//            else {
//                for (int k = 0; k < e.getValue()/2; ++k)
//                sb.append(e.getKey());
//            }
//        }
//        //permutation2
//        //add odd c
//        // add reverse part
//        return res;
//
//    }

    //268
    public int missingNumber(int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;
        for (int i = 0; i < nums.length; ++i){
            if (i != nums[i] && nums[i] < nums.length){ //since it's distinct, no need for nums[nums[i]] != nums[i] but attention
                                                        //nums[i] itself can out limit [1]
                int t = nums[nums[i]];
                nums[nums[i]] = nums[i];
                nums[i] = t;
                --i;
            }
        }
        for (int i = 0; i < nums.length; ++i){
            if (i != nums[i])
                return i;
        }
        return nums.length;
    }

    //270
    private int closetValue;
    public int closestValue(TreeNode root, double target) {
        if (root == null)
            return 0;
        closetValueHelper(root, target, Double.MAX_VALUE);
        return closetValue;
    }

    private void closetValueHelper(TreeNode root, double target, double diff){
        if (root == null)
            return;
        if (Math.abs(root.val - target) < diff){
            diff = Math.abs(root.val - target);
            this.closetValue = root.val;
        }
        if (target > root.val)
            closetValueHelper(root.right, target, diff);
        else
            closetValueHelper(root.left, target, diff);
    }

    //273
    public String numberToWords(int num) {
        if (num == 0)
            return "Zero";
        String res = "";
        final String [] thousands = {"", "Thousand", "Million", "Billion"};
        int i = 0;
        while (num > 0){
            if (num % 1000 > 0)
                res = numberToWordsHelper(num % 1000) + " " + thousands[i] + " " + res;
            num /= 1000;
            ++i;
        }
        return res.trim();
    }

    private final String[] less20 = {"", "One", "Two", "Three", "Four", "Five",
            "Six", "Seven", "Eight", "Nine", "Ten",
            "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen",
            "Sixteen", "Seventeen", "Eighteen", "Nineteen", "Twenty"};
    private final String[] tens = {"", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"};
    private String numberToWordsHelper(int num){
        if (num <= 20){
            return less20[num];
        }
        else if (num < 100){
            return tens[num / 10] + (num % 10 == 0 ? "": " " + less20[num % 10]);
        }
        else
            return less20[num / 100] + " Hundred" + (num % 100 == 0? "": " " + numberToWordsHelper(num % 100));
    }

    //277
    boolean knows(int a, int b){return true;}

    public int findCelebrity(int n) {
        if (n <= 1)
            return n;
        int l = 0, r = n - 1;
        while (l < r){
            if (knows(l, r))
                ++l;
            else
                --r;
        }
        for (int i = 0; i < n; ++i){
            if (i != l && (!knows(i, l) || knows(l, i)))
                return -1;
        }
        return l;
    }

    //278
    boolean isBadVersion(int version){return true;}

    public int firstBadVersion(int n) {
        if (n < 1)
            return 0;
        int l = 1, r = n, m;
        while (l <= r){
            m = l + ((r - l) >> 1);
            if (isBadVersion(m))
                r = m - 1;
            else
                l = m + 1;
        }
        return l;
    }

    //279
    public int numSquares(int n) {
        if (n < 1)
            return 0;
        int[] dp = new int[n+1];
        Arrays.fill(dp, Integer.MAX_VALUE);
        dp[0] = 0;
        for (int i = 0; i < n; ++i){
            for (int j = 1; i + j*j <= n; ++j){
                dp[i+j*j] = Math.min(dp[i+j*j], dp[i] + 1);
            }
        }
        return dp[dp.length - 1];
    }

    //281
    public class ZigzagIterator {
        List<Iterator<Integer>> iters;
        int index;

        public ZigzagIterator(List<Integer> v1, List<Integer> v2) {
            iters = new ArrayList<>(); //dont forget to initialize!
            if (v1 != null && !v1.isEmpty()) iters.add(v1.iterator());
            if (v2 != null && !v2.isEmpty()) iters.add(v2.iterator());
        }

        public int next() {
            int res = iters.get(index).next();
            index = (index + 1) % iters.size();
            return res;
        }

        public boolean hasNext() {
            if (iters.isEmpty())
                return false;
            if (iters.get(index).hasNext())
                return true;
            else {
                iters.remove(index);
                if (iters.isEmpty()) //this is needed
                    return false;
                index = (index + 1) % iters.size(); //% by 0 will report a / by 0 exception!
                return iters.get(index).hasNext();
            }
        }
    }

    //284
    class PeekingIterator implements Iterator<Integer> {
        Iterator<Integer> iter;
        Integer next;

        public PeekingIterator(Iterator<Integer> iterator) {
            // initialize any member here.
            iter = iterator;
            next = null;
        }

        // Returns the next element in the iteration without advancing the iterator.
        public Integer peek() {
            if (next == null)
                next = iter.next();
            return next;
        }

        // hasNext() and next() should behave the same as in the Iterator interface.
        // Override them if needed.
        @Override
        public Integer next() {
            if (next == null)
                next = iter.next();
            Integer t = next;
            next = null;
            return t;
        }

        @Override
        public boolean hasNext() {
            return next != null || iter.hasNext();
        }
    }

    //285
    public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
        //not found, right child's leftmost, last left
        if (root == null || p == null)
            return null;
        TreeNode lastLeft = null;
        while (root != null){
            if (p.val > root.val)
                root = root.right;
            else if (p.val < root.val){
                lastLeft = root;
                root = root.left;
            }
            else
                break;
        }
        if (root == null)//not found
            return null;
        if (root.right != null){
            root = root.right;
            while (root.left != null)
                root = root.left;
            return root;
        }
        return lastLeft;
    }

    //288
    public class ValidWordAbbr {
        //note the question is no OTHER words in the dict. -> either no word with the same abbrev, or exist the abbrev but just the same
        HashMap<String, String> hm;

        public ValidWordAbbr(String[] dictionary) {
            hm = new HashMap<>();
            for (String s : dictionary){
                if (s != null) {
                    String a = abbrev(s);
                    if (!hm.containsKey(a)|| hm.get(a).equals(s))
                        hm.put(a, s);
                    else
                        hm.put(a, ""); //as long as a dup i18n no matter what they cancel each other
                }
            }
        }

        private String abbrev(String s){
            if (s == null || s.length() <= 2)
                return s;
            StringBuilder sb = new StringBuilder();
            sb.append(s.charAt(0));
            sb.append(s.length() - 2);
            sb.append(s.charAt(s.length() - 1));
            return sb.toString();
        }

        public boolean isUnique(String word) {
            if (word == null)
                return false;
            String a = abbrev(word);
            return !hm.containsKey(a) || hm.get(a).equals(word);
        }
    }

    //289
    public void gameOfLife(int[][] board) {
        if (board == null || board.length == 0 || board[0].length == 0)
            return;
        int[] offR = {-1, -1, -1, 0,  0, 1, 1, 1}; //should be 8 elements!!
        int[] offC = {-1, 0,  1,-1,  1,-1, 0, 1};

        for (int i = 0; i < board.length; ++i){
            for (int j = 0; j < board[0].length; ++j){
                int live = 0;
                for (int k = 0; k < offR.length; ++k){
                    int r = i + offR[k], c = j + offC[k];
                    if (r >= 0 && r < board.length && c >= 0 && c < board[0].length){
                        if (board[r][c] == 1 || board[r][c] == 2)
                            ++live;
                    }
                }
                if (board[i][j] == 1 && (live < 2 || live > 3))
                    board[i][j] = 2;
                else if (board[i][j] == 0 && live == 3)
                    board[i][j] = 3;
            }
        }
        for (int i = 0; i < board.length; ++i) {
            for (int j = 0; j < board[0].length; ++j) {
                if (board[i][j] == 2)
                    board[i][j] = 0;
                else if (board[i][j] == 3)
                    board[i][j] = 1;
            }
        }
    }

    //295
    public class MedianFinder {
        Queue<Integer> minq = new PriorityQueue<>();
        Queue<Integer> maxq = new PriorityQueue<>(Collections.reverseOrder()); //note how max pq is initialized

        // Adds a number into the data structure.
        public void addNum(int num) {
            if (maxq.isEmpty())
                maxq.offer(num);
            else if (num > maxq.peek())
                minq.offer(num);
            else
                maxq.offer(num);

        }

        // Returns the median of current data stream
        public double findMedian() {
            while (maxq.size() > minq.size() + 1)
                minq.offer(maxq.poll());
            while (minq.size() > maxq.size() + 1)
                maxq.offer(minq.poll());
            if (maxq.size() == minq.size())
                return (maxq.peek() +minq.peek())/2.0;
            else
                return maxq.size() > minq.size() ? maxq.peek() : minq.peek();
        }
    }

    //296
    public int minTotalDistance(int[][] grid) {
        if (grid == null || grid.length == 0 || grid[0].length == 0)
            return 0;
        //the meeting point is the median point in each direction and are not orthogonal, no relational to each other
        //note it's median point, point in middle index. NOT Average point since there could be still moves
        List<Integer> rows = new ArrayList<>();
        List<Integer> cols = new ArrayList<>();
        for (int i = 0; i < grid.length; ++i){
            for (int j = 0; j < grid[0].length; ++j){
                if (grid[i][j] == 1){
                    rows.add(i);
                    cols.add(j);
                }
            }
        }
        //rows are sorted, cols not
        Collections.sort(cols);
        int res = 0;
        for (int i : rows)
            res += Math.abs(i - rows.get(rows.size()/2));
        for (int j : cols)
            res += Math.abs(j - cols.get(cols.size()/2));
        return res;
    }

    //297
    public class Codec {
        // Encodes a tree to a single string.
        public String serialize(TreeNode root) {
            if (root == null)
                return null;
            StringBuilder sb = new StringBuilder();
            serializeHelper(root, sb);

            return sb.toString();
        }

        private void serializeHelper(TreeNode root, StringBuilder sb){
            if (root == null){
                sb.append("#,");
                return;
            }
            sb.append(root.val);
            sb.append(',');
            serializeHelper(root.left, sb);
            serializeHelper(root.right, sb);
        }

        // Decodes your encoded data to tree.
        public TreeNode deserialize(String data) {
            if (data == null || data.length() == 0)
                return null;
            String[] tokens = data.split(","); //note split will return (1,2,) as 1 and 2<not care after the last deliminitor>
            List<Integer> index = new ArrayList<>();
            index.add(0);
            return deserializeHelper(tokens, index);
        }

        private TreeNode deserializeHelper(String[] tokens, List<Integer> index){
            int ind = index.get(0);

            if (ind == tokens.length)
                return null;

            String s = tokens[ind];
            index.set(0, ind+1); //must be set before the null check
            if (s.equals("#"))
                return null;

            TreeNode root = new TreeNode(Integer.parseInt(s));
            root.left = deserializeHelper(tokens, index);
            root.right = deserializeHelper(tokens, index);
            return root;
        }
    }

    //298
    private int longestconsec;
    public int longestConsecutive(TreeNode root) {
        if (root == null)
            return 0;
        this.longestconsec = 1;
        longestConsecutiveHelper(root, null, 1);
        return this.longestconsec;
    }

    private void longestConsecutiveHelper(TreeNode root, TreeNode parent, int k){
        if (root == null)
            return;
        if (root.left == null && root.right == null){
            if (parent != null)
                k = root.val == parent.val + 1 ? k + 1 : k;
            this.longestconsec = Math.max(longestconsec, k);
            return;
        }
        if (parent != null){
            if (root.val == parent.val + 1)
                ++k;
            else {
                this.longestconsec = Math.max(longestconsec, k);
                k = 1;
            }
        }
        longestConsecutiveHelper(root.left, root, k);
        longestConsecutiveHelper(root.right, root, k);
    }

    //303
    public class NumArray {
        private int[] dp; //sum of 0 - i

        public NumArray(int[] nums) {
            if (nums == null)
                return;
            dp = Arrays.copyOf(nums, nums.length); //Arrays.copyOf(T [], int new_len) note must supply new length!!
            for (int i = 1; i < dp.length; ++i)
                dp[i] += dp[i-1];
        }

        public int sumRange(int i, int j) {
            return i == 0? dp[j]: dp[j] - dp[i-1];
        }
    }

    //304
    public class NumMatrix {
        private int[][] dp;

        public NumMatrix(int[][] matrix) {
            if (matrix == null || matrix.length == 0 || matrix[0].length == 0)
                return; //2d array the inner array can be null! so has to check at all times!!!
            dp = new int[matrix.length+1][matrix[0].length+1];
            for (int i = 1; i < dp.length; ++i){
                int sum = 0;
                for (int j = 1; j < dp[0].length; ++j){
                    dp[i][j] = dp[i-1][j] + sum + matrix[i-1][j-1];
                    sum += matrix[i-1][j-1];
                }
            }
        }

        public int sumRegion(int row1, int col1, int row2, int col2) {
            return dp[row2+1][col2+1] - dp[row2+1][col1] - dp[row1][col2+1] + dp[row1][col1];
        }
    }

    //313
    public int nthSuperUglyNumber(int n, int[] primes) {
//        if (n <= 0 || primes == null || primes.length == 0)
//            return 0;
//        Queue<Integer> pq = new PriorityQueue<>();
//        pq.offer(1);
//        int res = 1;
//        while (n-- > 0){
//            res = pq.poll();
//            for (int p : primes)
//                pq.offer(res * p);
//        }
//        return res;
    }

    //314
    public List<List<Integer>> verticalOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null)
            return res;
        Queue<TreeNode> q1 = new LinkedList<>();
        Queue<Integer> q2  = new LinkedList<>();
        Map<Integer, List<Integer>> hm = new HashMap<>();
        int min = 0, max = 0;
        int cur = 1, next = 0;
        q1.offer(root);
        q2.offer(0);

        while (!q1.isEmpty()){
            TreeNode tn = q1.poll();
            int vi = q2.poll();
            if (!hm.containsKey(vi))
                hm.put(vi, new ArrayList<>());
            hm.get(vi).add(tn.val);

            if (tn.left != null){
                q1.offer(tn.left);
                q2.offer(vi - 1);
                min = Math.min(min, vi-1);
                ++next;
            }
            if (tn.right != null){
                q1.offer(tn.right);
                q2.offer(vi + 1);
                max = Math.max(max, vi+1);
                ++next;
            }

            if (--cur == 0){
                cur = next;
                next = 0;
            }
        }
        for (int i = min; i <= max; ++i)
            res.add(hm.get(i));
        return res;
    }

    //318
    public int maxProduct(String[] words) {
        //check two string has common letters - use bit array, in reality, use int instead, =>32 bits
        if (words == null || words.length == 0)
            return 0;
        //since all words are 26 small letters, one int is enough per word
        int[] masks = new int[words.length];
        for (int i = 0; i < words.length; ++i){
            for (int j = 0; j < words[i].length(); ++j){
                masks[i] |= (1 << (words[i].charAt(j) - 'a'));
            }
        }
        int max = 0;
        for (int i = 0; i < masks.length; ++i){
            for (int j = i + 1; j < masks.length; ++j){
                if ((masks[i] & masks[j]) == 0)
                    max = Math.max(max, words[i].length() * words[j].length());
            }
        }
        return max;
    }

    //329

    public int longestIncreasingPath(int[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0)
            return 0;
        int longestInc = 1;
        int[][] dp = new int[matrix.length][matrix[0].length];
        for (int i = 0; i < matrix.length; ++i){
            for (int j = 0; j < matrix[0].length; ++j){
                longestInc = Math.max(longestInc,longestIncreasingPathHelper(matrix, i, j, dp));
            }
        }
        return longestInc;
    }

    private int longestIncreasingPathHelper(int[][] matrix, int i, int j, int[][] dp){
        if (dp[i][j] > 0)
            return dp[i][j];

        int[][] off = {{-1,0}, {1, 0}, {0,-1}, {0, 1}};
        int lmax = 0;
        for (int k = 0; k < off.length; ++k) {
            int x = i + off[k][0], y = j + off[k][1];
            if (x < 0 || x >= matrix.length || y < 0 || y >= matrix[0].length || matrix[x][y] <= matrix[i][j])
                continue;
            lmax = Math.max(lmax,longestIncreasingPathHelper(matrix, x, y, dp));
        }
        dp[i][j] = lmax + 1;
        return dp[i][j];
    }

    //331
    public boolean isValidSerialization(String preorder) {
        //a non-null node will generate two new rec and recycle one. a null will not add any. but any node will consume 1.
        //note initial is 1 since we expect there is one and root is going to consume it.
        if (preorder == null || preorder.length() == 0)
            return false;
        String[] tokens = preorder.split(",");
        int total = 1;
        for (String t: tokens){
            if (--total < 0)
                return false;
            if (!t.equals("#"))
                total += 2;
        }
        return total == 0;
    }


    //341

    public interface NestedInteger{
        boolean isInteger();
        Integer getInteger();
        List<NestedInteger> getList();
    }


    public class NestedIterator implements Iterator<Integer> {
        Deque<Iterator<NestedInteger>> st;
        Iterator<NestedInteger> iter;
        NestedInteger nextNi;
        public NestedIterator(List<NestedInteger> nestedList) {
            st = new ArrayDeque<>();
            if (!nestedList.isEmpty())
                iter = nestedList.iterator();
        }

        @Override
        public Integer next() {
            int x = nextNi.getInteger();
            nextNi = null;
            return x;
        }

        @Override
        public boolean hasNext() {
            if (iter == null)
                return false;
            if (!iter.hasNext()){
                if (!st.isEmpty()) {
                    iter = st.pop();
                    return hasNext();
                }
                else
                    return false;
            }
            else {
                NestedInteger ni = iter.next();
                if (ni.isInteger()){
                    nextNi = ni;
                    return true;
                }
                else {
                    st.push(iter);
                    iter = ni.getList().iterator();
                    return hasNext();
                }
            }
        }
    }

    //344
    public String reverseString(String s) {
        if (s == null || s.length() == 0)
            return s;
        return new StringBuilder(s).reverse().toString();
    }

    //345
    public String reverseVowels(String s) {
        if (s == null || s.length() <= 1)
            return s;
        String check = "aeiouAEIOU";
        char[] sc = s.toCharArray();
        int l = 0, r = sc.length - 1;

        while (l < r){
            if (check.indexOf(sc[l]) == -1)
                ++l;
            else if (check.indexOf(sc[r]) == -1)
                --r;
            else {
                char x = sc[l];
                sc[l] = sc[r];
                sc[r] = x;
                ++l;
                --r;
            }
        }
        return new String(sc);
    }

    //346
    public class MovingAverage {
        private Queue<Integer> queue;
        private double sum;
        private int size;

        /** Initialize your data structure here. */
        public MovingAverage(int size) {
            queue = new LinkedList<Integer>(); //LinkedList only accept a collection to ctor, no size
            this.size = size; //if want to use the same name, must qualify the instance one with this!!!!! size = size NOT WORKING!
        }

        public double next(int val) {
            sum += val;
            queue.offer(val);
            if (queue.size() > size)
                sum -= queue.poll();
            return sum / queue.size();
        }
    }

    //359
    public class Logger {
        Map<String, Integer> hm;

        /** Initialize your data structure here. */
        public Logger() {
            hm = new HashMap<>();
        }

        /** Returns true if the message should be printed in the given timestamp, otherwise returns false.
         If this method returns false, the message will not be printed.
         The timestamp is in seconds granularity. */
        public boolean shouldPrintMessage(int timestamp, String message) {
            if (!hm.containsKey(message)){
                hm.put(message, timestamp);
                return true;
            }
            if (timestamp - hm.get(message) >= 10){
                hm.put(message, timestamp);
                return true;
            }
            return false;
        }
    }

    //361
    public int maxKilledEnemies(char[][] grid) {
        if (grid == null || grid.length == 0 || grid[0].length == 0)
            return 0;
        int rowenemy = 0;
        int[] colenemy = new int[grid[0].length];
        int res = 0;
        for (int i = 0; i < grid.length; ++i){
            for (int j = 0; j < grid[0].length; ++j){
                if (j == 0 || grid[i][j-1] == 'W'){
                    rowenemy = 0; //dont want to reset to 0
                    //update row enemy count till a Wall
                    for (int k = j; k < grid[0].length && grid[i][k] != 'W'; ++k)
                        rowenemy += grid[i][k] == 'E' ? 1 : 0;
                }
                if (i == 0 || grid[i-1][j] == 'W'){
                    colenemy[j] = 0;
                    for (int k = i; k < grid.length && grid[k][j] != 'W'; ++k)
                        colenemy[j] += grid[k][j] == 'E' ? 1 : 0;
                }
                if (grid[i][j] == '0')
                    res = Math.max(res, rowenemy + colenemy[j]);
            }
        }
        return res;
    }

    //362
    public class HitCounter {
        class Counter{
            int timestamp;
            int count;

            public Counter(int t, int c){
                timestamp = t;
                count = c;
            }

        }

        Queue<Counter> q;
        int hits;
        Counter c;

        /** Initialize your data structure here. */
        public HitCounter() {
            q = new LinkedList<>();
            c = new Counter(1, 0);
        }

        /** Record a hit.
         @param timestamp - The current timestamp (in seconds granularity). */
        public void hit(int timestamp) {
            if (timestamp == c.timestamp)
                ++c.count;
            else {
                q.offer(c);
                hits += c.count;
                c = new Counter(timestamp, 1);
            }
        }

        /** Return the number of hits in the past 5 minutes.
         @param timestamp - The current timestamp (in seconds granularity). */
        public int getHits(int timestamp) {
            q.offer(c); //when gethits, the last node is not commited yet so commit it first
            hits += c.count;
            c = new Counter(timestamp, 0);
            int start = timestamp - 300 +1 < 1 ? 1 : timestamp - 300+1; //last 5 min is 300- 300 +1!!

            while (!q.isEmpty() && q.peek().timestamp < start) {
                hits -= q.poll().count;
            }
            return hits;
        }
    }

    //368
    public List<Integer> largestDivisibleSubset(int[] nums) {
        List<Integer> res = new ArrayList<>();
        if (nums == null || nums.length == 0)
            return res;
        Arrays.sort(nums); //if (a, b, c) in ascending order, b % a == 0, c % b == 0 then must c % a == 0
        int[] dp = new int[nums.length];
        Arrays.fill(dp, 1);
        int max_dp = 1, max_index = 0;
        //when need to give the trace for a DP q. here is the example!!!
        int[] lastStep = new int[nums.length];
        Arrays.fill(lastStep, -1);

        for (int i = 0; i < nums.length; ++i){
            for (int j = i - 1; j >= 0; --j){
                if (nums[i] % nums[j] == 0){
                    if (dp[j] + 1 > dp[i]) {
                        dp[i] = dp[j] + 1;
                        lastStep[i] = j;
                    }

                    if (dp[i] > max_dp){
                        max_dp = dp[i];
                        max_index = i;
                    }
                    //break; //the first one on the left will give the largest # of divisible-> WRONG!!! [1,2,4,8,9,72] 9 HAS LESS THAN 8
                }
            }
        }
        //System.out.println(Arrays.toString(dp)); HOW TO PRINT AN ARRAY IN JAVA
        for (int i = max_index; i != -1; i = lastStep[i])
            res.add(nums[i]);
        return res;
    }

    //369
    public ListNode plusOne(ListNode head) {
        if (head == null)
            return head;
        ListNode cur = head, firstNon9 = null;
        while (cur != null){
            if (cur.val != 9)
                firstNon9 = cur;
            cur = cur.next;
        }
        if (firstNon9 != null) {
            ++firstNon9.val;
            while (firstNon9.next != null) {
                firstNon9.next.val = 0;
                firstNon9 = firstNon9.next;
            }
            return head;
        }
        cur = head;
        while (cur != null){
            cur.val = 0;
            cur = cur.next;
        }
        ListNode newHead = new ListNode(1);
        newHead.next = head;
        return newHead;
    }

    //371
//    public int getSum(int a, int b) {
//
//
//    }

    //379
    public class PhoneDirectory {
        BitSet bs;
        int max;

        /** Initialize your data structure here
         @param maxNumbers - The maximum numbers that can be stored in the phone directory. */
        public PhoneDirectory(int maxNumbers) {
            bs = new BitSet(maxNumbers); //note if < 64 will still create a size() = 64 BitSet
            max = maxNumbers;
        }

        /** Provide a number which is not assigned to anyone.
         @return - Return an available number. Return -1 if none is available. */
        public int get() {
            if (bs.cardinality() == max) //cardinality() return number of set bit
                return -1;
            int x = bs.nextClearBit(0);
            bs.set(x);
            return x;
        }

        /** Check if a number is available or not. */
        public boolean check(int number) {
            return !bs.get(number);
        }

        /** Recycle or release a number. */
        public void release(int number) {
            bs.clear(number);
        }
    }

    //380
    public class RandomizedSet {
        List<Integer> list;
        Map<Integer, Integer> hm;

        /** Initialize your data structure here. */
        public RandomizedSet() {
            list = new ArrayList<>();
            hm = new HashMap<>();
        }

        /** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
        public boolean insert(int val) {
            if (hm.containsKey(val))
                return false;
            list.add(val);
            hm.put(val, list.size() - 1);
            return true;
        }

        /** Removes a value from the set. Returns true if the set contained the specified element. */
        public boolean remove(int val) {
            if (!hm.containsKey(val))
                return false;
            int ind = hm.get(val);
            int last = list.get(list.size() - 1);
            list.set(ind, last);
            hm.put(last, ind);
            list.remove(list.size() - 1);
            hm.remove(val);
            list.forEach(System.out::println);
            return true;
        }

        /** Get a random element from the set. */
        public int getRandom() {
            return list.get(new Random().nextInt(list.size()));
        }
    }

    //387
    public int firstUniqChar(String s) {
        if (s == null || s.length() == 0)
            return -1;
        Map<Character, Integer> hm = new HashMap<>();
        for (int i = 0; i < s.length(); ++i){
            hm.put(s.charAt(i), hm.containsKey(s.charAt(i))? 2: 1);
        }
        for (int i = 0; i < s.length(); ++i){
            if (hm.get(s.charAt(i)) == 1)
                return i;
        }
        return -1;
    }


    //389
    public char findTheDifference(String s, String t) {
        if (s == null || t == null || s.length() +1 != t.length())
            return 0;

        char res = 0;
        for (int i = 0; i < s.length(); ++i) {
            res ^= s.charAt(i);
        }
        for (int i = 0; i < t.length(); ++i) {
            res ^= t.charAt(i);
        }

        return res;
    }

    //393
    public boolean validUtf8(int[] data) {
        if (data == null || data.length == 0)
            return true;
        int cnt = 0;
        for (int x : data){
            if (cnt == 0){
                if ((x >> 5) == 0b110)        cnt = 1;
                else if ((x >> 4) == 0b1110)  cnt = 2;
                else if ((x >> 3) == 0b11110) cnt = 3;
                else if ((x >> 7) != 0) return false;
            }
            else {
                if ((x >> 6) != 0b10) return false;
                --cnt;
            }
        }
        return cnt == 0; //note if there are missing bytes
    }
















}
