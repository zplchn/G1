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

    public static void main(String[] args){
        Solution st = new Solution();
        double x = 0;
        int v = 3;
        x += v;
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

    class Interval{
        int start, end;
        public Interval(){start = end = 0;}
        public Interval(int x, int y){
            start = x;
            end = y;
        }
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














}
