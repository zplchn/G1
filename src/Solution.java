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

    public interface NestedInteger{
        boolean isInteger();
        Integer getInteger();
        List<NestedInteger> getList();
    }

    public static void main(String[] args){
        Solution st = new Solution();
        String m1 = "a", m2 = "a*?";
        System.out.println(st.isMatch(m1, m2));
        long ts = System.currentTimeMillis();
        System.out.println(Long.numberOfLeadingZeros(ts)); //23 leading 0 in the long
        System.out.println(Long.toBinaryString(ts)); //41bits

        Map<Integer,List<Integer>> hm = new HashMap<>();

        //System.out.println(hm.get(1).get(0));
        StringBuilder sb = new StringBuilder();
        sb.append(-3);
        System.out.println(sb);







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

    //3
    public int lengthOfLongestSubstring(String s) {
        if (s == null || s.length() == 0)
            return 0;
        Map<Character, Integer> hm = new HashMap<>();
        int l = 0, max = 0;
        for (int i = 0; i < s.length(); ++i){
            if (hm.containsKey(s.charAt(i)) && hm.get(s.charAt(i)) >= l){
                max = Math.max(max, i - l);
                l = hm.get(s.charAt(i)) + 1;
            }
            hm.put(s.charAt(i), i);
        }
        max = Math.max(max, s.length() - l);
        return max;
    }

    //4
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        if (nums1 == null || nums2 == null)
            return 0;
        int len = nums1.length + nums2.length;
        if ((len & 1) == 1)
            return findMedianHelper(nums1, 0, nums1.length - 1, nums2, 0, nums2.length - 1, len/2 + 1);
        else
            return (findMedianHelper(nums1, 0, nums1.length - 1, nums2, 0, nums2.length - 1, len/2) +
                    findMedianHelper(nums1, 0, nums1.length - 1, nums2, 0, nums2.length - 1, len/2 + 1)) / 2.0;
    }

    private int findMedianHelper(int[] nums1, int ai, int aj, int[] nums2, int bi, int bj, int k){ //K IS 1- BASED
        //ending condition: 1. we cut in half, one array could be len = 0; 2. if k ==1, we return the smaller of the two heads
        int la = aj - ai + 1;
        int lb = bj - bi + 1;
        if (la > lb) //make sure a is always shorter
            return findMedianHelper(nums2, bi, bj, nums1, ai, aj, k);
        if (la == 0)
            return nums2[bi + k - 1];
        if (k == 1)
            return Math.min(nums1[ai], nums2[bi]);

        // pos of k/2
        int offA = Math.min(la, k/2);
        int offB = k - offA;
        if (nums1[ai + offA - 1] == nums2[bi + offB - 1])
            return nums1[ai + offA - 1];
        else if (nums1[ai + offA - 1] < nums2[bi + offB - 1])
            return findMedianHelper(nums1, ai + offA, aj, nums2, bi, bi + offB - 1, k - offA);
        else
            return findMedianHelper(nums1, ai, ai + offA - 1, nums2, bi + offB, bj, k - offB);
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

    //6
    public String convert(String s, int numRows) {
        if (s == null || s.length() == 0 || numRows <= 1)
            return s;
        int zigsize = 2 * numRows - 2;
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < numRows; ++i){
            for (int j = i; j < s.length(); j += zigsize){
                sb.append(s.charAt(j));
                if (i > 0 && i < numRows - 1 && j + zigsize - 2 * i < s.length())
                    sb.append(s.charAt(j + zigsize - 2 * i));
            }
        }
        return sb.toString();
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

    //8
    public int myAtoi(String str) {
        if (str == null)
            return 0;
        str = str.trim();
        if (str.length() == 0)
            return 0;
        long res = 0;
        boolean isNeg = false;
        int start = 0;
        if (str.charAt(0) == '+' || str.charAt(0) == '-') {
            isNeg = str.charAt(0) == '-';
            start = 1;
        }
        for (int i = start; i < str.length(); ++i){
            if (!Character.isDigit(str.charAt(i)))
                break;
            res = res * 10 + str.charAt(i) - '0';
            if (!isNeg && res > Integer.MAX_VALUE)
                return Integer.MAX_VALUE;
            else if (isNeg && -res < Integer.MIN_VALUE)
                return Integer.MIN_VALUE;
        }
        return (int)(isNeg?-res:res);
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

    //12
    public String intToRoman(int num) {
        if (num < 1)
            return "";
        int[]    i = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
        String[] r = {"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"};
        int k = 0;
        StringBuilder sb = new StringBuilder();
        while (num >= 1){
            if (num >= i[k]) {
                sb.append(r[k]);
                num -= i[k];
            }
            else
                ++k; //here is ++k haha
        }
        return sb.toString();
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

    //19
    public ListNode removeNthFromEnd(ListNode head, int n) {
        if (head == null || n < 1)
            return head;
        ListNode dummy = new ListNode(0), l, r;
        l = r = dummy;
        dummy.next = head;
        while (n-- > 0 && r != null)
            r = r.next;
        if (r == null)
            return head;
        while (r != null && r.next != null){
            l = l.next;
            r = r.next;
        }
        l.next = l.next.next;
        return dummy.next;
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

    //24
    public ListNode swapPairs(ListNode head) {
        if (head == null || head.next == null) //dont forget head.next == null
            return head;
        ListNode dummy = new ListNode(0), pre = dummy, cur = head;
        while (cur != null && cur.next != null){
            ListNode next = cur.next.next;
            pre.next = cur.next;
            cur.next.next = cur;
            cur.next = next;
            pre = cur;
            cur = next;
        }
        return dummy.next;
    }

    //25
    public ListNode reverseKGroup(ListNode head, int k) {
        if(head == null || head.next == null || k <= 1)
            return head;
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode left = dummy, cur = head;
        while (cur != null){
            int t = k;
            ListNode right = left;
            while (right != null && t-- > 0)
                right = right.next;
            if (t >= 0)
                break;
            ListNode pre = null;
            while (pre != right){
                ListNode next = cur.next;
                cur.next = pre;
                pre = cur;
                cur = next;
            }
            right = left.next;
            left.next.next = cur;
            left.next = pre;
            left = right;
        }
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

    //28
    public int strStr(String haystack, String needle) {
        if (haystack == null || needle == null || haystack.length() < needle.length())
            return -1;
        for (int i = 0; i <= haystack.length() - needle.length(); ++i){
            int j = 0;
            while (j < needle.length() && haystack.charAt(i + j) == needle.charAt(j))
                ++j;
            if (j == needle.length())
                return i;
        }
        return -1;
    }

    //29
//    public int divide(int dividend, int divisor) {
//        //here is the theory, any int can be expressed as a sum of power of 2. k * 2^k + (k-1) * 2 ^(k-1) + ... + k0 * 2^0. All k are either 0 or 1
//
//
//
//    }

    //30
    public List<Integer> findSubstring(String s, String[] words) {
        List<Integer> res = new ArrayList<>();
        if (s == null || s.length() == 0 || words == null || words.length == 0)
            return res;
        Map<String, Integer> hm = new HashMap<>();
        for (String w : words)
            hm.put(w, hm.getOrDefault(w, 0) +1);
        int wlen = words[0].length();
        for (int i = 0; i < wlen; ++i){
            int cnt = 0;
            Map<String, Integer> lm = new HashMap<>(hm);
            for (int j = i, l = i; j <= s.length() - wlen; j += wlen){
                String str = s.substring(j, j + wlen);
                if (lm.containsKey(str)){
                    lm.put(str, lm.get(str) - 1);
                    if (lm.get(str) >= 0)
                        ++cnt;
                    while (cnt == words.length){
                        String lstr = s.substring(l, l + wlen);
                        if (!lm.containsKey(lstr))
                            l += wlen;
                        else if (lm.get(lstr) < 0){
                            lm.put(lstr, lm.get(lstr) + 1);
                            l += wlen;
                        }
                        else {
                            if (j - l == wlen * (words.length - 1)){
                                res.add(l);
                            }
                            break;
                        }
                    }
                }
            }
        }
        return res;
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

    //32
    public int longestValidParentheses(String s) {
        if (s == null || s.length() == 0)
            return 0;
        Deque<Integer> st = new ArrayDeque<>();
        int max = 0, start = 0;
        for (int i = 0; i < s.length(); ++i){
            if (s.charAt(i) == '(')
                st.push(i);
            else if (st.isEmpty()){
                start = i + 1;
            }
            else {
                st.pop();
                max = Math.max(max, st.isEmpty()? i - start + 1: i - st.peek());
            }
        }
        return max;
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

    //38
    public String countAndSay(int n) {
        if (n < 1)
            return "";
        StringBuilder sb = new StringBuilder("1");
        while (n-- > 1){
            StringBuilder sb1 = new StringBuilder();
            int i = 1, cnt = 1;
            while (i < sb.length()){
                if (sb.charAt(i) != sb.charAt(i - 1)){
                    sb1.append(cnt);
                    sb1.append(sb.charAt(i - 1));
                    cnt = 1;
                }
                else
                    ++cnt;
                ++i;
            }
            sb1.append(cnt);
            sb1.append(sb.charAt(i - 1));
            sb = sb1;
        }
        return sb.toString();
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

    //43
    public String multiply(String num1, String num2) {
        if (num1 == null || num1.length() == 0 || num1.equals("0")
                || num2 == null || num2.length() == 0 || num2.equals("0"))
            return "0";
        StringBuilder sb1 = new StringBuilder(num1).reverse();
        StringBuilder sb2 = new StringBuilder(num2).reverse();

        int[] mul = new int[sb1.length() + sb2.length()];
        for (int i = 0; i < sb1.length(); ++i){
            for (int j = 0; j < sb2.length(); ++j){
                mul[i + j] += (sb1.charAt(i) - '0') * (sb2.charAt(j) - '0');
            }
        }
        StringBuilder sb = new StringBuilder();
        int carry = 0;
        for (int i = 0; i < mul.length; ++i){
            mul[i] += carry;
            sb.append(mul[i] % 10);
            carry = mul[i] / 10;
        }
        sb = sb.reverse();
        if (sb.charAt(0) == '0')
            return sb.substring(1);
        return sb.toString();
    }

    //44
    public boolean isMatch(String s, String p) {
        // ? - 1 char;  * 0-inf chars
        if (s == null || p == null) {
            return false;
        }
        boolean[][] dp = new boolean[s.length() + 1][p.length() + 1];
        dp[0][0] = true; //dont forget this. this is the baseline
        for (int j = 1; j < dp[0].length; ++j) {
            dp[0][j] = p.charAt(j - 1) == '*' && dp[0][j - 1];
        }
        for (int i = 1; i < dp.length; ++i) {
            for (int j = 1; j < dp[0].length; ++j) {
                if (s.charAt(i - 1) == p.charAt(j - 1) || p.charAt(j - 1) == '?') {
                    dp[i][j] = dp[i - 1][j - 1];
                } else if (p.charAt(j - 1) == '*') {//make sure dp index and string index are correctly handled
                    dp[i][j] = dp[i - 1][j] || dp[i][j - 1];
                }
            }
        }
        return dp[dp.length - 1][dp[0].length - 1];
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

    //51
    public List<List<String>> solveNQueens(int n) {
        List<List<String>> res = new ArrayList<>();
        if (n <= 0)
            return res;
        solveQueensHelper(n, 0, new int[n], res);
        return res;
    }

    private void solveQueensHelper(int n, int row, int[] colForQueens, List<List<String>> res){
        if (row == n){
            List<String> combi = new ArrayList<>();
            for (int c : colForQueens){
                char[] q = new char[n]; //note the ways to initialize a string with same char
                Arrays.fill(q, '.');
                q[c] = 'Q';
                combi.add(new String(q));
            }
            res.add(combi);
            return;
        }
        for (int j = 0; j < n; ++j){
            if (isValidQueen(colForQueens, row, j)){
                colForQueens[row] = j;
                solveQueensHelper(n, row + 1, colForQueens, res);
            }
        }
    }

    private boolean isValidQueen(int[] colForQueens, int r, int c){
        for (int i = 0; i < r; ++i){
            if (colForQueens[i] == c || Math.abs(colForQueens[i] - c) == r - i)
                return false;
        }
        return true;
    }

    //52
    public int totalNQueens(int n) {
        if (n <= 0)
            return 0;
        return totalNQueensHelper(n, 0, new int[n]);
    }

    private int totalNQueensHelper(int n, int row, int[] colForQueens){
        int cnt = 0;
        if (row == n)
            return 1;
        for (int j = 0; j < n; ++j){
            colForQueens[row] = j;
            if (isValidQueens(row, j, colForQueens))
                cnt += totalNQueensHelper(n, row + 1, colForQueens);
        }
        return cnt;
    }

    private boolean isValidQueens(int r, int c, int[] colForQueens){
        for (int i = 0; i < r; ++i){
            if (colForQueens[i] == c || r - i == Math.abs(colForQueens[i] - c))
                return false;
        }
        return true;
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

    //54
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> res = new ArrayList<>();
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0)
            return res;
        int m = matrix.length, n = matrix[0].length;
        int lvl = (Math.min(m, n) >> 1);
        for (int l = 1; l <= lvl; ++l){
            for (int j = l-1; j < n - l; ++j)
                res.add(matrix[l-1][j]);
            for (int i = l - 1; i < m - l; ++i)
                res.add(matrix[i][n - l]);
            for (int j = n - l; j >= l; --j)
                res.add(matrix[m - l][j]);
            for (int i = m - l; i >= l; --i)
                res.add(matrix[i][l-1]);
        }
        if (Math.min(m, n) % 2 == 1) {
            if (m <= n) { //note for m , n both odd, need to output the heart
                for (int j = lvl; j < n - lvl; ++j)
                    res.add(matrix[lvl][j]);
            }
            else  {
                for (int i = lvl; i < m - lvl; ++i)
                    res.add(matrix[i][lvl]);
            }
        }
        return res;
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

    //59
    public int[][] generateMatrix(int n) {
        if (n <= 0)
            return new int[0][0];
        int lvl = (n >> 1), cnt = 1;
        int[][] res = new int[n][n];
        for (int l = 1; l <= lvl; ++l){
            for (int j = l - 1; j < n - l; ++j)
                res[l-1][j] = cnt++;
            for (int i = l - 1; i < n - l; ++i)
                res[i][n-l] = cnt++;
            for (int j = n - l; j >= l; --j)
                res[n-l][j] = cnt++;
            for (int i= n - l; i >= l; --i)
                res[i][l-1] = cnt++;
        }
        if (n % 2 == 1)
            res[lvl][lvl] = cnt;
        return res;
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

    //61
    public ListNode rotateRight(ListNode head, int k) {
        if (head == null ||head.next == null || k <= 0)
            return head;
        ListNode cur = head;
        int len = 0;
        while (cur != null){
            ++len;
            cur = cur.next;
        }
        k %= len;
        ListNode right = head;
        while (k-- > 0)
            right = right.next;
        cur = head;
        while (right != null && right.next != null){
            right = right.next;
            cur = cur.next;
        }
        right.next = head;
        ListNode res = cur.next;
        cur.next = null;
        return res;
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

    //68
    public List<String> fullJustify(String[] words, int maxWidth) {
        List<String> res = new ArrayList<>();
        if (words == null || words.length == 0 || maxWidth <= 0)
            return res;
        int cnt = 0, last = 0; //cnt is current line total char, last is where we were in the words array

        for (int i = 0; i <= words.length; ++i){
            if (i != words.length && cnt + words[i].length() + (i - last) <= maxWidth)
                cnt += words[i].length(); //i - last is the bear minimum space between words needed
            else {
                //calculate space for current row
                int spaceLeft = maxWidth - cnt;
                int spaceSlot = i - last - 1;
                StringBuilder sb = new StringBuilder();

                if (i != words.length && spaceSlot > 0){ //this line has more than 1 words
                    int avgSpace = spaceLeft / spaceSlot;
                    int firstExtra = spaceLeft % spaceSlot;

                    for (int j = last; j < i; ++j){
                        sb.append(words[j]);
                        for (int a = 0; a < avgSpace; ++a)
                            sb.append(" ");
                        if (firstExtra > 0) {
                            sb.append(" ");
                            firstExtra = 0;
                        }
                    }
                }
                else { //only one word in this line or last line
                    for (int j = last; j < i; ++j) {
                        sb.append(words[j]);
                        for (int a = sb.length(); a < maxWidth; ++a)
                            sb.append(" ");
                    }
                }
                res.add(sb.toString());
                last = i;
                cnt = 0;
                --i;
            }
        }
        return res;
    }

    //69
    public int mySqrt(int x) {
        if (x <= 1)
            return x;
        int l = 1, r = x, m;
        while (l <= r){
            m = l + ((r-l) >> 1);
            if (m > x/m)
                r = m - 1;
            else if (m < x/m)
                l = m + 1;
            else
                return m;
        }
        return r;
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

    //76
    public String minWindow(String s, String t) {
        if (s == null || s.length() == 0 || t == null || t.length() ==0 || s.length() < t.length())
            return "";
//        Map<Character, Integer> hm = new HashMap<>();
        int[] hm = new int[128];
        Arrays.fill(hm, Integer.MIN_VALUE);
        for (int i = 0; i < t.length(); ++i){
            if (hm[t.charAt(i) -'0'] == Integer.MIN_VALUE)
                hm[t.charAt(i) -'0'] = 1;
            else
                ++hm[t.charAt(i) -'0'];
        }

        int cnt = 0, l = 0, min = s.length() + 1;
        String res = "";
        for (int i = 0; i < s.length(); ++i){
            char c = s.charAt(i);
            if (hm[c - '0'] != Integer.MIN_VALUE){
                --hm[c - '0'];
                if (hm[c - '0'] >= 0)
                    ++cnt;
                while (cnt == t.length()){
                    char lc = s.charAt(l);
                    if (hm[lc - '0'] == Integer.MIN_VALUE)
                        ++l;
                    else if (hm[lc - '0'] < 0){
                        ++hm[lc - '0'];
                        ++l;
                    }
                    else {
                        if (i - l + 1< min){
                            min = i -l + 1;
                            res = s.substring(l, i + 1);
                        }
                        break;
                    }
                }
            }
        }
        return res;
    }

    //77
    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> res = new ArrayList<>();
        if (n < 1 || k < 1)
            return res;
        combineHelper(n, k, 1, 0, new ArrayList<Integer>(), res);
        return res;
    }

    private void combineHelper(int n, int k, int start, int cnt, List<Integer> combi, List<List<Integer>> res){
        if (cnt == k){
            res.add(new ArrayList<>(combi));
            return;
        }
        for (int i = start; i <= n; ++i){ //i is all the way to n
            combi.add(i);
            combineHelper(n, k, i + 1, cnt + 1, combi, res);
            combi.remove(combi.size() - 1);
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

    //84
    public int largestRectangleArea(int[] heights) {
        if (heights == null || heights.length == 0)
            return 0;
        int max = 0;
        //stack storing acsending subsequence's index! may not contiguous
        Deque<Integer> st = new ArrayDeque<>();
        for (int i = 0; i <= heights.length; ++i){
            if (st.isEmpty() || (i != heights.length && heights[i] >= heights[st.peek()]))
                st.push(i);
            else {
                while (!st.isEmpty() && (i == heights.length || heights[i] < heights[st.peek()])){
                    max = Math.max(max, heights[st.pop()] * (st.isEmpty()? i : (i - st.peek() - 1))); //attention stack is index!
                }
                --i; //for loop back 1
            }
        }
        return max;
    }

    //85
    public int maximalRectangle(char[][] matrix) {
        //at each level do largest histogram
        if (matrix == null || matrix.length ==0 || matrix[0].length == 0)
            return 0;
        int[] height = new int[matrix[0].length];
        int res = 0;
        for (char[] m : matrix){
            for (int j = 0; j < m.length; ++j){
                height[j] = (m[j] == '1'? height[j] + 1: 0); //if 0 need to clear height!!!
            }
            //now check largest histogram of height
            res = Math.max(res, maximalRectangleHelper(height));
        }
        return res;
    }

    private int maximalRectangleHelper(int[] height){
        int res = 0;
        Deque<Integer> st = new ArrayDeque<>(); //store indices of acending subsequence
        for (int i = 0; i <= height.length; ++i){
            int cur = i == height.length? -1: height[i];
            while (!st.isEmpty() && height[st.peek()] > cur){//note here the trick use cur and assign - 1 when right bank
                res = Math.max(res, height[st.pop()]*(st.isEmpty()? i:(i - st.peek() - 1))); //pop first!!
            }
            st.push(i);
        }
        return res;
    }

    //86
    public ListNode partition(ListNode head, int x) {
        if (head == null)
            return head;
        ListNode less = new ListNode(0), lh = less;
        ListNode great = new ListNode(0), gh = great, cur = head;

        while (cur != null){
            if (cur.val < x){
                lh.next = cur;
                lh = lh.next;
            }
            else {
                gh.next = cur;
                gh = gh.next;
            }
            cur = cur.next;
        }
        gh.next = null;
        lh.next = great.next;
        return less.next;
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

    //91
    public int numDecodings(String s) {
        if (s == null || s.length() == 0)
            return 0;

        int[] dp = new int[2];
        dp[0] = 1; //note here 0 needs to be = 0
        int res = dp[1] = s.charAt(0) == '0'? 0 : 1;
        for (int i = 2; i <= s.length(); ++i){
            res = (s.charAt(i-1) != '0'? dp[1]: 0) + ((s.charAt(i-2) != '0' && Integer.parseInt(s.substring(i-2, i)) <= 26)? dp[0]: 0);
            dp[0] = dp[1];
            dp[1] = res;
        }
        return res;
    }

    //92
    public ListNode reverseBetween(ListNode head, int m, int n) {
        if (head == null || head.next == null || m < 1 || n < 1 || m >= n)
            return head;
        ListNode dummy = new ListNode(0), l = dummy;
        dummy.next = head;
        while (l != null && m-- > 1) {
            l = l.next;
            --n;
        }
        ListNode cur = l.next, pre = null;
        while (n-- > 0 && cur != null){
            ListNode next = cur.next;
            cur.next = pre;
            pre = cur;
            cur = next;
        }
        l.next.next = cur;
        l.next = pre;
        return dummy.next;
    }

    //93
    public List<String> restoreIpAddresses(String s) {
        List<String> res = new ArrayList<>();
        if (s == null || s.length() < 4)
            return res;
        restoreIpHelper(s, 0, 0, "", res);
        return res;
    }

    private void restoreIpHelper(String s, int i, int k, String pre, List<String> res){
        if (k == 3){
            String rest = s.substring(i);
            if (isValidIp(rest)) {
                pre += rest;
                res.add(pre);
            }
            return;
        }
        for (int j = i + 1; j <= s.length() && j <=i + 3; ++j){ //dont forget j need to be less than length()!!
            String p = s.substring(i, j);
            if (isValidIp(p))
                restoreIpHelper(s, j, k + 1, pre + p + ".", res);
        }
    }

    private boolean isValidIp(String s){
        if (s.length() == 0 || s.length() > 3 || (s.length() > 1 && s.charAt(0) == '0') || Integer.parseInt(s) > 255)
            return false;
        return true;
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

    //99
    public void recoverTree(TreeNode root) {
        if (root == null)
            return;
        recoverTreeHelper(root);
        int x = r1.val;
        r1.val = r2.val;
        r2.val = x;
    }
    private TreeNode recoverPre, r1, r2;
    private void recoverTreeHelper(TreeNode root){
        if (root == null)
            return;
        recoverTreeHelper(root.left);
        if (recoverPre != null && root.val <= recoverPre.val){
            if (r1 == null)
                r1 = recoverPre;
            r2 = root;
        }
        recoverPre = root;
        recoverTreeHelper(root.right);
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

    //105
    public TreeNode buildTree1(int[] preorder, int[] inorder) {
        if (preorder == null || inorder== null || preorder.length == 0 || preorder.length != inorder.length)
            return null;
        Map<Integer, Integer> hm = new HashMap<>();
        for (int i = 0; i < inorder.length; ++i)
            hm.put(inorder[i], i);
        return buildTreeHelper(preorder, 0, preorder.length - 1, inorder, 0, inorder.length - 1, hm);
    }

    private TreeNode buildTreeHelper(int[] preorder, int pl, int pr, int[] inorder, int il, int ir, Map<Integer, Integer> hm){
        if (il > ir)
            return null;
        TreeNode root = new TreeNode(preorder[pl]);
        int k = hm.get(preorder[pl]);
        root.left = buildTreeHelper(preorder, pl + 1, pl + k - il, inorder, il, k - 1, hm);
        root.right = buildTreeHelper(preorder, pl + k - il + 1, pr, inorder, k + 1, ir, hm);
        return root;
    }

    //106
    public TreeNode buildTree(int[] inorder, int[] postorder) {
        if (inorder == null || postorder == null || inorder.length == 0 || inorder.length != postorder.length)
            return null;
        Map<Integer, Integer> hm = new HashMap<>();
        for (int i = 0; i < inorder.length; ++i)
            hm.put(inorder[i], i);
        return buildTree2Helper(inorder, 0, inorder.length - 1, postorder, 0, postorder.length - 1, hm);
    }

    private TreeNode buildTree2Helper(int[] inorder, int il, int ir, int[] postorder, int pl, int pr, Map<Integer, Integer> hm){
        if (il > ir)
            return null;
        TreeNode root = new TreeNode(postorder[pr]);
        int k = hm.get(root.val);
        root.left = buildTree2Helper(inorder, il, k - 1, postorder, pl, pl + k - il - 1, hm);
        root.right = buildTree2Helper(inorder, k + 1, ir, postorder, pl + k - il, pr - 1, hm);
        return root;
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

    //115
    public int numDistinct(String s, String t) {
        if (s == null || t == null)
            return 0;
        int[][] dp = new int[t.length()+1][s.length()+1];
        Arrays.fill(dp[0], 1);
        for (int i = 1; i < dp.length; ++i){
            for (int j = 1; j < dp[0].length; ++j){
                dp[i][j] = dp[i][j-1];
                if (t.charAt(i-1) == s.charAt(j-1))
                    dp[i][j] += dp[i-1][j-1];
            }
        }
        return dp[dp.length - 1][dp[0].length - 1];
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

    //119
    public List<Integer> getRow(int rowIndex) {
        List<Integer> res = new ArrayList<>();
        if (rowIndex < 0) //0 is [1]
            return res;
        res.add(1);

        while (rowIndex-- > 0){
            int tmp = 0, pre = 1;
            res.add(0);
            for (int i = 1; i < res.size(); ++i){
                tmp = res.get(i);
                res.set(i, res.get(i) + pre);
                pre = tmp;
            }
        }
        return res;
    }

    //120
    public int minimumTotal(List<List<Integer>> tri) {
        if (tri == null || tri.size() == 0 || tri.get(0).size() == 0)
            return 0;
        for (int i = tri.size() - 2; i >= 0; --i){
            for (int j = 0; j < tri.get(i).size(); ++j){
                tri.get(i).set(j, tri.get(i).get(j) + Math.min(tri.get(i+1).get(j), tri.get(i+1).get(j+1)));
            }
        }
        return tri.get(0).get(0);
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

    //123
    public int maxProfit3(int[] prices) {
        if (prices == null || prices.length < 2)
            return 0;
        int[] dp = new int[prices.length];
        int lmin = prices[0], lmax = prices[prices.length - 1], res = 0;
        for (int i = 1; i < prices.length; ++i){
            dp[i] = Math.max(dp[i-1], prices[i] - lmin);
            lmin = Math.min(lmin, prices[i]);
        }
        res = dp[dp.length - 1]; //maybe should just buy once!
        for (int i = prices.length - 2; i >=0; --i){
            res = Math.max(res, dp[i] + Math.max(0, lmax - prices[i]));
            lmax = Math.max(lmax, prices[i]);
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

    //126
//    public List<List<String>> findLadders(String beginWord, String endWord, Set<String> wordList) {
//        List<List<String>> res = new ArrayList<>();
//        if (beginWord == null || endWord == null || beginWord.length() != endWord.length() || wordList == null || wordList.size() == 0)
//            return res;
//
//
//    }

    //127
    public int ladderLength(String beginWord, String endWord, Set<String> wordList) {
        //set up graph on wordList, bfs from begin to endword
        if (beginWord == null || endWord == null || wordList == null || wordList.size() == 0 || beginWord.length() != endWord.length())
            return 0;
        Queue<String> queue = new LinkedList<>();
        Set<String> visited = new HashSet<>();
        queue.offer(beginWord);
        visited.add(beginWord);
        int cur = 1, next = 0, len = 1;
        while (!queue.isEmpty()){
            String s = queue.poll();
            for (int i = 0; i < s.length(); ++i){
                char[] ca = s.toCharArray(); // this needs reset every time
                for (char c = 'a'; c <= 'z'; ++c){
                    ca[i] = c;
                    String ss = new String(ca);
                    if (ss.equals(endWord))
                        return len + 1;
                    if (wordList.contains(ss) && !visited.contains(ss)){
                        visited.add(ss);
                        queue.offer(ss);
                        ++next;
                    }
                }
            }
            if (--cur == 0){
                cur = next;
                next = 0;
                ++len;
            }
        }
        return 0; //not found should return 0

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

    //133
    class UndirectedGraphNode{
        int label;
        List<UndirectedGraphNode> neighbors;

        UndirectedGraphNode(int x){
            label = x;
            neighbors = new ArrayList<>();
        }
    }
    public UndirectedGraphNode cloneGraph(UndirectedGraphNode node) {
        if (node == null)
            return node;
        Map<UndirectedGraphNode, UndirectedGraphNode> hm = new HashMap<>();
        UndirectedGraphNode newNode = new UndirectedGraphNode(node.label);
        hm.put(node, newNode);
        cloneGraphHelper(node, newNode, hm);
        return newNode;
    }

    private void cloneGraphHelper(UndirectedGraphNode node, UndirectedGraphNode newNode, Map<UndirectedGraphNode, UndirectedGraphNode> hm){
        for (UndirectedGraphNode un : node.neighbors){
            if (!hm.containsKey(un)){
                hm.put(un, new UndirectedGraphNode(un.label));
                newNode.neighbors.add(hm.get(un));
                cloneGraphHelper(un, hm.get(un), hm);
            }
            else
                newNode.neighbors.add(hm.get(un));
        }
    }

    //134
    public int canCompleteCircuit(int[] gas, int[] cost) {
        if (gas == null || gas.length == 0 || cost == null || cost.length != gas.length)
            return 0;
        int start = 0, total = 0, local = 0;
        for (int i = 0; i < gas.length; ++i){
            int add = gas[i] - cost[i];
            local += add;
            if (local< 0){
                local = 0;
                start = i + 1;
            }
            total += add;
        }
        return total >= 0? start: -1;
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
    public int singleNumber1(int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;
        int res = 0;
        for (int i : nums)
            res ^= i;
        return res;
    }

    //137
    public int singleNumber(int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;
        int res = 0;
        for (int i = 0; i < 32; ++i){
            int cnt = 0;
            for (int j : nums){
                cnt += ((j >>> i) & 1);
            }
            res |= ((cnt % 3) << i);
        }
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
    public boolean wordBreak1(String s, Set<String> wordDict) {
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

    //140
    public List<String> wordBreak(String s, Set<String> wordDict) {
        List<String> res = new ArrayList<>();
        if (s == null || wordDict == null)
            return res;
        wordBreakHelper(s, 0, wordDict, "", res);
        return res;
    }

    private void wordBreakHelper(String s, int i, Set<String> wordDict, String pre, List<String> res){
        if (i == s.length()){
            res.add(pre.substring(1));
            return;
        }
        for (int k = i+1; k <= s.length(); ++k){
            String sub = s.substring(i, k);
            if (wordDict.contains(sub))
                wordBreakHelper(s, k, wordDict, pre + " " + sub, res);
        }
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

    //143
    public void reorderList(ListNode head) {
        if (head == null || head.next == null)
            return;
        ListNode slow, fast;
        slow = fast = head;
        while (fast != null && fast.next != null){
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode pre = null;
        while (slow != null){
            ListNode next = slow.next;
            slow.next = pre;
            pre = slow;
            slow = next;
        }
        fast = head;
        while (fast != null && pre != null){
            ListNode nf = fast.next;
            fast.next = pre;
            ListNode np = pre.next;
            pre.next = nf;
            fast = nf;
            pre = np;
        }
        if (fast != null)
            fast.next = null; //there is a cse when even the last node points to itself
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

    //145
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null)
            return res;
        Deque<TreeNode> st = new ArrayDeque<>();
        TreeNode pre = null;
        while (!st.isEmpty() || root != null){
            if (root != null){
                st.push(root);
                root = root.left;
            }
            else {
                if (pre != st.peek().right) { //always keep track of last looked node
                    root = st.peek().right;
                    pre = root;
                }
                else {
                    pre = st.pop();
                    res.add(pre.val);
                }
            }
        }
        return res;
    }

    //146
    class LRUCache {
        //doubly linked list(key,value) + hashmap(key-> listnode)

        class ListNode{
            int key, value;
            ListNode prev, next;
            ListNode(int k, int v){
                key = k;
                value = v;
                prev = next = null;
            }
        }

        private int size;
        private int capacity;
        private ListNode head, tail;
        private Map<Integer, ListNode> hm;

        public LRUCache(int capacity) {
            this.capacity = capacity;
            this.head = new ListNode(-1, -1);
            this.tail = new ListNode(-1, -1);
            head.next = tail;
            tail.prev = head;
            hm = new HashMap<>();
        }

        private void moveToHead(ListNode ln){
            if (head.next == ln)
                return;
            ln.prev.next = ln.next;
            ln.next.prev = ln.prev;
            attachToHead(ln);
        }

        private void attachToHead(ListNode ln){
            ln.next = head.next;
            ln.next.prev = ln;
            head.next = ln;
            ln.prev = head;
        }

        private void removeTail(){
            if (tail.prev == head)
                return;
            hm.remove(tail.prev.key);
            tail.prev.prev.next = tail;
            tail.prev = tail.prev.prev;
            --size;
        }

        public int get(int key) {
            if (!hm.containsKey(key))
                return -1;
            ListNode ln = hm.get(key);
            moveToHead(ln);
            return ln.value;
        }

        public void set(int key, int value) {
            if (hm.containsKey(key)){
                ListNode ln = hm.get(key);
                ln.value = value;
                moveToHead(ln);
            }
            else {
                if (this.size == this.capacity){
                    removeTail();
                }
                ListNode ln = new ListNode(key, value);
                hm.put(key, ln);
                attachToHead(ln);
                ++size;
            }
        }
    }

    //147
    public ListNode insertionSortList(ListNode head) {
        if (head == null || head.next == null)
            return head;
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode cur = head, pre;

        while (cur != null && cur.next != null){
            if (cur.val <= cur.next.val)
                cur = cur.next;
            else {
                pre = dummy;
                while (pre.next.val <= cur.next.val)
                    pre = pre.next;
                ListNode x = cur.next;
                cur.next = cur.next.next;
                x.next = pre.next;
                pre.next = x;
            }
        }
        return dummy.next;
    }

    //148
    public ListNode sortList(ListNode head) {
        //like postorder
        if (head == null || head.next == null)
            return head;
        ListNode slow, fast;
        slow = fast = head;
        //we must cut off into two halves!!!!!!
        while (fast != null && fast.next != null && fast.next.next != null){
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode rh = slow.next;
        slow.next = null;

        return mergeList(sortList(head), sortList(rh));
    }

    private ListNode mergeList(ListNode la, ListNode lb){
        if (la == null)
            return lb;
        if (lb == null)
            return la;
        ListNode dummy = new ListNode(0), pre = dummy;
        while (la != null && lb != null){
            if (la.val < lb.val){
                pre.next = la;
                la = la.next;
            }
            else {
                pre.next = lb;
                lb = lb.next;
            }
            pre = pre.next;
        }
        pre.next = la != null? la: lb;
        return dummy.next;
    }

    //149
    class Point{
        int x, y;
        Point(){x = 0; y = 0;}
        Point(int a, int b){x = a; y = b;}
    }
    public int maxPoints(Point[] points) {
        if (points == null || points.length == 0)
            return 0;
        int max = 1;
        for (int i = 0; i < points.length - 1; ++i){
            int same = 0;
            Map<Double, Integer> hm = new HashMap<>();
            for (int j = i + 1; j < points.length; ++j){
                if (points[i].x == points[j].x && points[i].y == points[j].y)
                    ++same;
                else if (points[i].x == points[j].x)
                    hm.put(Double.MAX_VALUE, hm.containsKey(Double.MAX_VALUE)? hm.get(Double.MAX_VALUE) + 1: 2);
                else if (points[i].y == points[j].y)
                    hm.put(0.0, hm.containsKey(0.0)? hm.get(0.0) + 1: 2);
                else {
                    double slope = (double)(points[j].y - points[i].y)/(points[j].x - points[i].x);
                    hm.put(slope, hm.containsKey(slope)? hm.get(slope) + 1: 2);
                }
            }
            int lmax = 1;
            for (int x : hm.values())
                lmax = Math.max(lmax, x);
            max = Math.max(max, lmax + same);
        }
        return max;
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

    //156
    public TreeNode upsideDownBinaryTree(TreeNode root) {
        if (root == null)
            return null;
        return upsideDownHelper(root, null);
    }

    private TreeNode upsideDownHelper(TreeNode root, TreeNode parent){
        if (root == null)
            return parent;
        TreeNode newRoot = upsideDownHelper(root.left, root);
        if (parent != null){
            root.left = parent.right;
            root.right = parent;
        }
        else {
            root.left = root.right = null;
        }
        return newRoot;
    }

    //157
    int read4(char[] buf){return 0;}
    public int read1(char[] buf, int n) {
        if (buf == null || buf.length == 0 || n <= 0)
            return 0;
        char[] buffer = new char[4];
        int i = 0;
        boolean eof = false;
        while (!eof && n > 0){
            int x = Math.min(n, read4(buffer)); //toread or left, one less than 4 then we r done
            n -= x;
            System.arraycopy(buffer, 0, buf, i, x);
            i += x;
            if (x < 4)
                eof = true;
        }
        return i;
    }

    //158
    private char[] buf4 = new char[4];
    private int buf4Idx = 0, buf4Left = 0;
    public int read(char[] buf, int n) {
        if (buf == null || buf.length == 0 || n < 0)
            return 0;
        int i = 0;
        boolean eof = false;
        while (!eof && n > 0){
            if (buf4Left == 0){
                buf4Left = read4(buf4);
                if (buf4Left < 4)
                    eof = true;
            }
            int x = Math.min(n, buf4Left);
            n -= x;
            System.arraycopy(buf4, buf4Idx, buf, i, x);
            i += x;
            buf4Left -= x;
            buf4Idx = (buf4Idx + x) % 4;
        }
        return i;
    }

    //159
    public int lengthOfLongestSubstringTwoDistinct(String s) {
        if (s == null || s.length() == 0)
            return 0;
        Map<Character, Integer> hm = new HashMap<>();
        int l = 0, r = 0, max = 0;
        while (r < s.length()){
            hm.put(s.charAt(r), hm.getOrDefault(s.charAt(r), 0) + 1);
            while (hm.size() > 2){
                hm.put(s.charAt(l), hm.get(s.charAt(l)) - 1);
                if (hm.get(s.charAt(l)) == 0)
                    hm.remove(s.charAt(l));
                ++l;
            }
            max = Math.max(max, r - l + 1);
            ++r;
        }
        return max;
    }

    //160
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null)
            return null;
        int n1 = 0, n2 = 0;
        ListNode ha = headA, hb = headB;
        while (ha.next != null){
            ++n1;
            ha = ha.next;
        }
        while (hb.next != null){
            ++n2;
            hb = hb.next;
        }
        if (ha != hb)
            return null; //not intersect

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

    //168
    public String convertToTitle(int n) {
        if (n < 1)
            return "";
        StringBuilder sb = new StringBuilder();
        while (n != 0){
            n -= 1; //every round need to -1 first to adjust to 0-based
            sb.append((char)(n % 26 + 'A')); //when int + char, must cast to char
            n /= 26;
        }
        return sb.reverse().toString();
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

    //172
    public int trailingZeroes(int n) {
        if (n <=0)
            return 0;
        int res = 0;
        while (n != 0){
            int x = n / 5;
            res += x;
            n = x;
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

    //174
    public int calculateMinimumHP(int[][] dungeon) {
        if (dungeon == null || dungeon.length == 0 || dungeon[0].length == 0)
            return 0;
        for (int i = dungeon.length - 1; i >= 0; --i){
            for (int j = dungeon[0].length - 1; j >= 0; --j){
                if (i == dungeon.length - 1 && j == dungeon[0].length - 1)
                    continue;
                else if (i == dungeon.length - 1)
                    dungeon[i][j] += Math.min(dungeon[i][j+1], 0);
                else if (j == dungeon[0].length - 1)
                    dungeon[i][j] += Math.min(dungeon[i+1][j], 0);
                else {
                    dungeon[i][j] += Math.max(Math.min(dungeon[i][j+1], 0), Math.min(dungeon[i+1][j], 0));
                }
            }
        }
        return -Math.min(dungeon[0][0], 0) +1; // must be at least 1 when leaving start point
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

    //187
    public List<String> findRepeatedDnaSequences(String s) {
        List<String> res = new ArrayList<>();
        if (s == null || s.length() <= 10)
            return res;
        Set<String> hs = new HashSet<>();
        Set<String> res2 = new HashSet<>();

        for (int j = 0; j+10 <= s.length(); ++j){
            String str = s.substring(j, j + 10);
            if (hs.contains(str))
                res2.add(str);
            else
                hs.add(str);
        }

        res.addAll(res2);
        return res;
    }

    //189
    public void rotate(int[] nums, int k) {
        if (nums == null || nums.length == 0 || k <= 0)
            return;
        k %= nums.length;
        rotateHelper(nums, 0, nums.length - 1);
        rotateHelper(nums, 0, k - 1);
        rotateHelper(nums, k, nums.length - 1);
    }

    private void rotateHelper(int[] nums, int i, int j){
        while (i < j){
            int t = nums[i];
            nums[i] = nums[j];
            nums[j] = t;
        }
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

    //191
    public int hammingWeight(int n) {
        int res = 0;
        while (n != 0){
            n = n & (n - 1); //clear least-significant 1. all 1 left to this 1 stays. all 0 right to this flip to 1
            //so n & (n -1) remove the least-significant 1
            ++res;
        }
        return res;
    }

    //198
    public int rob(int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;
        int[] dp = new int[nums.length + 1];
        dp[1] = nums[0];
        for (int i = 1; i < nums.length; ++i)
            dp[i+1] = Math.max(dp[i], nums[i] + dp[i-1]);
        return dp[dp.length - 1];
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

    //201
    public int rangeBitwiseAnd(int m, int n) {
    //low bit diff, high bit same
        int i = 0;
        while (m != n){
            m >>= 1;
            n >>= 1;
            ++i;
        }
        return m << i;
    }

    //202
    public boolean isHappy(int n) {
        if (n <= 0)
            return false;
        Set<Integer> hs = new HashSet<>();
        while (!hs.contains(n)){
            hs.add(n);
            int sum = 0;
            while (n != 0){
                sum += (n % 10) * (n % 10);
                n /= 10;
            }
            if (sum == 1)
                return true;
            n = sum;
        }
        return false;
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

    //204
    public int countPrimes(int n) {
        if (n < 2)
            return 0;
        boolean[] mark = new boolean[n+1];
        for (int i = 2; i * i < n; ++i){
            if (!mark[i]){
                for (int j = i; i * j < n; ++j)
                    mark[i * j] = true;
            }
        }
        int cnt = 0;
        for (int i = 2; i < n; ++i)
            if (!mark[i])
                ++cnt;
        return cnt;
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

    //207
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        //find cycle in a directed graph -> topological sort and see if every node is traversed.
        if (numCourses == 0 || prerequisites.length == 0 || prerequisites[0].length == 0)
            return true;
        //first we need to recreate the graph, a indegree counter array, a children list array
        int[] indegree = new int[numCourses];
        List<Integer>[] children = new List[numCourses]; //java does not support creating generic array, the right side use plain List!!!
        //for (List<Integer> l: children)
        //    l = new ArrayList<>(); FOREACH CAN LET U ASSIGN NEW VALUE, BUT DOESNT ACTUALLY ASSIGN AND NOTHING WILL CHANGE
        for (int i = 0; i < children.length; ++i)
            children[i] = new ArrayList<>();
        for (int i = 0; i < prerequisites.length; ++i){
            ++indegree[prerequisites[i][0]];
            children[prerequisites[i][1]].add(prerequisites[i][0]);
        }
        Queue<Integer> q = new LinkedList<>();
        int cnt = 0;
        for (int i = 0; i < indegree.length; ++i){
            if (indegree[i] == 0)
                q.offer(i);

        }
        while (!q.isEmpty()){
            int x = q.poll();
            ++cnt;
            for (int c : children[x]){
                if (--indegree[c] == 0)
                    q.offer(c);
            }
        }
        return cnt == numCourses;
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

    //210
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        if (numCourses <= 0 || prerequisites == null)
            return new int[0];

        int[] indegree = new int[numCourses];
        List<Integer>[] children = new List[numCourses];
        for (int i = 0; i < children.length; ++i){
            children[i] = new ArrayList<>();
        }
        for (int i = 0; i < prerequisites.length; ++i){
            ++indegree[prerequisites[i][0]];
            children[prerequisites[i][1]].add(prerequisites[i][0]);
        }

        Queue<Integer> q = new LinkedList<>();
        for (int i = 0; i < indegree.length; ++i){
            if (indegree[i] == 0)
                q.offer(i);
        }
        List<Integer> res = new ArrayList<>();
        while(!q.isEmpty()){
            int x = q.poll();
            res.add(x);
            for (int i : children[x]){
                if (--indegree[i] == 0)
                    q.offer(i);
            }
        }
        if (res.size() < numCourses)
            return new int[0];
        int[] r = new int[res.size()];

        for (int i = 0; i < res.size(); ++i)
            r[i] = res.get(i);
        return r;
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

    //213
    public int robCircle(int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;
        if (nums.length == 1)
            return nums[0];
        int max = 0;
        int[] dp = new int[nums.length];
        dp[1] = nums[0]; //here must take care of when nums = [1], so rule it out before this
        for (int i = 1; i < nums.length - 1; ++i)
            dp[i+1] = Math.max(dp[i], dp[i-1] + nums[i]);
        max = dp[dp.length - 1];
        Arrays.fill(dp, 0);
        dp[1] = nums[1];
        for (int i = 2; i < nums.length; ++i)
            dp[i] = Math.max(dp[i-1], dp[i-2] + nums[i]);
        return Math.max(max, dp[dp.length - 1]);
    }

    //215
    public int findKthLargest(int[] nums, int k) {
        if (nums == null || nums.length == 0 || k < 1)
            return -1;
        Queue<Integer> pq = new PriorityQueue<>();
        for (int x : nums){
            if (pq.size() < k)
                pq.offer(x);
            else if (x > pq.peek()){
                pq.poll();
                pq.offer(x);
            }
        }
        return pq.poll();
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

    //220
    public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
        if (nums == null || nums.length < 2 || k < 1)
            return false;
        //need to keep an window of size k and find nearby element within t, so storing the window in a treeset
        TreeSet<Integer> ts = new TreeSet<>();
        for (int i = 0; i < nums.length; ++i){
            Integer floor = ts.floor(nums[i]);
            Integer ceiling = ts.ceiling(nums[i]);
            if ((floor != null && nums[i] - floor <= t) || (ceiling != null && ceiling - nums[i] <= t))
                return true;
            if (ts.size() == k + 1)
                ts.remove(nums[i - k]);
        }
        return false;
    }

    //221
    public int maximalSquare(char[][] matrix) { //note it's a char array
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0)
            return 0;
        //dp is (i, j) being the bottom right cornor,dp[i][j] = Math.min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1,  note minimum is 0, min square size is 1;
        int[][] dp = new int[matrix.length + 1][matrix[0].length + 1];
        int res = 0;
        for (int i = 0; i < matrix.length; ++i){
            for (int j = 0; j < matrix[0].length; ++j){
                dp[i+1][j+1] = (matrix[i][j] == '1'? Math.min(dp[i][j], Math.min(dp[i][j+1], dp[i+1][j])) + 1: 0);//note not add matrix[i][j], if others are 1, itself is 0
                res = Math.max(res, dp[i+1][j+1]);
            }
        }
        return res*res;//return area
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

    //223
    public int computeArea(int A, int B, int C, int D, int E, int F, int G, int H) {
        int size1 = (D - B) * (C - A);
        int size2 = (H - F) * (G - E);
        if (E > C || G < A || F > D || H < B)
            return size1 + size2;
        return size1 + size2 - (Math.min(D, H) - Math.max(F, B))*(Math.min(C, G) - Math.max(A,E));
    }

    //224
    public int calculate(String s) {
        if (s == null || s.length() == 0)
            return 0;
        Deque<Integer> st = new ArrayDeque<>();
        st.push(1); //the first one used for +( left parenthesis for reference
        st.push(1); //the second one for first num, every num need to pop one op and start calculation immediately
        int res = 0; //we ignore overflow problem
        for (int i = 0; i < s.length(); ++i){
            char c = s.charAt(i);
            if (Character.isDigit(c)){
                int num = 0;
                while (i < s.length() && Character.isDigit(s.charAt(i))){
                    num = num * 10 + s.charAt(i++) - '0';
                }
                --i; //i could be at an op
                res += st.pop() * num;
            }
            else if (c == ')')
                st.pop();
            else {
                st.push(st.peek() * (c == '-'? -1 : 1)); //note for -(2+ 3) when meet (, also need to push because the first num
                                                        //here the 2 need to pop op and thus need to have its own op, the - need stay
            }
        }
        return res;
    }

    //225
    class MyStack {
        Queue<Integer> q1 = new LinkedList<>();
        Queue<Integer> q2 = new LinkedList<>();

        // Push element x onto stack.
        public void push(int x) {
            q1.offer(x);
        }

        // Removes the element on top of the stack.
        public void pop() {
            while (q1.size() != 1)
                q2.offer(q1.poll());
            q1.poll();
            Queue<Integer> t = q1;
            q1 = q2;
            q2 = t;
        }

        // Get the top element.
        public int top() {
            while (q1.size() != 1)
                q2.offer(q1.poll());
            int x = q1.poll();
            q2.offer(x);
            Queue<Integer> t = q1;
            q1 = q2;
            q2 = t;
            return x;
        }

        // Return whether the stack is empty.
        public boolean empty() {
            return q1.isEmpty();
        }
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

    //227
    public int calculate2(String s) {
        if (s == null)
            return 0;
        s = s.replaceAll("\\s+", ""); //string.replaceAll(str, str) \\s+ is all space!!!
        if (s.length() == 0)
            return 0;
        s = s + "+"; //need to clear the last operand
        int res = 0, last = 0, cur = 0;
        char ops ='+';
        for (int i = 0; i < s.length(); ++i){
            if (Character.isDigit(s.charAt(i))) {
                cur = cur * 10 +  s.charAt(i) - '0';
            }
            else {
                int tlast = 0;
                switch(ops){
                    case '+':
                        res += cur;
                        tlast = cur;
                        break;
                    case '-':
                        res -= cur;
                        tlast = -cur;
                        break;
                    case '*':
                        res = res - last + last * cur;
                        tlast = last * cur;
                        break;
                    case '/':
                        res = res - last + last / cur; //t != 0
                        tlast = last / cur;
                        break;
                    default:
                        break;
                }
                last = tlast;
                cur = 0;
                ops = s.charAt(i);
            }
        }
        return res;
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

    //229
    public List<Integer> majorityElement2(int[] nums) {
        List<Integer> res = new ArrayList<>();
        if (nums == null || nums.length == 0)
            return res;
        int m1 = -1, n1 = 0, m2 = -1, n2 = 0;
        for (int x: nums){
            if (x == m1) //even if the first one is -1 this is still correct
                ++n1;
            else if (x == m2)
                ++n2;
            else if (n1 == 0) {
                m1 = x;
                n1 = 1;
            }
            else if (n2 == 0) {
                m2 = x;
                n2 = 1;
            }

            else {
                --n1;
                --n2;
            }
        }
        n1 = n2 = 0;
        for (int x: nums){
            if (x == m1)
                ++n1;
            if (x == m2)
                ++n2;
        }
        if (n1 > nums.length / 3)
            res.add(m1);
        if (m1 != m2 && n2 > nums.length / 3)
            res.add(m2);
        return res;
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

    //232
    class MyQueue {
        Deque<Integer> s1 = new ArrayDeque<>();
        Deque<Integer> s2 = new ArrayDeque<>();

        // Push element x to the back of queue.
        public void push(int x) {
            s1.push(x);
        }

        // Removes the element from in front of queue.
        public void pop() {
            if (!s2.isEmpty())
                s2.pop();
            else {
                while (s1.size() != 1)
                    s2.push(s1.pop());
                s1.pop();
            }
        }

        // Get the front element.
        public int peek() {
            if (!s2.isEmpty())
                return s2.peek();
            else {
                while (!s1.isEmpty())
                    s2.push(s1.pop());
                return s2.peek();
            }
        }

        // Return whether the queue is empty.
        public boolean empty() {
            return s1.isEmpty() && s2.isEmpty();
        }
    }

    //234
    public boolean isPalindrome(ListNode head) {
        if (head == null || head.next == null)
            return true;
        ListNode s, f;
        s = f = head;
        while (f != null && f.next != null){
            s = s.next;
            f = f.next.next;
        }
        ListNode pre = null;
        while (s != null){
            ListNode next = s.next;
            s.next = pre;
            pre = s;
            s = next;
        }
        f = head;
        while (f != null && pre != null){
            if (f.val != pre.val)
                return false;
            f = f.next;
            pre = pre.next;
        }
        return true;
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

    //238
    public int[] productExceptSelf(int[] nums) {
        if (nums == null || nums.length <= 1)
            return nums;
        int[] res = new int[nums.length];
        res[0] = 1;
        for (int i = 1; i < res.length; ++i){
            res[i] = res[i-1] * nums[i-1];
        }
        int right = nums[res.length - 1];
        for (int i = res.length - 2; i >= 0; --i){
            res[i] *= right;
            right *= nums[i];
        }
        return res;
    }

    //239
    public int[] maxSlidingWindow(int[] nums, int k) {
        //use deque and store descending seq in the window. add every time the last new one, and decide if output then shrink the left
        if (nums == null || k <= 0 || nums.length == 0)
            return nums;
        Deque<Integer> dq = new ArrayDeque<>();
        int[] res = new int[nums.length - k + 1];

        for (int r = 0, j = 0; r < nums.length; ++r){
            while (!dq.isEmpty() && nums[r] > dq.peekLast())
                dq.pollLast();
            dq.offer(nums[r]);
            if (r >= k - 1){
                res[j++] = dq.peekFirst();
                if (dq.peekFirst() == nums[r - k + 1]) //when move, left is shrinked status
                    dq.pollFirst();
            }
        }
        return res;
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

    //243
    public int shortestDistance(String[] words, String word1, String word2) {
        if (words == null || word1 == null || word2 == null)
            return -1;
        int i1 = -1, i2 = -1, min = words.length;
        for (int i = 0; i < words.length; ++i){
            if (words[i].equals(word1)){
                if (i2 != -1)
                    min = Math.min(min, i - i2);
                i1 = i;
            }
            else if (words[i].equals(word2)){
                if (i1 != -1)
                    min = Math.min(min, i - i1);
                i2 = i;
            }
        }
        return min;
    }

    //244
    public class WordDistance {
        private Map<String, List<Integer>> hm;

        public WordDistance(String[] words) {
            if (words == null || words.length == 0)
                return;
            hm = new HashMap<>(); //dont forget to create this !!!
            for (int i = 0; i < words.length; ++i){
                if (words[i] == null) continue; //null is possible
                if (!hm.containsKey(words[i]))
                    hm.put(words[i], new ArrayList<Integer>());
                hm.get(words[i]).add(i);
            }
        }

        public int shortest(String word1, String word2) {
            if (word1 == null || word2 == null || hm == null || !hm.containsKey(word1) || !hm.containsKey(word2))
                return -1;
            List<Integer> l1 = hm.get(word1);
            List<Integer> l2 = hm.get(word2);
            int i1 = 0, i2 = 0, min = Integer.MAX_VALUE;
            while (i1 < l1.size() && i2 < l2.size()){
                min = Math.min(min, Math.abs(l1.get(i1) - l2.get(i2)));
                if (l1.get(i1) < l2.get(i2))
                    ++i1;
                else
                    ++i2;
            }
            return min;
        }
    }

    //245
    public int shortestWordDistance(String[] words, String word1, String word2) {
        if (words == null || words.length == 0 || word1 == null || word2 == null)
            return -1;
        int i1 = -1, i2 = -1, min = words.length;
        boolean isSame = word1.equals(word2);
        for (int i = 0; i < words.length; ++i){
            if (words[i].equals(word1)){
                if (isSame && i1 != -1)
                    min = Math.min(min, i - i1);
                else if (!isSame && i2 != -1)
                    min = Math.min(min, i - i2);
                i1 = i;
            }
            else if (words[i].equals(word2)){
                if (i1 != -1)
                    min = Math.min(min, i - i1);
                i2 = i;
            }
        }
        return min;
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

    //250
    private int uni;
    public int countUnivalSubtrees(TreeNode root) {
        if (root == null)
            return 0;
        countUniHelper(root);
        return this.uni;
    }

    private boolean countUniHelper(TreeNode root){
        if (root == null)
            return true;
        //if (countUniHelper(root.left) && countUniHelper(root.right) && (root.left == null || root.left.val == root.val) && (root.right == null || root.right.val == root.val)){
        //note why the commentted is wrong!!! it's because the short circuiting. when left is not uni, it wont enter the right check!!!
        //seperate postorder two subtree check at all times!

        boolean l = countUniHelper(root.left);
        boolean r = countUniHelper(root.right);
        if ( l&& r && (root.left == null || root.left.val == root.val) && (root.right == null || root.right.val == root.val)){
            this.uni++;
            return true;
        }
        return false;
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

    //254
    public List<List<Integer>> getFactors(int n) {
        List<List<Integer>> res = new ArrayList<>();
        if (n <= 1)
            return res;
        getFactorsHelper(n, 2, new ArrayList<>(), res);
        return res;
    }

    private void getFactorsHelper(int n, int start, List<Integer> combi, List<List<Integer>> res){
        if (n == 1){
            if (combi.size() > 1) //when just n itself we filter it out
                res.add(new ArrayList<>(combi));
            return;
        }
        for (int i = start; i <= (int)Math.sqrt(n); ++i){ //first acending factors only need to supply up to Math.sqrt(n) (double)
            if (n % i== 0){
                combi.add(i);
                getFactorsHelper(n / i, i, combi, res);
                combi.remove(combi.size() - 1);
            }
        }
        combi.add(n); //then like 16 = 2 * 8. the last n = 8 needs to be inserted into the combi
        getFactorsHelper(1, n + 1, combi, res);
        combi.remove(combi.size() - 1);
    }

    //256
    public int minCost(int[][] costs) {
        if (costs == null || costs.length == 0 || costs[0].length == 0)
            return 0;
        for (int i = 1; i < costs.length; ++i){
            for (int j = 0; j < costs[0].length; ++j){
                costs[i][j] += Math.min(costs[i-1][(j+1)%costs[0].length], costs[i-1][(j+2)%costs[0].length]);
            }
        }
        int res = costs[costs.length - 1][0]; //res initialized to max!!
        for (int x : costs[costs.length - 1])
            res = Math.min(res, x);
        return res;
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

    //258
    public int addDigits(int num) {
        if (num <= 0)
            return num;
        /*this called treeroot and 1 - 1 , 9 - 9, 10 - 1, 18 - 9, 19 -1 so the treeroot rotate every 9*/
        return (num - 1) % 9 + 1; // dest rotate every 9 so need to mod 9. src is 1-based, need to - 1. target is 1-based, so add 1 last
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

    //260
    public int[] singleNumber3(int[] nums) {
        int[] res = {0, 0};
        if (nums == null || nums.length < 2)
            return res;
        int xor = 0;
        for (int i : nums)
            xor ^= i;
        //xor is a ^ b, find any bit = 1 means they diff and and everyone    a & -a = rightmost 1 mask
        xor &= -xor;
        for (int i : nums){
            if ((i & xor) == 0)
                res[0] ^= i;
            else
                res[1] ^= i;
        }
        return res;
    }

    //261
    public boolean validTree(int n, int[][] edges) {
        //connected, exist topological order no cycle BFS
        if (n <= 0 || edges == null || edges.length != n - 1)
            return false;
        if (n == 1 && edges.length == 0)
            return true; //sigle point need to handle seperately
        int cnt = 0;
        List<Integer>[] children = new List[n];
        for (int i = 0; i < children.length; ++i)
            children[i] = new ArrayList<>();
        int[] degree = new int[n];
        for (int[] edge : edges){
            ++degree[edge[0]];
            ++degree[edge[1]];
            children[edge[0]].add(edge[1]);
            children[edge[1]].add(edge[0]);
        }
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < degree.length; ++i){
            if (degree[i] == 1)
                queue.offer(i);
        }
        while (!queue.isEmpty()){
            int x = queue.poll();
            ++cnt;
            for (int c : children[x]){
                if (--degree[c] == 1)
                    queue.offer(c);
            }
        }
        return cnt == n;
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

    //265
    public int minCostII(int[][] costs) {
        if (costs == null || costs.length == 0 || costs[0].length == 0)
            return 0;
        int min1 = 0, min2 = 0;
        for (int i = 0; i < costs.length; ++i){
            int tmin1 = Integer.MAX_VALUE, tmin2 = Integer.MAX_VALUE;
            for (int j = 0; j < costs[0].length; ++j){
                if (i == 0 || min1 == costs[i-1][j])
                    costs[i][j] += min2;
                else
                    costs[i][j] += min1;
                if (costs[i][j] < tmin1){
                    tmin2 = tmin1;
                    tmin1 = costs[i][j];
                }
                else if (costs[i][j] < tmin2)
                    tmin2 = costs[i][j];
            }
            min1 = tmin1;
            min2 = tmin2;
        }
        return min1;
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

    //271
    public class Codec {

        // Encodes a list of strings to a single string.
        public String encode(List<String> strs) {
            if (strs == null || strs.size() == 0)
                return "";
            StringBuilder sb = new StringBuilder();
            for (String s : strs){
                sb.append(s.length());
                sb.append("#");
                sb.append(s);
            }
            return sb.toString();
        }

        // Decodes a single string to a list of strings.
        public List<String> decode(String s) {
            List<String> res = new ArrayList<>();
            if (s == null || s.length() == 0)
                return res;
            int index = 0;
            while (index < s.length()){
                int t = s.indexOf('#', index); //indexOf(int ch/String s, [int start])
                if (t == -1)
                    return res;
                int len = Integer.parseInt(s.substring(index, t));
                res.add(s.substring(t + 1, t + len + 1));
                index = t + len + 1;
            }

            return res;
        }
    }

    //272
    public List<Integer> closestKValues(TreeNode root, double target, int k) {
        List<Integer> res = new ArrayList<>();
        if (root == null || k <= 0)
            return res;
        Queue<Integer> queue = new LinkedList<>();
        closestKValuesHelper(root, queue, k, target);
        res.addAll(queue);
        return res;
    }
    private boolean foundClosestK;
    private void closestKValuesHelper(TreeNode root, Queue<Integer> queue, int k, double target){
        if (root == null)
            return;
        if (!foundClosestK)
            closestKValuesHelper(root.left, queue, k, target);
        if (queue.size() < k)
            queue.offer(root.val);
        else if (Math.abs(target - queue.peek()) > Math.abs(root.val - target)){ //abs calculate the distance
            queue.poll();
            queue.offer(root.val);
        }
        else
            foundClosestK = true;
        if (!foundClosestK)
            closestKValuesHelper(root.right, queue, k, target);
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

    //274
    public int hIndex(int[] citations) {
        if (citations == null || citations.length == 0)
            return 0;
        Arrays.sort(citations);
        //                   1
        //               1
        //<---------------------- x: greater than myself y:absolute citations, now find the crossing when y drops below x
        //           1
        //         1
        //from right to left, x axis - total#greater than me, y - citations[i], y start >> x -> y=x -> y < x
        int i = citations.length -1;
        while (i >= 0 && citations[i] >= (citations.length - i))
            --i;
        return citations.length - i - 1;
    }

    //275
    public int hIndex2(int[] citations) {
        //this time citations is already sorted
        if (citations == null || citations.length == 0)
            return 0;
        int l = 0, r = citations.length - 1,m;
        while (l <= r){
            m = l + ((r - l) >> 1);
            if (citations[m] < citations.length - m)
                l = m + 1;
            else
                r = m - 1;
        }
        return citations.length - l;
    }

    //276
    public int numWays(int n, int k) {
        if (n < 1 || k < 1)
            return 0;
        // dp[i] = dp[i-2]*(k-1) + dp[i-1]*(k-1) //first one means at i we are the same as i-1, so i-2* k; second one means i we are differnet with i-1, so still k-1
        int[] dp = {k, k*k};
        if (n <= 2)
            return dp[n-1];
        for (int i = 2; i < n; ++i){
            int x = dp[1] * (k-1) + dp[0]* (k-1);
            dp[0] = dp[1];
            dp[1] = x;
        }
        return dp[dp.length - 1];
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

    //280
    public void wiggleSort(int[] nums) {
        if (nums == null || nums.length <= 1)
            return;
        for (int i = 0; i < nums.length -1; ++i){
            if ((i % 2 == 0 && nums[i] > nums[i+1]) ||
                    (i % 2 == 1 && nums[i] < nums[i+1])){
                int t = nums[i];
                nums[i] = nums[i + 1];
                nums[i + 1] = t;
            }
        }
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

    //282
    public List<String> addOperators(String num, int target) {
        List<String> res = new ArrayList<>();
        if (num == null || num.length() == 0)
            return res;
        addOperatorsHelper(num, target, 0, 0, 0, "", res);
        return res;
    }

    private void addOperatorsHelper(String num, int target, int start, long pre, long last, String combi, List<String> res){
        if (start == num.length()){
            if (pre == target)
                res.add(combi);
            return;
        }
        for (int i = start + 1; i <= num.length(); ++i){
            String s = num.substring(start, i);
            if (s.length() > 1 && s.charAt(0) == '0') //and note we dont take 05!!
                continue;
            long si = Long.parseLong(s); //note here it can overflow INF + 2 and like so need to treat as long!
            if (combi.length() == 0)//first one
                addOperatorsHelper(num, target, i, si, si, s, res);
            else {
                addOperatorsHelper(num, target, i, pre + si, si, combi + "+" + s, res);
                addOperatorsHelper(num, target, i, pre - si, -si, combi + "-" + s, res);
                addOperatorsHelper(num, target, i, pre - last + si * last, last * si, combi + "*" + s, res);
            }
        }
    }

    //283
    public void moveZeroes(int[] nums) {
        if (nums == null || nums.length == 0)
            return;
        int l = 0, r = 0;
        while (r < nums.length){
            if (nums[r] != 0) {
                nums[l++] = nums[r];
                nums[r] = 0;
            }
            ++r;
        }
        while (l < nums.length)
            nums[l++] = 0;
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

    //286
    public void wallsAndGates(int[][] rooms) {
        if (rooms == null || rooms.length == 0 || rooms[0].length == 0)
            return;
        final int[][] offset = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        int cur = 0, next = 0, lvl = 0;
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < rooms.length; ++i) {
            for (int j = 0; j < rooms[0].length; ++j) {
                if (rooms[i][j] == 0) {
                    int c = i * rooms[0].length + j;
                    queue.offer(c); //the trick here is here, we dont start from every 0 and traverse the """"entire table""""
                                    //instead, we list all node at dist = 0, 1,2 level all at once, the INF closer to one spot wont be further shorten by another
                    ++cur;
                }
            }
        }

        while (!queue.isEmpty()) {
            int t = queue.poll();
            int x = t / rooms[0].length;
            int y = t % rooms[0].length;

            for (int k = 0; k < offset.length; ++k) {
                int nx = x + offset[k][0];
                int ny = y + offset[k][1];
                int nt = nx * rooms[0].length + ny;
                if (nx >= 0 && nx < rooms.length && ny >= 0 && ny < rooms[0].length && rooms[nx][ny] == Integer.MAX_VALUE) {
                    rooms[nx][ny] = lvl + 1; //only update for the INF ones, nearer will be garanteed to be update earlier!!!
                    queue.offer(nt);
                    ++next;
                }
            }

            if (--cur == 0) {
                cur = next;
                next = 0;
                lvl += 1;
            }
        }
    }

    //287
    public int findDuplicate(int[] nums) {
        if (nums == null || nums.length == 0)
            return -1;
        //let's say 1,2,3 in a len=4 array, so there will be a dup for sure. next, there will be a cycle follow i->num[i]->num[num[i]]
        //because all number is <= 3 back to the array. the dup one will be where the cycle start since there are more than 1 point to it
        int slow, fast;
        slow = fast = 0;
        while (true){
            slow = nums[slow];
            fast = nums[nums[fast]];
            if (slow == fast)
                break;
        }
        slow = 0;
        while (slow != fast){
            slow = nums[slow];
            fast = nums[fast];
        }
        return slow;
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

    //290
    public boolean wordPattern(String pattern, String str) {
        if (pattern == null || str == null)
            return false;
        String[] tokens = str.split("\\s+");
        if (pattern.length() != tokens.length)
            return false;
        Map<Character, String> hm = new HashMap<>();
        Set<String> hs = new HashSet<>();
        for (int i = 0; i < pattern.length(); ++i){
            if (hm.containsKey(pattern.charAt(i))) {
                if (!hm.get(pattern.charAt(i)).equals(tokens[i]))
                    return false;
            }
            else if (hs.contains(tokens[i]))
                return false;
            else {
                hm.put(pattern.charAt(i), tokens[i]);
                hs.add(tokens[i]);
            }
        }
        return true;
    }

    //291
    public boolean wordPatternMatch(String pattern, String str) {
        if (pattern == null || str == null)
            return false;
        if (pattern.length() == 0)
            return str.length() == 0;
        if (str.length() == 0)
            return false;
        return wordPatternHelper(pattern, str, 0, 0, new HashMap<Character, String>());
    }

    private boolean wordPatternHelper(String pattern, String str, int p, int s, Map<Character, String> hm){
        if (p == pattern.length())
            return s == str.length();
        if (s == str.length())
            return false;
        if (hm.containsKey(pattern.charAt(p))){
            if (str.startsWith(hm.get(pattern.charAt(p)), s))
                return wordPatternHelper(pattern, str, p + 1, s + hm.get(pattern.charAt(p)).length(), hm);
            return false; //if not equal need return immediately
        }
        for (int i = s + 1; i <= str.length(); ++i){
            String t = str.substring(s, i);
            if (hm.containsValue(t)) //dont forget containsValue's case
                continue;
            hm.put(pattern.charAt(p), t);
            if (wordPatternHelper(pattern, str, p + 1, i, hm))
                return true;
            hm.remove(pattern.charAt(p));
        }
        return false;
    }

    //292
    public boolean canWinNim(int n) {
        return n % 4 != 0; // when 4 must fail. so 5,6,7 can remove 1-3to make competitor a 4, competitor must fail, we win;
        //8 no matter how many we remove, competitor become 5,6,7 they win. so every mod 4 we lose
    }

    //293
    public List<String> generatePossibleNextMoves(String s) {
        List<String> res = new ArrayList<>();
        if (s == null || s.length() < 2)
            return res;
        int i = 0;
        while ((i = s.indexOf("++",i)) >= 0){
            StringBuilder sb = new StringBuilder(s);
            sb.replace(i, i+2, "--");
            res.add(sb.toString());
            ++i;
        }
        return res;
    }

    //294
    public boolean canWin(String s) {
        if (s == null || s.length() < 2)
            return false;
        int i = 0;
        while ((i = s.indexOf("++", i)) >= 0){
            if (!canWin(s.substring(0, i) + "-" + s.substring(i+2))) //note we can just replace with single "-" instead of "--" for even more
                return true;
            ++i;
        }

        return false;
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
    public class Codec2 {
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

    //299
    public String getHint(String secret, String guess) {
        if (secret == null || secret.length() == 0 || guess == null || guess.length() != secret.length())
            return "0A0B";
        Map<Character, Integer> hm = new HashMap<>();
        int A = 0, B = 0;
        //"1122" -》 "1222" need only put in the map for misplaced number otherwise the first 2 will reduce matching 2
        for (int i = 0; i < secret.length(); ++i){
            if (secret.charAt(i) != guess.charAt(i))
                hm.put(secret.charAt(i), hm.containsKey(secret.charAt(i))? hm.get(secret.charAt(i)) + 1: 1);
            else
                ++A;
        }

        for (int i = 0; i < guess.length(); ++i){
            if (secret.charAt(i) != guess.charAt(i) && hm.containsKey(guess.charAt(i)) && hm.get(guess.charAt(i)) > 0){
                hm.put(guess.charAt(i), hm.get(guess.charAt(i)) - 1);
                ++B;
            }
        }
        return A + "A" + B + "B";
    }

    //300
    public int lengthOfLIS(int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;
        //we maintain a list which any elem is the last elem of a bare minimum LIS, meaning they are the minimum possble tail of any length
        //and we update based on binary-search to find the first tail
        List<Integer> tails = new ArrayList<>();
        tails.add(nums[0]);
        for (int i = 1; i < nums.length; ++i){
            int l = 0, r = tails.size() - 1, m;
            while (l <= r){
                m = l + ((r - l) >> 1);
                if (tails.get(m) < nums[i])
                    l = m + 1;
                else
                    r = m - 1;
            }
            if (l == tails.size())
                tails.add(nums[i]);
            else
                tails.set(l, nums[i]);
        }
        return tails.size();
    }

    //303
    public class NumArray1 {
        private int[] dp; //sum of 0 - i

        public NumArray1(int[] nums) {
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

    //307
    public class NumArray {
        int[] tree;//binary index tree, tree[] array 1-based storing sum of "lowbit" number of previous elements, ressulting consecutive intervals
        int[] arr;//look up time is o(logn)

        public NumArray(int[] nums) {
            if (nums == null)
                return;
            tree = new int[nums.length + 1];
            arr = new int[nums.length + 1];
            for (int i = 0; i < nums.length; ++i)
                update(i, nums[i]);
        }

        void update(int i, int val) { //update is get delta and popup index + lowbit(index) till array size
            int delta = val - arr[i+1];
            for (int k = i+1; k < arr.length; k += (k & -k)){
                tree[k] += delta;
            }
            arr[i+1] = val;
        }

        private int getSum(int i){ //get sum is start from index and keep minus lowbit(index) till index == 1 (1-based)
            int res = 0;
            for (int k = i + 1; k >=0; k -= (k&-k)){
                res += tree[k];
            }
            return res;
        }

        public int sumRange(int i, int j) {
            return getSum(j) - getSum(i-1);
        }
    }

    //310
    public List<Integer> findMinHeightTrees(int n, int[][] edges) {
        //the minimum height trees will be in the middle one or two nodes along the diameter path of the undirected graph.
        //there will be either 1 or 2 MHT that is the middle 1 or 2 nodes in the middle of longest path
        List<Integer> res = new ArrayList<>();
        if (n <= 0 || edges == null || edges.length != n - 1) //may be single node and no edge
            return res;
        if (n == 1){
            res.add(0);
            return res; // cannot calculate indegree = 1
        }

        //build graph using indegree and children
        int[] indegree = new int[n];
        List<Integer>[] neighbours = new List[n];

        for (int i = 0; i < neighbours.length; ++i)
            neighbours[i] = new ArrayList<>();

        for (int i = 0; i < edges.length; ++i){
            ++indegree[edges[i][0]];
            ++indegree[edges[i][1]];
            neighbours[edges[i][0]].add(edges[i][1]);
            neighbours[edges[i][1]].add(edges[i][0]);
        }

        //peel the leaves and stop when left <= 2
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < indegree.length; ++i){
            if (indegree[i] == 1)//leaf
                queue.offer(i);
        }
        int left = n;
        while (left > 2){
            int size = queue.size();
            left -= size; //all leaves cannot be the new root
            while (size-- > 0) {
                int x = queue.poll();
                for (int k : neighbours[x]) {
                    if (--indegree[k] == 1)
                        queue.offer(k);
                }
            }
        }
        res.addAll(queue);
        return res;
    }

    //313
    //public int nthSuperUglyNumber(int n, int[] primes) {
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
    //}

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

    //317
    public int shortestDistance(int[][] grid) {
        if (grid == null || grid.length == 0 || grid[0].length == 0)
            return 0;
        //bfs start off every house, need every empty spot reachable, so maintain one total dist array, one count array
        int[][] count = new int[grid.length][grid[0].length];
        int[][] dist = new int[grid.length][grid[0].length];
        int[][] offset = {{-1, 0}, {1, 0}, {0,-1}, {0,1}};
        int totalHouse = 0;
        for (int i = 0; i < grid.length; ++i){
            for (int j = 0; j < grid[0].length; ++j){
                if (grid[i][j] == 1){
                    ++totalHouse;

                    int lvl = 1, cur =1, next = 0;
                    Queue<int[]> queue = new LinkedList<>();
                    queue.offer(new int[]{i, j});
                    boolean[][] visited = new boolean[grid.length][grid[0].length];
                    while (!queue.isEmpty()){
                        int[] p = queue.poll();

                        //mark first, incluing set the level than put in queue
                        for (int k = 0; k < offset.length; ++k){
                            int x = p[0] + offset[k][0], y = p[1] + offset[k][1];

                            if (x >= 0 && x < grid.length && y >= 0 && y < grid[0].length && !visited[x][y] && grid[x][y] == 0){
                                visited[x][y] = true;
                                count[x][y] += 1;
                                dist[x][y] += lvl;
                                queue.offer(new int[]{x, y});
                                ++next;
                            }
                        }
                        if(--cur == 0){ //bfs must track layer to be count per layer
                            cur = next;
                            next = 0;
                            ++lvl;
                        }
                    }
                }
            }
        }

        //find min sum dist
        int min = Integer.MAX_VALUE;
        for (int i = 0; i < grid.length; ++i){
            for (int j = 0; j < grid[0].length; ++j){
                if (grid[i][j] == 0 && count[i][j] == totalHouse){
                    min = Math.min(min, dist[i][j]);
                }
            }
        }
        return min == Integer.MAX_VALUE? -1: min;
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

    //319
    public int bulbSwitch(int n) {
        if (n < 0)return 0;
        return (int)Math.sqrt(n); //a bulb only got change states at the factor number , 8 (1 on, 2 off, 4 on ,8 off) only perfect square
    }

    //322
    public int coinChange(int[] coins, int amount) {
        //not greedy say we want 6 - {4,3,1} 4 should not be selected
        if (coins == null || coins.length == 0 || amount < 0)
            return 0; //if negative, then could be unlimited way of giving coins. a neg + pos
        Arrays.sort(coins);
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, amount + 1);
        dp[0] = 0;
        for (int i = 1; i < dp.length; ++i){
            for (int j = 0; j < coins.length && coins[j] <= i; ++j){
                if (dp[i - coins[j]] <= amount)
                    dp[i] = Math.min(dp[i], dp[i - coins[j]] + 1);
            }
        }
        return dp[dp.length -1] > amount? -1: dp[dp.length - 1];
    }

    //323
    public int countComponents(int n, int[][] edges) {
        if (n <= 0 || edges == null || edges.length == 0 || edges[0].length != 2)
            return n;
        //construct the graph
        List<Integer>[] children = new List[n];
        for (int[] edge : edges){
            if (children[edge[0]] == null)
                children[edge[0]] = new ArrayList<Integer>();
            if (children[edge[1]] == null)
                children[edge[1]] = new ArrayList<Integer>();
            children[edge[0]].add(edge[1]);
            children[edge[1]].add(edge[0]);
        }

        //Mask the visited
        BitSet visited = new BitSet(n);
        int cnt = 0;

        for (int i = 0; i < n; ++i){
            if (!visited.get(i)){
                countComponentsHelper(children, i, visited);
                ++cnt;
            }
        }
        return cnt;
    }

    private void countComponentsHelper(List<Integer>[] children, int i, BitSet visited){
        visited.set(i);
        if (children[i] != null){
            for (int j = 0; j < children[i].size(); ++j){
                if (!visited.get(children[i].get(j)))
                    countComponentsHelper(children, children[i].get(j), visited);
            }
        }
    }

    //325
    public int maxSubArrayLen(int[] nums, int k) {
        //use hashmap to store prefix sum: meaning [sum from 0 to cur i, i] and move right and check if sum - k exist, then it's between after that spot and cur
        //prefix + k = sum(0 to cur)
        //both neg and pos numbers cannot use two pointers.
        if (nums == null || nums.length == 0)
            return 0;
        int max = 0, sum = 0;
        Map<Integer, Integer> hm = new HashMap<>(); //prefix sum
        hm.put(0, -1); //so that we dont update prefix = 0 by some middle point, sum - k will be every elem from index = 0 to cur
        for (int i = 0; i < nums.length; ++i){
            sum += nums[i];
            if (hm.containsKey(sum - k))
                max = Math.max(max, i - hm.get(sum - k));
            if (!hm.containsKey(sum))
                hm.put(sum, i); //we dont update prefix sum cuz we want longest
        }
        return max;
    }

    //326
    public boolean isPowerOfThree(int n) {
        //a trick solution will be get the maximum of power of 3 X, X % n == 0 will be power of three
        if (n <= 0)
            return false;
        while (n % 3 == 0){
            n /= 3;
        }
        return n == 1;
    }

    //328
    public ListNode oddEvenList(ListNode head) {
        if (head == null || head.next == null)
            return head;
        ListNode dummy = new ListNode(0);
        ListNode pre = dummy, cur = head, last = head;

        while (cur != null && cur.next != null){
            pre.next = cur.next;
            pre = pre.next;
            cur.next = cur.next.next;
            last = cur;
            cur = cur.next;
        }
        pre.next = null;
        last = cur != null ? cur : last; //it can not have the last odd or does
        last.next = dummy.next;
        return head;
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

    //332
    public List<String> findItinerary(String[][] tickets) {
        List<String> res = new ArrayList<>();
        if (tickets == null || tickets.length == 0 ||tickets[0].length == 0)
            return res;
        //this question is essentially find euler path in a directed graph.
        /*
        a directed graph each node has an indegree and outdegree, and will have 0 or 2 nodes with indegree != outdegree
        such node will be a src when out>in and a sink when in > out

                        JFK
                      /
                    SFO ---------> SJC
                  /     \
                 LAX ->   SAN


        HERE, JFK AND SJC ARE the odd nodes, and a path when traverse every single EDGE once and only once is a Euler path
        Note, path like SJC are call a bridge, meaning one way deadend, such route will be chosen at the end, we always go others first

        The algorithm to find a euler path is called fluery's algorithm
        for each node, we loop it's neighbours
            while has neighbours
                take a neighbour and delete it from my neighbours
                    continue with this new neighbours
            when no available neighbours, we inserted to the head the path

        So the trick is, insert into head, it's okay we cross bridge first!!!
        So as long as we come to SJC and SJC has no available neighbours, we inserted to the head of path, and it will always be
        the last one, so it's all set.
        then we go the circle and second time come to SFO, it has no neighbours so needs to be inserted before SJC and then backwards
        to SAN and LAX, etc.
         */

        //first we create the graph, since it needs minimum lex order, we need the neighbours sorted
        Map<String, Queue<String>> hm = new HashMap<>();
        for (String[] pair: tickets){
            hm.putIfAbsent(pair[0], new PriorityQueue<>());//if not exist then put the (key, value)
            hm.get(pair[0]).offer(pair[1]);
        }
        findItineraryHelper("JFK", res, hm);
        Collections.reverse(res);
        return res;
    }

    private void findItineraryHelper(String dept, List<String> res, Map<String, Queue<String>> hm){
        while (hm.containsKey(dept) && !hm.get(dept).isEmpty()){
            String child = hm.get(dept).poll();
            findItineraryHelper(child, res, hm);
        }
        res.add(dept); //sink will always be the first
    }

    //333
    //we need each node return a range(min, max) denoting the subtree of it and a flag, size
    class NodeRtn{
        int min = Integer.MAX_VALUE; //note here is the inverse, work perfect for null node left always small right always large
        int max = Integer.MIN_VALUE;
        boolean isBST;
        int size;
    }
    private int largestBST;
    public int largestBSTSubtree(TreeNode root) { //sub tree must contain all descendants
        if (root == null)
            return 0;
        largestBSTSubtreeHelper(root);
        return largestBST;
    }

    private NodeRtn largestBSTSubtreeHelper(TreeNode root){
        NodeRtn rtn = new NodeRtn();

        if (root == null){
            rtn.isBST = true;
            return rtn;
        }

        NodeRtn lrtn = largestBSTSubtreeHelper(root.left);
        NodeRtn rrtn = largestBSTSubtreeHelper(root.right);

        if (lrtn.isBST && rrtn.isBST && root.val > lrtn.max && root.val < rrtn.min){
            rtn.min = Math.min(lrtn.min, root.val); //left could be null
            rtn.max = Math.max(rrtn.max, root.val);
            rtn.isBST = true;
            rtn.size = lrtn.size + rrtn.size + 1;
            this.largestBST = Math.max(this.largestBST, rtn.size);
        }
        return rtn;
    }

    //334
    public boolean increasingTriplet(int[] nums) {
        if (nums == null || nums.length < 3)
            return false;
        int min1 = nums[0], min2 = Integer.MAX_VALUE;
        for (int i = 1; i < nums.length; ++i){
            if (nums[i] < min1)
                min1 = nums[i];
            else if (nums[i] < min2)
                min2 = nums[i];
            else
                return true;
        }
        return false;
    }

    //337
    public int rob(TreeNode root) {
        if (root == null)
            return 0;
        int[] res = robHelper(root);
        return Math.max(res[0], res[1]);
    }

    private int[] robHelper(TreeNode root){
        int[] dp = new int[2];
        if (root == null)
            return dp;
        int[] ldp = robHelper(root.left);
        int[] rdp = robHelper(root.right);
        //dp[not use root, max of (max(left), max(Right)) OR use root, root+ no use root left + no use root right)
        dp[0] = Math.max(ldp[0], ldp[1]) + Math.max(rdp[0], rdp[1]);
        dp[1] = root.val + ldp[0] + rdp[0];
        return dp;
    }

    //338
    public int[] countBits(int num) {
        if (num < 0)
            return new int[0];
        int[] res = new int[num + 1];
        int l = 0;
        for (int i = 1; i < res.length; ++i){ //start with 2^0 = 1 repeat when every 2^n
            if ((i & (i-1)) == 0)
                l = 0;
            res[i] = res[l++] + 1;
        }
        return res;
    }

    //339
    public int depthSum(List<NestedInteger> nestedList) {
        if (nestedList == null || nestedList.size() == 0)
            return 0;
        return depthSumHelper(nestedList, 1);
    }

    private int depthSumHelper(List<NestedInteger> list, int depth){
        int res = 0;
        for (NestedInteger ni : list){
            if (ni.isInteger())
                res += depth * ni.getInteger();
            else
                res += depthSumHelper(ni.getList(), depth + 1);
        }
        return res;
    }

    //340
    public int lengthOfLongestSubstringKDistinct(String s, int k) {
        if (s == null || s.length() == 0)
            return 0;
        Map<Character, Integer> hm = new HashMap<>();
        int l = 0, r = 0, max = 0;

        while (r < s.length()){
            //put, if larger, shrink left
            char rc = s.charAt(r);
            hm.put(rc, hm.getOrDefault(rc, 0) + 1);
            if (hm.size() > k){
                max = Math.max(max, r - l);
                while (hm.size() > k){
                    char lc = s.charAt(l);
                    hm.put(lc, hm.get(lc) - 1);
                    if (hm.get(lc) == 0)
                        hm.remove(lc);
                    ++l;
                }
            }
            ++r;
        }
        max = Math.max(max, r - l);
        return max;
    }

    //341
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

    //342
    public boolean isPowerOfFour(int num) {
        //num > 0 , num&(num-1) == 0 is power of 2. for 4, ...01010101 we need one of the odd bit has a one. so &0x55555555
        return (num > 0 && (num & (num - 1)) == 0 && (num & 0x55555555) == num);
    }

    //343
    public int integerBreak(int n) {
        //when n > 4. always maximum product is have as many 3 as possible until the remainder is <= 4, when 4 is 2*2 or 1*4 is 4 > 3*1
        if (n <= 3)
            return n-1; //less than 3 special
        int res = 1;
        while (n > 4){
            res *= 3;
            n -= 3;
        }
        return res * n;
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

    //347
    public List<Integer> topKFrequent(int[] nums, int k) {
        List<Integer> res = new ArrayList<>();
        if (nums == null || nums.length == 0 || k < 1)
            return res;
        //first stat, then use minheap o(nlogk)
        Map<Integer, Integer> hm = new HashMap<>();
        for (int x : nums){
            hm.put(x, hm.getOrDefault(x, 0) + 1);
        }
        //bucket sort
        List<Integer>[] bucket = new List[nums.length + 1]; //note right side cannot give generics
        for (Map.Entry<Integer, Integer> e : hm.entrySet()){
            if (bucket[e.getValue()] == null)
                bucket[e.getValue()] = new ArrayList<>();
            bucket[e.getValue()].add(e.getKey());
        }
        for (int i = bucket.length - 1; i >= 0; --i){
            if (bucket[i] != null) {
                for (int x : bucket[i]) {
                    res.add(x);
                    if (res.size() == k)
                        break;
                }
            }
        }
        return res;
/*
        class Tuple{
            int first;
            int second;
            Tuple(){first = second = 0;}
            Tuple(int x, int y){first = x; second = y;}
        }

        Queue<Tuple> pq = new PriorityQueue<>((t1, t2)->t1.second - t2.second);

        for(Map.Entry<Integer, Integer> e: hm.entrySet()){
            if (pq.size() < k)
                pq.offer(new Tuple(e.getKey(), e.getValue()));
            else if (e.getValue() > pq.peek().second){
                pq.poll();
                pq.offer(new Tuple(e.getKey(), e.getValue()));
            }
        }
        for (Tuple t : pq)
            res.add(t.first);


        return res;
        */
    }

    //348
    public class TicTacToe {
        //This q just need to return if there is a win/lose, not to use gaming theory to act as computer
        //counters for rows/cols/diags/anti_diags
        private int[] rows, cols;
        private int diags, anti_diags, size;
        /** Initialize your data structure here. */
        public TicTacToe(int n) {
            rows = new int[n];
            cols = new int[n];
            size = n;
        }

        /** Player {player} makes a move at ({row}, {col}).
         @param row The row of the board.
         @param col The column of the board.
         @param player The player, can be either 1 or 2.
         @return The current winning condition, can be either:
         0: No one wins.
         1: Player 1 wins.
         2: Player 2 wins. */
        public int move(int row, int col, int player) {
            player = (player == 1? player: -1);
            rows[row] += player;
            cols[col] += player;
            diags += (row == col? player: 0);
            anti_diags += (row + col == size-1? player: 0); //note anti_diags is x + y == N - 1
            //immediately know the change make a win or lose
            if (Math.abs(rows[row]) == size ||
                    Math.abs(cols[col]) == size ||
                    Math.abs(diags) == size ||
                    Math.abs(anti_diags) == size)
                return player == 1? 1: 2;
            return 0;
        }
    }

    //349
    public int[] intersection(int[] nums1, int[] nums2) {
        if (nums1 == null || nums1.length == 0 || nums2 == null || nums2.length == 0)
            return new int[0];
        if (nums1.length < nums2.length)
            return intersection(nums2, nums1);
        Set<Integer> hs = new HashSet<>();
        for (int x : nums2)
            hs.add(x);
        Set<Integer> hs2 = new HashSet<>();
        for (int x : nums1){
            if (hs.contains(x))
                hs2.add(x);
        }
        int[] res = new int[hs2.size()];
        int i = 0;
//        Iterator<Integer> iter = hs2.iterator();
//        while (iter.hasNext())
//            res[i++] = iter.next();
        for (int k : hs2) //hashset support foreach loop. Note there is no way to directly transform a Set to a primitive int array.
                            // only hs.toArray(new Integer[hs.size()]); but this is Integer[]
            res[i++] = k;
        return res;
    }

    //350
    public int[] intersect(int[] nums1, int[] nums2) {
        if (nums1 == null || nums1.length == 0 || nums2 == null || nums2.length == 0)
            return new int[0];
        Arrays.sort(nums1);
        Arrays.sort(nums2);
        List<Integer> res = new ArrayList<>();
        int i1 = 0, i2 = 0;
        while (i1 < nums1.length && i2 < nums2.length){
            if (nums1[i1] < nums2[i2])
                ++i1;
            else if (nums1[i1] > nums2[i2])
                ++i2;
            else {
                res.add(nums1[i1]);
                ++i1;
                ++i2;
            }
        }
        int i = 0;
        int[] r = new int[res.size()];
        for (int x : res)
            r[i++] = x;
        return r;
    }

    //351
    public int numberOfPatterns(int m, int n) {
        if (m < 1 || m > 9 || n < 1 || n > 9)
            return 0;
        int[][] mid = new int[10][10];
        mid[1][3] = mid[3][1] = 2;
        mid[1][7] = mid[7][1] = 4;
        mid[9][7] = mid[7][9] = 8;
        mid[9][3] = mid[3][9] = 6;
        mid[2][8] = mid[8][2] = mid[4][6] = mid[6][4] = mid[1][9] = mid[9][1] = mid[3][7] = mid[7][3] = 5;
        boolean[] visited = new boolean[10];
        visited[0] = true;
        return numberOfPatternsHelper(m, n, mid, 1, 1, visited) * 4 +
                numberOfPatternsHelper(m, n, mid, 2, 1, visited) * 4 +
                numberOfPatternsHelper(m, n, mid, 5, 1, visited);
    }

    private int numberOfPatternsHelper(int m, int n, int[][] mid, int s, int len, boolean[] visited){
        int cnt = 0; //since this dfs, the middle steps will also count, so this is the technic. set a counter to 0
        if (len >= m && len <= n)
            cnt = 1;
        if (len > n)
            return 0;
        visited[s] = true;
        for (int i = 1; i <= 9; ++i){
            if (!visited[i] && visited[mid[s][i]])
                cnt += numberOfPatternsHelper(m, n, mid, i, len + 1, visited); //and use += all sub senarios originated based on this step
        }
        visited[s] = false;
        return cnt; //and return the total to the parent
    }

    //352
    public class SummaryRanges {
        private TreeMap<Integer, Interval> tm; //here must use TreeMap not Map, because otehrwise lowerEntry() and like wont be available to Map interface!

        /** Initialize your data structure here. */
        public SummaryRanges() {
            tm = new TreeMap<>(); //key -start
        }

        public void addNum(int val) {
            if (tm.containsKey(val))
                return; //otherwise lower and higher will not make sense
            Integer lower = tm.lowerKey(val);
            Integer higher= tm.higherKey(val);
            //move betwen [***0]_1 _2 _3 [***] 4 cases
            if (lower != null && higher != null && tm.get(lower).end + 1 == val && val + 1 == higher){
                tm.get(lower).end = tm.get(higher).end;
                tm.remove(higher);
            }
            else if (lower != null && val <= tm.get(lower).end + 1){
                tm.get(lower).end = Math.max(tm.get(lower).end, val);
            }
            else if (higher != null && val + 1 == higher){
                Interval ni = new Interval(val, tm.get(higher).end);
                tm.put(val, ni);
                tm.remove(higher);
            }
            else
                tm.put(val, new Interval(val, val));
        }

        public List<Interval> getIntervals() {
            return new ArrayList<>(tm.values());
        }
    }

    //353
    public class SnakeGame {

        /** Initialize your data structure here.
         @param width - screen width
         @param height - screen height
         @param food - A list of food positions
         E.g food = [[1,1], [1,0]] means the first food is positioned at [1,1], the second is at [1,0]. */

        private int width, height, foodIdx, score;
        private int[][] food;
        Deque<Integer> snake; //this contains the whole snake; move : add/remove head tail
        Set<Integer> body; // this quickly checks if eats itself. so each move is o(1) lookthrough deque is o(n)

        public SnakeGame(int width, int height, int[][] food) {
            this.width = width;
            this.height = height;
            this.food = food;
            snake = new ArrayDeque<>();
            body = new HashSet();
            snake.offerLast(0);
            body.add(0);
        }

        /** Moves the snake.
         @param direction - 'U' = Up, 'L' = Left, 'R' = Right, 'D' = Down
         @return The game's score after the move. Return -1 if game over.
         Game over when snake crosses the screen boundary or bites its body. */
        public int move(String direction) {
            if (direction == null || !"ULRD".contains(direction))
                return -1;
            // first remove tail, then calculate new head, validate --> first make sure not eat itself or go out of bounder
            //then check if it eats food, it does, bump the score and add the tail back

            int tail = snake.peekLast(); //cannot remove because snake could be len = 1
            body.remove(tail); //but need to mark off the set

            int head = snake.peekFirst();
            int hr = head / width;
            int hc = head % width;
            switch(direction){
                case "U":
                    --hr;
                    break;
                case "D":
                    ++hr;
                    break;
                case "L":
                    --hc;
                    break;
                case "R":
                    ++hc;
                    break;
                default:
                    break;
            }
            head = hr * width + hc;
            if (hr < 0 || hr >= height || hc < 0 || hc >= width || body.contains(head))
                return -1;
            snake.offerFirst(head);
            body.add(head);

            if (foodIdx < food.length && food[foodIdx][0] == hr && food[foodIdx][1] == hc){
                body.add(snake.peekLast());
                ++foodIdx;
                ++score;
            }
            else
                snake.pollLast();
            return score;
        }
    }

    //354
    public int maxEnvelopes(int[][] envelopes) {
        if (envelopes == null || envelopes.length == 0 || envelopes[0].length != 2)
            return 0;
        //(width, height) sort according to width, then this problem becomes to find the LIS subsequence of height 2d->1d , same as 300 LIS
        Arrays.sort(envelopes, (a1, a2) -> a1[0] == a2[0]? a2[1] - a1[1]: a1[0] - a2[0]); //Comparator<int[]>
        //here when width is the same , put the higher height before lower, so the LIS will use the heigher, otherwise (2, 6) (2, 8) will be counted but should not
        List<Integer> tails = new ArrayList<>();
        tails.add(envelopes[0][1]);
        for (int i = 1; i < envelopes.length; ++i){
            int y = envelopes[i][1];
            int l = 0, r = tails.size() - 1, m;
            while (l <= r){
                m = l + ((r - l) >> 1);
                if (y > tails.get(m))
                    l = m + 1;
                else
                    r = m - 1;
            }
            if (l == tails.size())
                tails.add(y);
            else
                tails.set(l, y);
        }
        return tails.size();
    }

    //356
    public boolean isReflected(int[][] points) {
        if (points == null||points.length == 0) //this treat empty and single points to be valid cases
            return true;
        //store map of x -> set(y), find every x check if min+max-x exist and y matches
        //Map<Integer, Set<Integer>> hm = new HashMap<>();
        Set<String> hs = new HashSet<>();
        int min = points[0][0], max = points[0][0];
        for (int[] p : points){
            hs.add(p[0] + "#" + p[1]); //this technical should be used when java does not have pair class, use string is faster than Map<x, Set<>> kindof
            min = Math.min(min, p[0]);
            max = Math.max(max, p[0]);
        }
        long t = min + max;
        for (int [] p : points){
            int ref = (int)(t - p[0]);
            if (!hs.contains(ref + "#" + p[1]))
                return false;
        }
        return true;

    }

    //357
    public int countNumbersWithUniqueDigits(int n) {
        //n=1 10 n>= 2 9*9*8*7 every digit is 11-k purely math permutation
        if (n <= 0)
            return 1; //res is [0, 10^x)
        if (n == 1)
            return 10;
        int tmp = 9,res = 10;
        for (int k = 2; k <= n; ++k){
            tmp *= (11 - k);
            res += tmp;//cumulative under 10^x
        }
        return res;
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

    //360
    public int[] sortTransformedArray(int[] nums, int a, int b, int c) {
        //ax2 + bx + c is a parabola, when a> 0 open up, the middle are smaller, otherwise vice versa, so we can two pointer and output are already sorted o(n)
        if (nums == null || nums.length == 0)
            return new int[0];
        int i = a >= 0? nums.length - 1: 0;
        int l = 0, r = nums.length - 1;
        int[] res = new int[nums.length];
        while (l <= r){
            int lv = sortTransformedArrayHelper(nums[l], a, b, c);
            int rv = sortTransformedArrayHelper(nums[r], a, b, c);
            if (a >= 0) {
                if (lv > rv) {
                    res[i--] = lv;
                    ++l;
                } else {
                    res[i--] = rv;
                    --r;
                }
            }
            else {
                if (lv > rv){
                    res[i++] = rv;
                    --r;
                }
                else {
                    res[i++] = lv;
                    ++l;
                }
            }
        }
        return res;
    }

    private int sortTransformedArrayHelper(int x, int a, int b, int c){
        return a * x * x + b * x + c;
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

    //363
//    public int maxSumSubmatrix(int[][] matrix, int k) {
//
//    }

    //366
    public List<List<Integer>> findLeaves(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null)
            return res;
        while (root.left != null || root.right != null) {
            List<Integer> combi = new ArrayList<Integer>();
            findLeavesHelper(root, null, combi, res);
        }
        res.add(Arrays.asList(root.val));
        return res;
    }

    private void findLeavesHelper(TreeNode root, TreeNode pre, List<Integer> combi, List<List<Integer>> res){
        if (root == null)
            return;
        if (root.left == null && root.right == null){
            combi.add(root.val);
            if (pre.left == root)
                pre.left = null;
            else
                pre.right = null;
            return;
        }
        findLeavesHelper(root.left, root, combi, res);
        findLeavesHelper(root.right, root, combi, res);
    }

    //367
    public boolean isPerfectSquare(int num) {
        if (num < 0)
            return false;
        else if (num <= 1)
            return true;

        long l = 1, r = num, m;//must convert to long to prevent overflow!!
        while (l <= r){
            m = l + ((r-l) >> 1);

            if (m*m > num)
                r = m - 1;
            else if (m*m < num)
                l = m + 1;
            else
                return true;
        }
        return false;
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

    //370
    public int[] getModifiedArray(int length, int[][] updates) {
        int[] res = new int[length];
        if (updates == null || updates.length == 0 || updates[0].length != 3)
            return res;
        for (int i = 0; i < updates.length; ++i){
            res[updates[i][0]] += updates[i][2];
            if (updates[i][1] + 1 < length)
                res[updates[i][1] + 1] -= updates[i][2];
        }
        for (int i = 1; i < res.length; ++i)
            res[i] += res[i-1];
        return res;
    }

    //371
    public int getSum(int a, int b) {
        if (b == 0)
            return a;
        int add = a ^ b;
        int carry = a & b << 1;
        return getSum(add, carry);
    }

    //373
    public List<int[]> kSmallestPairs(int[] nums1, int[] nums2, int k) {
        List<int[]> res = new ArrayList<int[]>();
        if (nums1 == null || nums1.length == 0 || nums2 == null || nums2.length == 0 || k <= 0)
            return res;
        //kth smallest use MAX HEAP!!!
        Queue<int[]> pq = new PriorityQueue<>(k, (a1, a2)->a2[0] + a2[1] - a1[0] - a1[1]); //PriorityQueue(initial_capacity, Comparator)
        for (int i = 0; i < Math.min(k, nums1.length); ++i){
            for (int j = 0; j < Math.min(k, nums2.length); ++j){
                if (pq.size() < k)
                    pq.offer(new int[]{nums1[i], nums2[j]}); //Look at how create and initialize an array
                else if (nums1[i] + nums2[j] < pq.peek()[0] + pq.peek()[1]){
                    pq.poll();
                    pq.offer(new int[]{nums1[i], nums2[j]});
                }
            }
        }
        res.addAll(pq);
        return res;
    }

    //374
    int guess(int num){return 0;}

    public int guessNumber(int n) {
        if (n < 1)
            return -1;
        int l = 1, r = n, m;
        while (l <= r){
            m = l + ((r - l) >> 1);
            int x = guess(m);
            if (x < 0)
                r = m - 1;
            else if (x > 0)
                l = m + 1;
            else
                return m;
        }
        return -1;
    }

    //377
    public int combinationSum4(int[] nums, int target) {
        //same to 322 coin change.every element can literally taken by unlimitted times, so dp[i] += dp[i-nums[k]] for every k < i, dp[i] means for target=i, #of combi
        if (nums == null || nums.length == 0)
            return 0;
        int[] dp = new int[target + 1];
        dp[0] = 1;
        Arrays.sort(nums);
        for (int i = 1; i < dp.length; ++i){
            for (int j = 0; j < nums.length && nums[j] <= i ; ++j){
                dp[i] += dp[i - nums[j]];
            }
        }
        return dp[dp.length - 1];
    }

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

    //382
    /* Resevoir sampling , sample K elem from a N sequece
    S has items to sample, R will contain the result

    ReservoirSample(S[1..n], R[1..k])
    // fill the reservoir array
    for i = 1 to k
        R[i] := S[i] FILL THE FIRST K

    // FROM K+1, LOOP AND GET RANDOM NUMBER UP TO CUR, IF FALL IN THE RESEVOIR, REPLACE WITH CUR
    for i = k+1 to n
        j := random(1, i)   // important: inclusive range
        if j <= k
            R[j] := S[i]
    */
    public class Solution1 {
        private ListNode head;
        private Random random;
        /** @param head The linked list's head.
        Note that the head is guaranteed to be not null, so it contains at least one node. */
        public Solution1(ListNode head) {
            this.head = head;
            random = new Random();
        }

        /** Returns a random node's value. */
        public int getRandom() {
            //set a k = 1 resevoir
            if (head == null) return -1;
            ListNode cur = head;
            int res = cur.val;
            int i = 1;
            while (cur != null){
                if (random.nextInt(i) == 0) { //K = 1 SIZE RESEVOIR
                    res = cur.val; //the res is the resevoir
                }
                cur = cur.next;
                ++i;
            }
            return res;
        }
    }

    //383
    public boolean canConstruct(String ransomNote, String magazine) {
        if (ransomNote == null || magazine == null)
            return false;
        //since it's all ascii, array is faster than hm
        int[] counter = new int[128];
        for (int i = 0; i < magazine.length(); ++i)
            ++counter[magazine.charAt(i)];
        for (int i = 0; i < ransomNote.length(); ++i){
            if (--counter[ransomNote.charAt(i)] < 0)
                return false;
        }
        return true;
    }

    //384
    public class Solution3 {
        private int[] nums;
        private int[] lnums;
        private Random random;
        public Solution3(int[] nums) {
            this.nums = nums;
            this.lnums = Arrays.copyOf(nums, nums.length); //Here Arrays.copyOf(int[] a, int newLen) return a deep copy of Array!
            random = new Random();
        }

        /** Resets the array to its original configuration and return it. */
        public int[] reset() {
            this.lnums = Arrays.copyOf(nums, nums.length);
            return this.lnums;
        }

        /** Returns a random shuffling of the array. */
        public int[] shuffle() {
            for (int i = 1; i < lnums.length; ++i){
                int rdm = random.nextInt(i+1); //note here must include itself, like (1,2) if not include 2 will always be (2,1)
                int t = lnums[rdm];
                lnums[rdm] = lnums[i];
                lnums[i] = t;
            }
            return lnums;
        }
    }

    //386
    public List<Integer> lexicalOrder(int n) {
        List<Integer> res = new ArrayList<>();
        if (n < 1)
            return res;
        for (int i = 1; i <= 9; ++i){
            lexicalOrderHelper(n, i, res);
        }
        return res;
    }

    private void lexicalOrderHelper(int n, int i, List<Integer> res){
        if (i > n)
            return;
        res.add(i);
        for (int j = i * 10; j <= n && j <= i * 10 + 9; ++j){
            lexicalOrderHelper(n, j, res);
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

    //388
    public int lengthLongestPath(String input) {
        //use stack to store the length of sum of length of dir, stack size will be the current depth. when meet a file, get the count and update max
        if (input == null || input.length() == 0)
            return 0;
        String[] tokens = input.split("\\n");
        Deque<Integer> st = new ArrayDeque<>();
        int max = 0;

        for (String s : tokens){
            int depth = s.lastIndexOf('\t') + 1; //note \t or \n is a ascii char !!! string.lastIndexOf(char/string) return the start index from the right, -1 if not found
            //adjust level
            while (depth < st.size())
                st.pop();
            int len = s.substring(depth).length();

            if (s.indexOf('.') != -1)//it's a file
                max = Math.max(max, len + (st.isEmpty()? 0: st.peek() + st.size()));
            else {//it's a dir
                st.push(len + (st.isEmpty()? 0 : st.peek())); //cumulative len of dir
            }
        }
        return max;
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

    //392
    public boolean isSubsequence(String s, String t) {
        if (s == null || t == null)
            return false;
        for (int i = 0, j = 0; i < s.length(); ++i){
            char sc = s.charAt(i);
            while (j < t.length() && t.charAt(j) != sc)
                ++j;
            if (j == t.length())
                return false;
            else
                ++j; //need to pass this "aa" -"ab"
        }
        return true;
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

    //394
    public String decodeString(String s) {
        if (s == null || s.length() == 0)
            return "";
        //when meet [, push stack for both number and string, when ] pop both
        Deque<Integer> stNum = new ArrayDeque<>();
        Deque<StringBuilder> stStr = new ArrayDeque<>();
        StringBuilder sb = new StringBuilder();
        int num = 0;
        for (int i = 0; i < s.length(); ++i){
            char c = s.charAt(i);

            if (Character.isDigit(c)){
                num = num * 10 + c - '0';
            }
            else if (c == '['){
                stNum.push(num);
                num = 0;
                stStr.push(sb);
                sb = new StringBuilder();
            }
            else if (c == ']'){
                StringBuilder pre = stStr.pop();
                int times = stNum.pop();
                while (times-- > 0)
                    pre.append(sb);
                sb = pre;
            }
            else {
                sb.append(c);
            }
        }
        return sb.toString();
    }

    //396
    public int maxRotateFunction(int[] A) {
        if (A == null || A.length == 0)
            return 0;
        // fx+1 = fx + sum - A[n-i]
        int sum = 0, t = 0;
        for (int i = 0; i < A.length; ++i) {
            sum += A[i];
            t += i * A[i];
        }
        int max = t;
        for (int i = A.length - 1; i > 0; --i){
            t = t + sum - A.length * A[i];
            max = Math.max(max, t);
        }
        return max;
    }
    //397
    public int integerReplacement(int n) {
        if (n <= 1)
            return 0;
        if (n % 2 == 0)
            return 1 + integerReplacement(n / 2);
        else {
            return 1 + Math.min(integerReplacement(n - 1), integerReplacement(n + 1));
        }
    }

    //398
    public class Solution2 {
        private int[] nums;
        private Random random;

        public Solution2(int[] nums) {
            this.nums = nums;
            random = new Random();
        }

        public int pick(int target) {
            //K = 1 Resevoir; resevoir sampling must walk through the entire sequence!
            int res = -1;
            for (int cnt = 0, i = 0; i < nums.length; ++i){
                if (nums[i] == target){
                    ++cnt;
                    if (random.nextInt(cnt) == 0)
                        res = i;
                }
            }
            return res;
        }
    }

    //404
    private int sumLeft;
    public int sumOfLeftLeaves(TreeNode root) {
        if (root == null)
            return 0;
        sumOfLeftLeavesHelper(root, null);
        return this.sumLeft;
    }

    private void sumOfLeftLeavesHelper(TreeNode root, TreeNode pre){
        if (root == null)
            return;
        if (root.left == null && root.right == null){
            if (pre != null && pre.left == root)
                this.sumLeft += root.val;
            return;
        }
        sumOfLeftLeavesHelper(root.left, root);
        sumOfLeftLeavesHelper(root.right, root);
    }

}
