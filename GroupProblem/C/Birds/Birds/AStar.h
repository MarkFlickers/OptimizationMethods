#pragma once
#include <vector>
#include <memory>
#include <cstdint>
#include <functional>
#include <queue>
#include <unordered_set>
#include <unordered_map>


struct Move {
    uint16_t src_branch;
    uint8_t src_pos;
    uint16_t dst_branch;
    uint8_t dst_pos;
    char bird;
};

typedef struct{
    size_t steps_amount;
    std::vector<Move> Moves;
} SolvedTree;

class TreeState {
private:
    std::vector<char> data_;          // ¬се данные в одном векторе
    uint16_t branches_count_;
    uint8_t branch_len_;
    mutable size_t hash_;
    mutable bool hash_computed_ = false;

    void flatten(const std::vector<std::vector<char>>& state);

public:
    TreeState() : branches_count_(0), branch_len_(0) {}
    TreeState(uint16_t branches, uint8_t branch_len, const std::vector<std::vector<char>>& state);
    TreeState(uint16_t branches, uint8_t branch_len, const std::vector<char>& flat_data);
    void computeHash() const;
    char& at(uint16_t branch, uint8_t pos);
    const char& at(uint16_t branch, uint8_t pos) const;
    const std::vector<char>& getData() const { return data_; }
    uint16_t getBranchesCount() const { return branches_count_; }
    uint8_t getBranchLen() const { return branch_len_; }
    size_t getHash() const;
    bool operator==(const TreeState& other) const;
};

class Tree {
private:
    TreeState state_;
    size_t unperfectness_;
    //size_t current_L_;  //  оличество полностью собранных веток
    //size_t max_L_;      // ћаксимально возможное количество собранных веток

    size_t computeUnperfectnessIncremental(const Tree& parent, const Move& move);
    void computeUnperfectness();

    //void computeCurrentL();
    //void computeMaxL();
    //bool isBranchComplete(uint16_t branch_index) const;

public:
    Tree() : state_(), unperfectness_(0) {} //  онструктор по умолчанию
    Tree(const TreeState& state);
    Tree(const Tree& parent, const Move& move);

    const TreeState& getState() const { return state_; }
    size_t getUnperfectness() const { return unperfectness_; }
    size_t getHash() const { return state_.getHash(); }
    //size_t getCurrentL() const { return current_L_; }
    //size_t getMaxL() const { return max_L_; }
    //size_t getHash() const { return state_.getHash(); }
    //bool isTargetState() const { return current_L_ == max_L_; }
    bool isTargetState() const { return unperfectness_ == 0; }
    bool operator==(const Tree& other) const;
};

class Node {
private:
    Tree tree_;
    int g_;
    size_t f_;
    Node* parent_;
    Move move_;
    size_t hash_;

public:
    Node() : tree_(), g_(0), f_(0), parent_(nullptr), move_(), hash_(0) {}
    Node(Tree&& tree, Node* parent, const Move& move, int g);

    const Tree& getTree() const { return tree_; }
    int getG() const { return g_; }
    size_t getF() const { return f_; }
    Node* getParent() const { return parent_; }
    const Move& getMove() const { return move_; }
    size_t getHash() const { return hash_; }

    bool operator>(const Node& other) const;
};

class NodePool
{
private:
    std::vector<std::unique_ptr<Node>> nodes_;
    size_t next_index_ = 0;

public:
    void initialize(size_t capacity);
    Node* createNode(Tree&& tree, Node* parent, const Move& move, int g);
    void reset(void);
};

class AStarSolver {
private:
    TreeState start_state_;
    mutable std::unordered_map<size_t, std::vector<Move>> moves_cache_;
    mutable std::unordered_map<size_t, std::unordered_map<size_t, Tree>> apply_move_cache_;
    mutable std::unordered_map<size_t, Node*> node_registry_;
    NodePool node_pool_;

    static constexpr int MAX_DEPTH = 1000;
    static constexpr size_t CACHE_CLEANUP_THRESHOLD = 50000;

    Tree applyMoveWithCache(const Tree& tree, const Move& move);
    size_t hashMove(const Move& move) const;
    bool shouldPrune(const Node& node) const;
    bool registerNode(Node* node);
    void cleanupCaches();
    std::vector<Move> findPossibleMovesOptimized(const Tree& tree) const;

public:
    AStarSolver(const TreeState& start_state);
    SolvedTree solve();
    Tree applyMove(const Tree& tree, const Move& move) const;
};

size_t computeBranchUnperfectnessWithCache(const TreeState& state, uint32_t branch_index);
size_t computeBranchUnperfectness(const TreeState & state, uint32_t branch_index);
size_t computeBranchHash(const char* branch_data, uint32_t branch_len);