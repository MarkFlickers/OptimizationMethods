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

struct SolvedTree {
    size_t steps_amount;
    std::vector<Move> Moves;
    std::vector<std::vector<char>> Resultant_tree;
};

class TreeState {
private:
    std::vector<std::vector<char>> branches_;
    uint8_t birds_rename_table[27] = {0};
    uint16_t total_branches_;
    uint8_t branch_len_;
    mutable size_t hash_ = 0;
    mutable bool hash_computed_ = false;

    void createBirdsRenameTable(const std::vector<std::vector<char>>& old_state);
    void normalizeBirds();
    void sortBranches();
    bool isBranchEmpty(const std::vector<char>& branch) const;
    int compareBranches(const std::vector<char>& a, const std::vector<char>& b) const;

public:
    TreeState() = default;
    TreeState(const std::vector<std::vector<char>>& state);
    TreeState(const std::vector<std::vector<char>>& state, uint8_t *old_birds_rename_table);

    void computeHash() const;
    std::vector<std::vector<char>> getBranchesWithOriginalBirds(void) const;
    const std::vector<std::vector<char>>& getBranches() const;
    uint16_t getTotalBranches() const { return total_branches_; }
    uint8_t getBranchLen() const { return branch_len_; }
    size_t getHash() const;
    bool operator==(const TreeState& other) const;
    TreeState applyMove(const Move& move) const;
};

class Tree {
private:
    TreeState state_;
    size_t unperfectness_;

    size_t computeUnperfectnessIncremental(const Tree& parent, const Move& move);
    void computeUnperfectness();

public:
    Tree() = default;
    Tree(const TreeState& state);
    Tree(const Tree& parent, const Move& move);

    const TreeState& getState() const { return state_; }
    size_t getUnperfectness() const { return unperfectness_; }
    size_t getHash() const { return state_.getHash(); }
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
    Node() = default;
    Node(Tree&& tree, Node* parent, const Move& move, int g);

    const Tree& getTree() const { return tree_; }
    int getG() const { return g_; }
    size_t getF() const { return f_; }
    Node* getParent() const { return parent_; }
    const Move& getMove() const { return move_; }
    size_t getHash() const { return hash_; }
    void update(Node* parent, const Move& move, int g);

    bool operator>(const Node& other) const;
};

class AStarSolver {
private:
    TreeState start_state_;
    mutable std::unordered_map<size_t, Node*> node_registry_;

    static constexpr int MAX_DEPTH = 1000;

    bool shouldPrune(const Node& node) const;
    bool registerNode(Node* node);
    std::vector<Move> findPossibleMoves(const Tree& tree) const;

public:
    AStarSolver(const TreeState& start_state);
    SolvedTree solve();
};