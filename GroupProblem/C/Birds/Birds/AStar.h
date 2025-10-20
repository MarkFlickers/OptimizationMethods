#include <vector>
#include <memory>
#include <cstdint>
#include <functional>
#include <queue>
#include <unordered_set>


struct Move {
    uint32_t src_branch;
    uint32_t src_pos;
    uint32_t dst_branch;
    uint32_t dst_pos;
    char bird;
};

typedef struct{
    size_t steps_amount;
    std::vector<Move> Moves;
} SolvedTree;

class TreeState {
private:
    std::vector<char> data_;          // Все данные в одном векторе
    uint32_t branches_count_;
    uint32_t branch_len_;
    size_t hash_;

    void flatten(const std::vector<std::vector<char>>& state);

public:
    TreeState() = default;
    TreeState(uint32_t branches, uint32_t branch_len, const std::vector<std::vector<char>>& state);
    TreeState(uint32_t branches, uint32_t branch_len, const std::vector<char>& flat_data);

    void computeHash(void);
    // Доступ к элементам
    char& at(uint32_t branch, uint32_t pos) { return data_[branch * branch_len_ + pos]; }
    const char& at(uint32_t branch, uint32_t pos) const { return data_[branch * branch_len_ + pos]; }

    // Получение сырых данных
    const std::vector<char>& getData() const { return data_; }
    uint32_t getBranchesCount() const { return branches_count_; }
    uint32_t getBranchLen() const { return branch_len_; }
    size_t getHash() const { return hash_; }

    bool operator==(const TreeState& other) const;
};

class Tree {
private:
    TreeState state_;
    size_t unperfectness_;

    void computeUnperfectness();
    size_t computeBranchUnperfectness(uint32_t branch_index);

public:
    Tree(const TreeState& state);
    Tree(const Tree& parent, const Move& move);

    const TreeState& getState() const { return state_; }
    size_t getUnperfectness() const { return unperfectness_; }
    size_t getHash() const { return state_.getHash(); }

    bool operator==(const Tree& other) const;
};

class Node {
private:
    Tree tree_;
    int g_;
    size_t f_;
    std::shared_ptr<Node> parent_;
    Move move_;

public:
    Node(Tree&& tree, std::shared_ptr<Node> parent, const Move& move, int g);

    const Tree& getTree() const { return tree_; }
    int getG() const { return g_; }
    size_t getF() const { return f_; }
    std::shared_ptr<Node> getParent() const { return parent_; }
    const Move& getMove() const { return move_; }
    size_t getHash() const { return tree_.getHash(); }

    bool operator>(const Node& other) const;
};

class AStarSolver {
private:
    TreeState start_state_;

public:
    AStarSolver(const TreeState& start_state);
    SolvedTree solve();
    std::vector<Move> findPossibleMoves(const Tree& tree) const;
    Tree applyMove(const Tree& tree, const Move& move) const;

private:
    void processNeighbors(const Node& current_node,
        std::priority_queue<Node, std::vector<Node>, std::greater<Node>>& open_list,
        std::unordered_set<size_t>& closed_set,
        std::vector<std::shared_ptr<Node>>& all_nodes) const;
};