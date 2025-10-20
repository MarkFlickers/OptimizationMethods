#include <iostream>
#include <vector>
#include "AStar.h"
#include <cstdlib>

AStarSolver *GlobalSolver_p;

typedef struct{
	uint32_t branches_on_tree;
	uint8_t branch_len;
	std::vector<std::vector<char>> TreeState;
} input_data_t;

//char Tree1[][2] = {{1, 2}, {2, 1}, {0, 0}};
//char Tree2[][4] = {{4, 3, 2, 1}, {1, 6, 3, 5}, {1, 8, 7, 7}, {1, 10, 5, 9}, {6, 10, 6, 8}, {11, 2, 11, 2}, {13, 6, 11, 4}, {13, 12, 4, 8}, {12, 11, 13, 10}, {12, 10, 4, 13}, {12, 9, 5, 7}, {3, 8, 9, 2}, {9, 3, 5, 7}, {0, 0, 0, 0}, {0, 0, 0, 0}};
//char Tree3[][6] = {{1, 2, 2, 2, 3, 1}, {2, 1, 3, 3, 1, 1}, {2, 1, 3, 2, 3, 3}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}};
//char Tree4[][4] = {{2, 3, 2, 1}, {1, 3, 5, 4}, {1, 6, 4, 2}, {1, 7, 6, 7}, {8, 8, 6, 5}, {8, 3, 2, 4}, {8, 7, 5, 5}, {4, 3, 7, 6}, {0, 0, 0, 0}, {0, 0, 0, 0}};
//char Tree5[][3] = {{2, 2, 1}, {5, 4, 3}, {5, 3, 6}, {9, 8, 7}, {9, 10, 4}, {8, 11, 6}, {13, 9, 12}, {13, 3, 1}, {13, 6, 5}, {1, 14, 7}, {10, 12, 4}, {14, 8, 14}, {12, 10, 11}, {15, 2, 11}, {7, 15, 15}, {0, 0, 0}, {0, 0, 0}};

std::vector<std::vector<char>> Tree1 = {{1, 2}, {2, 1}, {0, 0}};
std::vector<std::vector<char>> Tree2 = {{4, 3, 2, 1}, {1, 6, 3, 5}, {1, 8, 7, 7}, {1, 10, 5, 9}, {6, 10, 6, 8}, {11, 2, 11, 2}, {13, 6, 11, 4}, {13, 12, 4, 8}, {12, 11, 13, 10}, {12, 10, 4, 13}, {12, 9, 5, 7}, {3, 8, 9, 2}, {9, 3, 5, 7}, {0, 0, 0, 0}, {0, 0, 0, 0}};
std::vector<std::vector<char>> Tree3 = {{1, 2, 2, 2, 3, 1}, {2, 1, 3, 3, 1, 1}, {2, 1, 3, 2, 3, 3}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}};
std::vector<std::vector<char>> Tree4 = {{2, 3, 2, 1}, {1, 3, 5, 4}, {1, 6, 4, 2}, {1, 7, 6, 7}, {8, 8, 6, 5}, {8, 3, 2, 4}, {8, 7, 5, 5}, {4, 3, 7, 6}, {0, 0, 0, 0}, {0, 0, 0, 0}};
std::vector<std::vector<char>> Tree5 = {{2, 2, 1}, {5, 4, 3}, {5, 3, 6}, {9, 8, 7}, {9, 10, 4}, {8, 11, 6}, {13, 9, 12}, {13, 3, 1}, {13, 6, 5}, {1, 14, 7}, {10, 12, 4}, {14, 8, 14}, {12, 10, 11}, {15, 2, 11}, {7, 15, 15}, {0, 0, 0}, {0, 0, 0}};
// 
//char **Tree;
//
input_data_t DATA1 = {
	.branches_on_tree = 3,
	.branch_len = 2,
	.TreeState = Tree1,
};

input_data_t DATA2 = {
	.branches_on_tree = 15,
	.branch_len = 4,
	.TreeState = Tree2,
};

input_data_t DATA3 = {
	.branches_on_tree = 5,
	.branch_len = 6,
	.TreeState = Tree3,
};

input_data_t DATA4 = {
	.branches_on_tree = 10,
	.branch_len = 4,
	.TreeState = Tree4,
};

input_data_t DATA5 = {
	.branches_on_tree = 17,
	.branch_len = 3,
	.TreeState = Tree5,
};


int main( void )
{
	input_data_t DATA = DATA2;
	auto start_state = TreeState(DATA.branches_on_tree, DATA.branch_len, DATA.TreeState);
	auto Solver = AStarSolver(start_state);
	GlobalSolver_p = &Solver;
	auto ans = Solver.solve();
	printf("%d\n", ans.steps_amount);
	for(uint32_t i = 1; i < ans.steps_amount + 1; i++)
	{
		printf("%d \t%d \t%d \t%d\n", i, ans.Moves[i].src_branch, ans.Moves[i].dst_branch, ans.Moves[i].bird);
	}
	//while( 1 );
	system("pause");
	return 0;
}