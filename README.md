## Multi-Agent Search Algorithms

Minimax Algorithm: <br>
• Adversarial search technique used for decision-making in game theory. <br>
• Handles multiple minimizing agents (ghosts) and a maximizing agent (Pacman). <br>
• Treats ghosts as adversaries.

Alpha-Beta Pruning: <br>
• Optimization of the minimax algorithm. <br>
• Reduces the number of nodes evaluated in the search tree without affecting final result.

Expectimax Algorithm: <br>
• Extends minimax by handling probabilistic outcomes. <br>
• Models behavior as agents that act randomly rather than optimally. <br>
• Assuming probabilistic/random behavior

Evaluation Function Design: <br>
State Evaluation - crafting heuristics to assess the desirability of a game state (distance to food, proximity to ghosts, score). <br>
Action Evaluation - assessing the immediate outcome of actions to inform decision-making.

Search Tree Management: <br>
• Recursive implementation <br>
• Depth management to balance performance and decision quality.

Reflex Agents: <br>
• Agents that make decisions based solely on the current state.
