# Hierarchical State Representation
from typing import List
class HierarchicalStateRepresentation():
    
    class HSR_Global():    
        def __init__(self, num_mazes : int) -> None:
            self.current_maze = 0
            self.total_mazes = num_mazes
            self.gem_remain_in_maze = [0] * num_mazes
            self.maze_completed = [False] * num_mazes
            self.score = 0
            self.delta_score = 0
            self.action_count = 0
        
        def update_global_state(self, collect_gem : bool = False, complete_current_maze : bool = False):
            if collect_gem is True:
                self.gem_remain_in_maze[self.current_maze] = max(0, self.gem_remain_in_maze[self.current_maze])
            if complete_current_maze is True:
                self.maze_completed[self.current_maze] 
        
        def update_global_score(self, delta):
            self.delta_score = delta
            self.score += delta
        
        def get_global(self):
            return {
                'maze_idx' : self.current_maze,
                'gem_remain_in_maze' : self.gem_remain_in_maze,
                'maze_completed' : self.maze_completed,
                'score' : self.score,
                'delta_score' : self.delta_score,
                'action_count' : self.action_count
            }
    
    class HSR_Local():
        def __init__(self, maze_layout : List[List[int]], start_pos, next_maze_pos, prev_maze_pos, agent_pos, gems_pos) -> None:
            self.maze_layout = maze_layout
            self.start_pos = start_pos
            self.next_maze_pos = next_maze_pos
            self.prev_maze_pos = prev_maze_pos
            self.agent_pos = agent_pos
            self.gems_pos = gems_pos
            self.visited_pos = set()
        
        # @TODO: update from env