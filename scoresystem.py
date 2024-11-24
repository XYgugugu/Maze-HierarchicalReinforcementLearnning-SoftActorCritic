class ScoreSystem():
    def __init__(self) -> None:
        self.penalty_per_action_default = -1
        self.penalty_dup_step = -2
        self.path_tracker = set()
    
    def default_penalty(self):
        return self.penalty_per_action_default
    
    def record_pos(self, pos):
        self.path_tracker.add(pos)
    
    def reset_path_tracker(self):
        self.path_tracker.clear()
    
    def verify_dup_action(self, pos):
        return pos in self.path_tracker
    
    def action_move_reward(self, pos):
        if self.verify_dup_action(pos):
            return self.penalty_dup_step
        return 0
    def action_gem_reward(self, reward):
        return reward
    
    def new_discovery_reward(self, new_maze_index):
        return new_maze_index*50