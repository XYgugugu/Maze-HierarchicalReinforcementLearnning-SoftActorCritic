from maze import Maze
from player import Player

class GameEnv():
    
    def __init__(self, maze_file = './data/mazes.txt') -> None:
        self.maze = Maze(maze_file)
        self.player = Player(self.maze.start_pos)
        self.game_over = False
    
    def move(self, direction: int):
        d = None
        if direction == 1:
            d = 'left'
        elif direction == 2:
            d = 'down'
        elif direction == 3:
            d = 'right'
        elif direction == 4:
            d = 'up'
        if d is not None:
            self.player.move(d, self.maze)
        
        self.check_tp_prev_maze()
        self.check_tp_next_maze()
        self.check_gem()
    
    def get_total_score(self):
        return self.player.score

    def get_delta_score(self):
        return self.player.delta_score

    def get_player_pos(self):
        return self.player.pos
    
    def check_tp_prev_maze(self):
        if self.maze.is_prev(self.player.pos):
            self.maze.switch_to_prev_maze()
            self.player.score_system.reset_path_tracker()
            self.player.tp(self.maze.start_pos, self.maze.current_maze_index)
    
    def check_tp_next_maze(self):
        if self.maze.is_exit(self.player.pos):
            if self.maze.switch_to_next_maze():
                self.player.score_system.reset_path_tracker()
                self.player.tp(self.maze.start_pos, self.maze.current_maze_index)
            else:
                self.game_over = True
    
    def check_gem(self):
        isGem, gem = self.maze.is_gem(self.player.pos)
        if isGem:
            from maze import Gem
            self.player.delta_score += self.player.score_system.action_gem_reward(gem.get_reward(self.player))
            self.player.score += self.player.delta_score
            x,y = self.player.pos
            self.maze.grid[x][y] = 0