import pygame
from scoresystem import ScoreSystem

class Player:
    def __init__(self, start_pos):
        self.pos = start_pos  # 玩家初始位置
        self.discovery = set([0])
        
        self.score = 0
        self.delta_score = 0
        
        self.score_system = ScoreSystem()
        self.score_system.record_pos(start_pos)

    def move(self, direction, maze):
        from maze import Gem
        
        x, y = self.pos
        self.delta_score = 0
        # 获取当前迷宫的网格尺寸
        grid_size_x = len(maze.grid)       # 当前迷宫的行数
        grid_size_y = len(maze.grid[0])    # 当前迷宫的列数

        # 基于迷宫网格限制玩家的移动
        if direction == 'up' and x > 0 and maze.grid[x - 1][y] != 1:
            x, y = (x - 1, y)
        elif direction == 'down' and x < grid_size_x - 1 and maze.grid[x + 1][y] != 1:
            x, y = (x + 1, y)
        elif direction == 'left' and y > 0 and maze.grid[x][y - 1] != 1:
            x, y = (x, y - 1)
        elif direction == 'right' and y < grid_size_y - 1 and maze.grid[x][y + 1] != 1:
            x, y = (x, y + 1)
        
        # 检测目标是否为宝石
        if isinstance(maze.grid[x][y], Gem):
            gem = maze.grid[x][y]
            # 获取判定
            if gem.verify_collectibility(self):
                self.pos = (x, y)
        else:
            self.pos = (x, y)
        
        # 评分
        self.delta_score += (self.score_system.default_penalty() + self.score_system.action_move_reward(self.pos))
        self.score_system.record_pos(self.pos)
        self.score += self.delta_score

    def tp(self, dest, current_maze_index):
        # 到达新区域 - 加分
        if current_maze_index not in self.discovery:
            self.discovery.add(current_maze_index)
            self.delta_score = self.score_system.new_discovery_reward(current_maze_index)
            self.score += self.delta_score
        self.pos = dest
        self.score_system.record_pos(self.pos)
    
    def draw(self, screen, cell_size):
        # 绘制玩家
        player_rect = pygame.Rect(self.pos[1] * cell_size, self.pos[0] * cell_size, cell_size, cell_size)
        pygame.draw.rect(screen, (0, 0, 255), player_rect)
