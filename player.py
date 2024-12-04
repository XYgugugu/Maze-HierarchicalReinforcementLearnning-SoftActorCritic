import pygame
from scoresystem import ScoreSystem

class Player:
    def __init__(self, start_pos):
        self.pos = start_pos  # 玩家初始位置
        self.discovery = set([0])
        
        self.score = 0
        self.delta_score = 0
        self.state_dim = len(start_pos)
        self.score_system = ScoreSystem()
        self.score_system.record_pos(start_pos)
        self.action_dim = 4

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

    def step(self,action,maze,screen,CELL_SIZE):
        font = pygame.font.SysFont('Arial', 24)
        current_time = pygame.time.get_ticks()
        MOVE_DELAY = 200
        last_move_time = 0  # 记录上次移动的时间
    # 只有在上次移动 200 毫秒后才允许再次移动
        if current_time - last_move_time > MOVE_DELAY:
            keys = action
            if keys[pygame.K_UP]:
                self.move('up', maze)
                last_move_time = current_time  # 更新上次移动时间
            if keys[pygame.K_DOWN]:
                self.move('down', maze)
                last_move_time = current_time  # 更新上次移动时间
            if keys[pygame.K_LEFT]:
                self.move('left', maze)
                last_move_time = current_time  # 更新上次移动时间
            if keys[pygame.K_RIGHT]:
                self.move('right', maze)
                last_move_time = current_time  # 更新上次移动时间

        # 绘制迷宫和玩家
        screen.fill((255, 255, 255))  # 背景颜色
        maze.draw(screen, CELL_SIZE)  # 绘制迷宫
        self.draw(screen, CELL_SIZE)  # 绘制玩家
        
        # 计分板
        score_text = font.render(f"Score: {self.score}", True, (0,0,0))
        delta_score_text = font.render(f"Delta Score: {self.delta_score}", True, (0,0,0))
        screen.blit(score_text, (650, 30))
        screen.blit(delta_score_text, (650, 70))

        # 检查玩家是否到达下层口
        if maze.is_prev(self.pos):
            maze.switch_to_prev_maze()
            self.score_system.reset_path_tracker()
            # player = Player(maze.start_pos, score_system=player.score_system)
            self.tp(maze.start_pos, maze.current_maze_index)

        # 检查玩家是否到达出口
        if maze.is_exit(self.pos):
            print("Reached the exit!")
            if maze.switch_to_next_maze():  # 切换到下一个迷宫 (若没有则退出游戏)
                self.score_system.reset_path_tracker()
                self.tp(maze.start_pos, maze.current_maze_index) # 重置玩家位置
            else:
                running = False
        
        # 检查玩家是否拾取宝石
        isGem, gem = maze.is_gem(self.pos)
        if isGem:
            self.delta_score += self.score_system.action_gem_reward(gem.get_reward(self))
            self.score += self.delta_score
            x, y = self.pos
            maze.grid[x][y] = 0

        # 更新屏幕
        pygame.display.flip()
        end_ = False
        return self.pos, self.delta_score, end_,{}

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
