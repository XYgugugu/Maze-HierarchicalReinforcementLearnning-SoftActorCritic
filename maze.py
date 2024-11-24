import pygame
from player import Player

class Maze:
    def __init__(self, file_path):
        # 从文件加载多个迷宫
        self.mazes, self.dict_gem = self.load_mazes_from_file(file_path)
        self.current_maze_index = 0
        self.grid, self.start_pos, self.prev_pos, self.exit_pos = self.mazes[self.current_maze_index]

    def load_mazes_from_file(self, file_path):
        mazes = []
        dict_gem = {}
        with open(file_path, 'r') as f:
            content = f.read().strip().split("\n\n")  # 迷宫之间用空行分隔
            for maze_idx, maze_block in enumerate(content):
                lines = maze_block.strip().splitlines()
                grid_size = int(lines[0])  # 读取迷宫的尺寸
                grid = []
                start_pos = None
                exit_pos = None
                prev_pos = None
                for x, line in enumerate(lines[1:]):  # 跳过第一行 grid_size
                    row = []
                    for y, char in enumerate(line.strip().split()):
                        if char == '1':
                            row.append(1)  # 墙壁
                        elif char == '0':
                            row.append(0)  # 空地
                        elif char == '2':
                            row.append(2)  # 下一层/出口
                            exit_pos = (x, y)
                        elif char == '3':
                            row.append(3)  # 上一层
                            prev_pos = (x, y)
                        elif char.startswith('gem1'):
                            gem_info = char.split('_')
                            gem = Gem1(int(gem_info[1]), int(gem_info[2]))
                            dict_gem[(maze_idx, x, y)] = gem
                            row.append(gem)
                        elif char.startswith('gem2'):
                            gem_info = char.split('_')
                            gem_rewards = [int(r) for r in gem_info[1].split('-')]
                            gem_cond = [int(c) for c in gem_info[2].split('-')]
                            gem = Gem2(gem_rewards, gem_cond)
                            dict_gem[(maze_idx, x, y)] = gem
                            row.append(gem)
                        elif char == 'S':
                            row.append(0)  # 玩家起始位置为空地
                            start_pos = (x, y)
                    grid.append(row)
                mazes.append((grid, start_pos, prev_pos, exit_pos))
        return mazes, dict_gem

    def switch_to_next_maze(self):
        # 切换到下一个迷宫
        if self.current_maze_index < len(self.mazes) - 1:
            self.current_maze_index += 1
            self.grid, self.start_pos, self.prev_pos, self.exit_pos = self.mazes[self.current_maze_index]
            return True
        return False
    def switch_to_prev_maze(self):
        if self.current_maze_index != 0:
            self.current_maze_index -= 1
            self.grid, self.start_pos, self.prev_pos, self.exit_pos = self.mazes[self.current_maze_index]


    def draw(self, screen, cell_size):
        # 绘制当前迷宫
        for x in range(len(self.grid)):
            for y in range(len(self.grid[x])):
                rect = pygame.Rect(y * cell_size, x * cell_size, cell_size, cell_size)
                if self.grid[x][y] == 1:
                    pygame.draw.rect(screen, (0, 0, 0), rect)  # 墙壁
                elif self.grid[x][y] == 2:
                    pygame.draw.rect(screen, (0, 255, 0), rect)  # 出口
                elif self.grid[x][y] == 3:
                    pygame.draw.rect(screen, (123,123,0), rect) # 前一层
                elif type(self.grid[x][y]) is Gem1:
                    pygame.draw.rect(screen, (124, 255, 124), rect) # 宝石1
                elif type(self.grid[x][y]) is Gem2:
                    pygame.draw.rect(screen, (255, 78, 255), rect) # 宝石2
                else:
                    pygame.draw.rect(screen, (255, 255, 255), rect)  # 空地
                pygame.draw.rect(screen, (0, 0, 0), rect, 1)  # 格子的边框

    def is_exit(self, pos):
        return self.grid[pos[0]][pos[1]] == 2

    def is_prev(self, pos):
        return self.grid[pos[0]][pos[1]] == 3

    def is_gem(self, pos):
        return isinstance(self.grid[pos[0]][pos[1]], Gem), self.grid[pos[0]][pos[1]]

class Gem():
    def __init__(self, reward, cond) -> None:
        self.reward = reward
        self.condition = cond
    def verify_collectibility(self, player: Player) -> bool:
        pass
    def get_reward(self, player: Player = None) -> int:
        pass

class Gem1(Gem):
    def __init__(self, reward, cond) -> None:
        super().__init__(reward, cond)
    
    def verify_collectibility(self, player: Player) -> bool:
        return self.condition in player.discovery
    
    def get_reward(self, player: Player = None) -> int:
        return self.reward
    
    
class Gem2(Gem):
    def __init__(self, reward, cond) -> None:
        super().__init__(reward, cond)
    
    def verify_collectibility(self, player: Player) -> bool:
        return True
    
    def get_reward(self, player: Player = None):
        if player is None:
            return 0
        count = 0
        for c in self.condition:
            if c in player.discovery:
                count += 1
        return self.reward[count]