import pygame
from maze import Maze
from player import Player

# 初始化 Pygame
pygame.init()

pygame.font.init()
font = pygame.font.SysFont('Arial', 24)

# 游戏参数
SCREEN_SIZE = (2000, 1000)
CELL_SIZE = 50  # 每个格子的像素大小

# 创建游戏窗口
screen = pygame.display.set_mode(SCREEN_SIZE)

# 从文件加载多个迷宫
maze = Maze('./data/mazes.txt')
player = Player(maze.start_pos)  # 使用加载的玩家初始位置

# 定义移动的时间间隔为 200 毫秒
MOVE_DELAY = 200
last_move_time = 0  # 记录上次移动的时间

# 主游戏循环
running = True
while running:
    # 事件处理
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 获取当前时间
    current_time = pygame.time.get_ticks()

    # 只有在上次移动 200 毫秒后才允许再次移动
    if current_time - last_move_time > MOVE_DELAY:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            player.move('up', maze)
            last_move_time = current_time  # 更新上次移动时间
        if keys[pygame.K_DOWN]:
            player.move('down', maze)
            last_move_time = current_time  # 更新上次移动时间
        if keys[pygame.K_LEFT]:
            player.move('left', maze)
            last_move_time = current_time  # 更新上次移动时间
        if keys[pygame.K_RIGHT]:
            player.move('right', maze)
            last_move_time = current_time  # 更新上次移动时间

    # 绘制迷宫和玩家
    screen.fill((255, 255, 255))  # 背景颜色
    maze.draw(screen, CELL_SIZE)  # 绘制迷宫
    player.draw(screen, CELL_SIZE)  # 绘制玩家
    
    # 计分板
    score_text = font.render(f"Score: {player.score}", True, (0,0,0))
    delta_score_text = font.render(f"Delta Score: {player.delta_score}", True, (0,0,0))
    screen.blit(score_text, (650, 30))
    screen.blit(delta_score_text, (650, 70))

    # 检查玩家是否到达下层口
    if maze.is_prev(player.pos):
        maze.switch_to_prev_maze()
        player.score_system.reset_path_tracker()
        # player = Player(maze.start_pos, score_system=player.score_system)
        player.tp(maze.start_pos, maze.current_maze_index)

    # 检查玩家是否到达出口
    if maze.is_exit(player.pos):
        print("Reached the exit!")
        if maze.switch_to_next_maze():  # 切换到下一个迷宫 (若没有则退出游戏)
            player.score_system.reset_path_tracker()
            player.tp(maze.start_pos, maze.current_maze_index) # 重置玩家位置
        else:
            running = False
    
    # 检查玩家是否拾取宝石
    isGem, gem = maze.is_gem(player.pos)
    if isGem:
        player.delta_score += player.score_system.action_gem_reward(gem.get_reward(player))
        player.score += player.delta_score
        x, y = player.pos
        maze.grid[x][y] = 0

    # 更新屏幕
    pygame.display.flip()
print(f"Game over with score: {player.score + 500}")
pygame.quit()
