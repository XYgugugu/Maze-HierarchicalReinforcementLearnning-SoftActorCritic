from Actor_critic import SAC
import pygame
from maze import Maze
from player import Player
import torch
from policies import ReplayBuffer
import matplotlib.pyplot as plt

pygame.init()

pygame.font.init()
font = pygame.font.SysFont('Arial', 24)

SCREEN_SIZE = (2000, 1000)
CELL_SIZE = 50  

screen = pygame.display.set_mode(SCREEN_SIZE)

maze = Maze('./data/mazes.txt')
player = Player(maze.start_pos)  

MOVE_DELAY = 20
last_move_time = 0 

state_dim = 4
action_dim = 4
batch_size = 20
action_mapping = {
    0: "up",
    1: "down",
    2: "left",
    3: "right"
}
Save_path = "/Users/rzty/Regular/HRL/Maze-HierarchicalReinforcementLearnning-SoftActorCritic/"
SAC_test = SAC(state_dim, action_dim, gamma=0.99, tau=0.005, alpha=0.1, lr=3e-5)
replay_buffer = ReplayBuffer(batch_size, state_dim, 1)


def buffer_generator(steps):
    global last_move_time
    # player = Player(maze.start_pos) 
    for i in range(steps):
            # 事件处理
            # 获取当前时间
            pygame.time.delay(20) 
            current_time = pygame.time.get_ticks()
            # 只有在上次移动 20 毫秒后才允许再次移动
            if current_time - last_move_time > MOVE_DELAY:
                initial_state = [player.pos[0],player.pos[1],len(player.discovery),maze.current_maze_index]
                tensor_state = torch.tensor(initial_state, dtype=torch.float32)
                action,_ = SAC_test.actor.sample_action(tensor_state)
                direction = action_mapping[action.item()]
                player.move(direction, maze)
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
            new_state = [player.pos[0],player.pos[1],len(player.discovery),maze.current_maze_index]
            if i!=steps-1:
                replay_buffer.add(initial_state,action,player.delta_score,new_state,0)
            else:
                replay_buffer.add(initial_state,action,player.delta_score,new_state,1)
            # 更新屏幕
            pygame.display.flip()
    print(f"Game over with score: {player.score}")
    name = "buffer1"
    # replay_buffer.save(Save_path,name)
    

# def train_sac(batch_size):
   

if __name__=="__main__":
    critic1_loss = []
    critic2_loss = []
    actor_loss = []
    max_iteration = 100
    _,axes = plt.subplots(3,1)
    for i in range(max_iteration):
        buffer_generator(batch_size)
        critic1,critic2,actor_l =  SAC_test.update(replay_buffer,batch_size)
        critic1_loss.append(critic1)
        critic2_loss.append(critic2)
        actor_loss.append(actor_l)

        
        axes[0].cla()
        axes[0].plot(range(len(critic1_loss)), critic1_loss, label='Critic 1 Loss', color='blue')
        axes[0].legend()

        axes[1].cla()
        axes[1].plot(range(len(critic2_loss)), critic2_loss, label='Critic 2 Loss', color='orange')
        axes[1].legend()

        axes[2].cla()
        axes[2].plot(range(len(actor_loss)), actor_loss, label='Actor Loss', color='green')
        axes[2].legend()

        # 设置标题和轴标签（重置时需要重新添加）
        axes[0].set_title("Critic 1 Loss")
        axes[0].set_xlabel("Iterations")
        axes[0].set_ylabel("Loss")

        axes[1].set_title("Critic 2 Loss")
        axes[1].set_xlabel("Iterations")
        axes[1].set_ylabel("Loss")

        axes[2].set_title("Actor Loss")
        axes[2].set_xlabel("Iterations")
        axes[2].set_ylabel("Loss")

    # 更新绘图
        pygame.display.flip()
        plt.pause(0.1) 
    pygame.quit()
    plt.show()




