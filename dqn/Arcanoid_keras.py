import math
import time

import numpy as np
import random

import gym
from gym import spaces
from gym.utils import seeding

import pygame

from ddqn_keras import DDQNAgent
from utils import plotLearning

import ipdb


np.set_printoptions(formatter={'float': '{: 0.2f}'.format}, suppress=True) 

REWARD = 1
PENALTY = -1

GAME_HEIGHT = 100
GAME_WIDTH = 400
# The top of the block (y position)
top = 20 
# Number of blocks to create
blockcount = 8
SPEED = 5
FPS = 1
SHOW_EVERY = 1
# Define some colors
black = (0, 0, 0)
white = (255, 255, 255)
blue = (0, 0, 255)
 
# Size of break-out blocks
block_width = 23
block_height = 15

class Block(pygame.sprite.Sprite):
    """This class represents each block that will get knocked out by the ball
    It derives from the "Sprite" class in Pygame """
 
    def __init__(self, color, x, y):
        """ Constructor. Pass in the color of the block,
            and its x and y position. """
 
        # Call the parent class (Sprite) constructor
        super().__init__()
 
        # Create the image of the block of appropriate size
        # The width and height are sent as a list for the first parameter.
        self.image = pygame.Surface([block_width, block_height])
 
        # Fill the image with the appropriate color
        self.image.fill(color)
 
        # Fetch the rectangle object that has the dimensions of the image
        self.rect = self.image.get_rect()
 
        # Move the top left of the rectangle to x,y.
        # This is where our block will appear..
        self.rect.x = x
        self.rect.y = y
 
 
class Ball(pygame.sprite.Sprite):
    """ This class represents the ball
        It derives from the "Sprite" class in Pygame """
 
    # Speed in pixels per cycle
    speed = SPEED
 
    # Floating point representation of where the ball is
    x = 0.0
    y = 0.0
 
    width = SPEED
    height = SPEED
 
    # Constructor. Pass in the color of the block, and its x and y position
    def __init__(self):
        # Call the parent class (Sprite) constructor
        super().__init__()
 
        # Create the image of the ball
        self.image = pygame.Surface([self.width, self.height])
 
        # Color the ball
        self.image.fill(white)
 
        # Get a rectangle object that shows where our image is
        self.rect = self.image.get_rect()
 
        # Get attributes for the height/width of the screen
        self.screenheight = pygame.display.get_surface().get_height()
        self.screenwidth = pygame.display.get_surface().get_width()
        
        self.x = GAME_WIDTH/2
        # Direction of ball (in degrees)
        # self.direction = np.random.choice([180+75, 180-75, 0], 1)[0]
        self.direction = random.randint(180-75, 180+75)
        # self.direction = 180
        # rnd = random.randint(1, 100)
        # if rnd <= 50:
        #     self.direction = 75
        # else:
        #     self.direction = 360-75
 
    def bounce(self, diff):
        """ This function will bounce the ball
            off a horizontal surface (not a vertical one) """
 
        self.direction = (180 - self.direction) % 360
        # self.direction -= diff 
        # Avoid loops
        if self.direction % 90 < 1:
            self.direction += 1
        elif self.direction % 90 > 89:
            self.direction -= 1
 
    def update(self):
        result = 0
        """ Update the position of the ball. """
        # Sine and Cosine work in degrees, so we have to convert them
        direction_radians = math.radians(self.direction)
 
        # Change the position (x and y) according to the speed and direction
        self.x += self.speed * math.sin(direction_radians)
        self.y -= self.speed * math.cos(direction_radians)
 
        # Move the image to where our x and y are
        self.rect.x = self.x
        self.rect.y = self.y
 
        # Do we bounce off the top of the screen?
        if self.y <= 0:
            result = 1 # Win
            self.bounce(0)
            self.y = 1
 
        # Do we bounce off the left of the screen?
        if self.x <= 0:
            self.direction = (360 - self.direction) % 360
            self.x = 1
 
        # Do we bounce of the right side of the screen?
        if self.x > self.screenwidth - self.width:
            self.direction = (360 - self.direction) % 360
            self.x = self.screenwidth - self.width - 1
 
        # Did we fall off the bottom edge of the screen?
        if self.y > GAME_HEIGHT:
            self.y = 0.0
            result = 2 #'Lost'
        
        return result
 
 
class Player(pygame.sprite.Sprite):
    """ This class represents the bar at the bottom that the
    player controls. """
 
    def __init__(self):
        """ Constructor for Player. """
        # Call the parent's constructor
        super().__init__()
 
        self.width = SPEED*8
        self.height = SPEED+1
        self.image = pygame.Surface([self.width, self.height])
        self.image.fill((white))
 
        # Make our top-left corner the passed-in location.
        self.rect = self.image.get_rect()
        self.screenheight = pygame.display.get_surface().get_height()
        self.screenwidth = pygame.display.get_surface().get_width()
 
        self.rect.x = GAME_WIDTH/2 
        self.rect.y = self.screenheight-self.height
 
    def update(self):
        """ Update the player position. """
        # Get where the mouse is
        # pos = pygame.mouse.get_pos()
        key_pressed = [i for i, k in enumerate(pygame.key.get_pressed()) if k==1]
        # print(key_pressed[79], key_pressed[80])
        if len(key_pressed) >0:
            if key_pressed[0] == 79:
                self.rect.x += SPEED*4
            elif key_pressed[0] == 80:
                self.rect.x -= SPEED*4
        # Set the left side of the player bar to the mouse position
        # self.rect.x = pos[0]
        # Make sure we don't push the player paddle
        # off the right side of the screen
        if self.rect.x > self.screenwidth - self.width:
            self.rect.x = self.screenwidth - self.width
        elif self.rect.x <= 0:
            self.rect.x = 0


class Arcanoid(gym.Env):
    """
    Description:
        The agent (a paddle) is started. For any given
        state the agent may choose to move to the left, right or cease
        any movement.
    Observation:
        Type: Box(2)
        Num    Observation               
        0      Paddle Position (x)           
        1      Ball Position (x,y)            
    Actions:
        Type: Discrete(3)
        Num    Action
        0      Move to the Left
        1      Don't move
        2      Move to the Right
    Reward:
         TBD
    """

    def __init__(self, goal_velocity=0):
        
        # Call this function so the Pygame library can initialize itself
        pygame.init()
         
        # Create a screen
        self.screen = pygame.display.set_mode([GAME_WIDTH, GAME_HEIGHT])
         
        # Set the title of the window
        pygame.display.set_caption('Breakout')
         
        # Enable this to make the mouse disappear when over our window
        # pygame.mouse.set_visible(0)
         
        # This is a font we use to draw text on the screen (size 36)
        font = pygame.font.Font(None, 36)
         
        # Create a surface we can draw on
        background = pygame.Surface(self.screen.get_size())
         
        # Create sprite lists
        self.blocks = pygame.sprite.Group()
        self.balls = pygame.sprite.Group()
        self.allsprites = pygame.sprite.Group()
         
        # Create the player paddle object
        self.player = Player()
        self.allsprites.add(self.player)
         
        # Create the ball
        self.ball = Ball()
        self.allsprites.add(self.ball)
        self.balls.add(self.ball)

        self.low = np.array(
            [0, 0, 0], dtype=np.float32
        )
        self.high = np.array(
            [GAME_WIDTH, GAME_WIDTH, GAME_HEIGHT], dtype=np.float32
        )

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            self.low, self.high, dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action):
        done = False
        reward = 0

        # Limit to 30 fps
        # clock.tick(FPS)
     
        # Clear the screen
        self.screen.fill(black)
     
        # Process the events in the game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit_program = True

        if action == 0:
            self.player.rect.x -= SPEED*2  
        elif action == 2:
            self.player.rect.x += SPEED*2
            
        
        # Update the player and ball positions
        self.player.update()
        res = self.ball.update()
        if res == 2: # Lost
            reward = PENALTY
            done = True
        elif res == 1: # Win
            reward = REWARD
            # done = True

        # if abs(self.player.rect.x + 20 - self.ball.x) <= 10:
        #     if random.randint(1, 100) <= 5:
        #         reward = REWARD
        # else:
        #     reward = PENALTY

        # See if the ball hits the player paddle
        if pygame.sprite.spritecollide(self.player, self.balls, False):            
            # The 'diff' lets you try to bounce the ball left or right
            # depending where on the paddle you hit it
            diff = (self.player.rect.x + self.player.width/2) - (self.ball.rect.x+self.ball.width/2)
     
            # Set the ball's y position in case
            # we hit the ball on the edge of the paddle
            self.ball.y = self.screen.get_height() - self.player.rect.height - self.ball.rect.height - SPEED
            self.ball.bounce(diff)

        # Check for collisions between the ball and the blocks
        deadblocks = pygame.sprite.spritecollide(self.ball, self.blocks, True)
     
        # If we actually hit a block, bounce the ball
        if len(deadblocks) > 0:
            self.ball.bounce(0)
     
        self.state = np.array([self.player.rect.x, self.ball.x, self.ball.y])
        return self.state, reward, done, {}

    def reset(self):
        self.__init__()
        x = random.randint(0, GAME_WIDTH)
        self.player.rect.x = x + np.random.randint(-3*self.player.width, 3*self.player.width, 1)[0]
        self.ball.x = x
        # self.ball.y = 0
        self.state = np.array([self.player.rect.x, self.ball.x, self.ball.y])
        return self.state

    def render(self):
        # Draw Environment
        self.allsprites.draw(self.screen) 
        # Flip the screen and show what we've drawn
        pygame.display.flip()

    def close(self):        
        pygame.quit()

 
# Clock to limit speed
clock = pygame.time.Clock()
 
# Exit the program?
exit_program = False
 
env = Arcanoid()
ddqn_agent = DDQNAgent(alpha=0.005, gamma=0.99, n_actions=env.action_space.n, epsilon=1.0,
              batch_size=64, input_dims=env.observation_space.shape[0]*2, replace_target=1000)
ddqn_scores = []
eps_history = []
history = []
n_games = 500_000
start = time.time()
for i in range(n_games):
    done = False
    observation = env.reset()
    observation = observation / [GAME_WIDTH, GAME_WIDTH, GAME_HEIGHT]
    observation = np.append([-1,-1,-1], observation) 
    while not done:
        action = ddqn_agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        observation_ = observation_ / [GAME_WIDTH, GAME_WIDTH, GAME_HEIGHT]
        observation_ = np.append(observation[-3:], observation_) 
        if not i or not i % SHOW_EVERY:
            env.render()
        # ipdb.set_trace()
        
        ddqn_agent.remember(observation, action, reward, observation_, int(done))

        observation = observation_
        if random.randint(1, 100) <= 15:
            ddqn_agent.learn()

        # Display
        eps_history.append(ddqn_agent.epsilon)
        ddqn_scores.append(reward)
        if not ddqn_agent.memory.mem_cntr % 1_000:
            pos = []
            neg = []
            for r in ddqn_scores[-1_000:]:
                if r > 0:
                    pos.append(r)
                else:
                    neg.append(r)
            ratio = np.sum(pos) / (np.sum(pos) + abs(np.sum(neg))) 

            end = time.time()
            print('episode: ', i, ' ratio %.2f' % ratio,
                  ' rewards %.2f' % np.sum(pos), ' penalties %.2f' % np.sum(neg), f'epsilon {ddqn_agent.epsilon:.2f}',
                  f'mem_cntr {ddqn_agent.memory.mem_cntr}, time {end - start:.2f}'
                  # f'reward_memory {ddqn_agent.reward_memory.mem_cntr}, penalty_memory {ddqn_agent.penalty_memory.mem_cntr}, other_memory {ddqn_agent.other_memory.mem_cntr}, '
                  )
            start = time.time()
 

# x = [i+1 for i in range(n_games)]
# plotLearning(x, ddqn_scores, eps_history, 'Arcanoid-ddqn.png')
env.close()