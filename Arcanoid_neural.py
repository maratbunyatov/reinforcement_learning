"""
 Sample Breakout Game
 
 Sample Python/Pygame Programs
 Simpson College Computer Science
 http://programarcadegames.com/
 http://simpson.edu/computer-science/
"""
 
# --- Import libraries used for this program
 
import math
import pygame

import numpy as np
import sys
import math
import time
from PIL import Image
import cv2


np.set_printoptions(formatter={'float': '{: 0.2f}'.format}, suppress=True) 

ALPHA = 0.1
GAMMA = .995

REWARD = 1
PENALTY = -2

SHOW_EVERY = 10_000
PRINT = False

INPUT_SIZE = 3
HIDDEN_SIZE = 32
OUTPUT_SIZE = 3

GAME_HEIGHT = 50
GAME_WIDTH = 400
# The top of the block (y position)
top = 20 
# Number of blocks to create
blockcount = 8
SPEED = 5
FPS = 0

# Exploration settings
epsilon = .0  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = 50
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# Define some colors
black = (0, 0, 0)
white = (255, 255, 255)
blue = (0, 0, 255)
 
# Size of break-out blocks
block_width = 23
block_height = 15




class Model:
  def __init__(self):
    self.l_2 = np.append(np.random.randn(HIDDEN_SIZE) / np.sqrt(HIDDEN_SIZE), 1)

    self.w_1_2 = np.random.randn(INPUT_SIZE, HIDDEN_SIZE) / np.sqrt(INPUT_SIZE) # Xavier intitialization 
    self.w_1_2_b = np.random.randn(HIDDEN_SIZE) / np.sqrt(INPUT_SIZE) # bias weights
    self.w_2_3 = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE) / np.sqrt(HIDDEN_SIZE) 
    self.w_2_3_b = np.random.randn(OUTPUT_SIZE) / np.sqrt(HIDDEN_SIZE) 


  def s2x(self, s):
    result = []
    for i in range(len(s)):
        result.append(s[i] / GAME_HEIGHT)
    # result.append(feature*feature / (255**2))
    # result.append(abs(s[PADDLE_IDX]-s[X_IDX])**2 / 255**2)
    # result.append(0 if (s[PADDLE_IDX]-s[X_IDX]) / 255 <= 0 else 1)    
    return np.array(result)

  def perceive(self, s):
    x = self.s2x(s)    
    # forward propagate
    self.l_2 = self.sigmoid_activation(self.w_1_2.T.dot(x.T) + self.w_1_2_b.T.reshape(HIDDEN_SIZE,1))
    l3_activations = self.sigmoid_activation(self.w_2_3.T.dot(self.l_2).T + self.w_2_3_b)
    return l3_activations[0]

  def input_gradient(self, s):
    return self.s2x(s)

  def softmax_activation(self, z):
    # Softmax
    # z -= np.max(z)
    # sm = (np.exp(z).T / np.sum(np.exp(z), axis=0)).T
    sm = (np.exp(z) / np.sum(np.exp(z)))
    return sm

  def softmax_gradient(self, s):
    # Softmax
    # Take the derivative of softmax element w.r.t the each logit which is usually Wi * X
    # input s is softmax value of the original input x. 
    # s.shape = (1, n) 
    # i.e. s = np.array([0.3, 0.7]), x = np.array([0, 1])
    # initialize the 2-D jacobian matrix.
    jacobian_m = np.diag(s)
    for i in range(len(jacobian_m)):
        for j in range(len(jacobian_m)):
            if i == j:
                jacobian_m[i][j] = s[i] * (1-s[i])
            else: 
                jacobian_m[i][j] = -s[i]*s[j]
    return jacobian_m

  def sigmoid_activation(self, input):
    # Sigmoid
    return 1 / (1 + np.exp(-input))

  def sigmoid_gradient(self, s):
    # sigmoid(x) * (1 - sigmoid(x));, Assumes s is the sigmoid value
    return s * (1 - s)


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
 
    # Direction of ball (in degrees)
    direction = 200
 
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
 
    def bounce(self, diff):
        """ This function will bounce the ball
            off a horizontal surface (not a vertical one) """
 
        self.direction = (180 - self.direction) % 360
        self.direction -= diff
 
    def update(self):
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
            return True
        else:
            return False
 
 
class Player(pygame.sprite.Sprite):
    """ This class represents the bar at the bottom that the
    player controls. """
 
    def __init__(self):
        """ Constructor for Player. """
        # Call the parent's constructor
        super().__init__()
 
        self.width = SPEED*8
        self.height = SPEED
        self.image = pygame.Surface([self.width, self.height])
        self.image.fill((white))
 
        # Make our top-left corner the passed-in location.
        self.rect = self.image.get_rect()
        self.screenheight = pygame.display.get_surface().get_height()
        self.screenwidth = pygame.display.get_surface().get_width()
 
        self.rect.x = 0
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
 
# Call this function so the Pygame library can initialize itself
pygame.init()
 
# Create an 800x600 sized screen
screen = pygame.display.set_mode([GAME_WIDTH, GAME_HEIGHT])
 
# Set the title of the window
pygame.display.set_caption('Breakout')
 
# Enable this to make the mouse disappear when over our window
# pygame.mouse.set_visible(0)
 
# This is a font we use to draw text on the screen (size 36)
font = pygame.font.Font(None, 36)
 
# Create a surface we can draw on
background = pygame.Surface(screen.get_size())
 
# Create sprite lists
blocks = pygame.sprite.Group()
balls = pygame.sprite.Group()
allsprites = pygame.sprite.Group()
 
# Create the player paddle object
player = Player()
allsprites.add(player)
 
# Create the ball
ball = Ball()
allsprites.add(ball)
balls.add(ball)
 

 
# --- Create blocks
 
# Five rows of blocks
for row in range(0):
    # 32 columns of blocks
    for column in range(0, blockcount):
        # Create a block (color,x,y)
        block = Block(blue, column * (block_width + 2) + 1, top)
        blocks.add(block)
        allsprites.add(block)
    # Move the top of the next row down
    top += block_height + 2
 
# Clock to limit speed
clock = pygame.time.Clock()
 
# Exit the program?
exit_program = False
 

# Main program loop
def step(action): 
    reward = 0

    clock.tick(FPS)
 
    # Clear the screen
    screen.fill(black)
 
    # Process the events in the game
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit_program = True

    if action == 0:
        player.rect.x -= SPEED*2  
    elif action == 2:
        player.rect.x += SPEED*2
        
    
    # Update the player and ball positions
    player.update()
    if ball.update():
        reward = PENALTY

    # See if the ball hits the player paddle
    if pygame.sprite.spritecollide(player, balls, False):
        reward = REWARD
        # The 'diff' lets you try to bounce the ball left or right
        # depending where on the paddle you hit it
        diff = (player.rect.x + player.width/2) - (ball.rect.x+ball.width/2)
 
        # Set the ball's y position in case
        # we hit the ball on the edge of the paddle
        ball.y = screen.get_height() - player.rect.height - ball.rect.height - SPEED
        ball.bounce(diff)

    # Check for collisions between the ball and the blocks
    deadblocks = pygame.sprite.spritecollide(ball, blocks, True)
 
    # If we actually hit a block, bounce the ball
    if len(deadblocks) > 0:
        ball.bounce(0)
    
    return reward


# initialize model
model = Model()
target_model = Model()

step(1) 
step_counter = 0
episode_rewards = [0]
while True:
    step_counter += 1
    s_t1 = np.array([
        player.rect.x, 
        ball.x,
        ball.y,
    ])

    # Peceive
    Q_t1 = model.perceive(s_t1.reshape(1,INPUT_SIZE))

    # Act, Reward
    if np.random.random() > epsilon:  
        a = np.argmax(Q_t1)
    else:
        a = np.random.randint(0, OUTPUT_SIZE)
    r = step(a)
    if r != 0:
        episode_rewards.append(r)
    s_t2 = np.array([
        player.rect.x, 
        ball.x,
        ball.y,
    ])

    # Peceive'               
    Q_t2 = target_model.perceive(s_t2.reshape(1,INPUT_SIZE))
    max_value_t2 = np.max(Q_t2)

    # Learn  
    # if r != 0: print(f"Q: {Q_t1}")
    Q_t1_train = np.copy(Q_t1)
    Q_t1_train[a] = r + GAMMA*max_value_t2
    value_t1 = Q_t1[a]
    # if r != 0: print(f"Q: {Q_t1}")
    # if r != 0: print(weights_prefit)
    # if r != 0: print(f"W Pre : {((output_layer.get_weights()[0].sum(axis=0) + output_layer.get_weights()[1]))}")

    # Back-propagate   
    e_softmax = (r + GAMMA*max_value_t2 - value_t1) * model.sigmoid_gradient(Q_t1[a])
    model.w_2_3[:,a] += ALPHA * e_softmax * model.l_2.reshape(HIDDEN_SIZE,)
    model.w_2_3_b[a] += ALPHA * e_softmax

    error_l_2 = e_softmax * model.w_2_3[:,a].reshape(HIDDEN_SIZE,1)
    error_l_2 = error_l_2 * model.sigmoid_gradient(model.l_2)
    model.w_1_2 += ALPHA * (error_l_2.reshape(HIDDEN_SIZE,1).dot(model.input_gradient(s_t1).reshape(INPUT_SIZE,1).T)).T
    model.w_1_2_b += (ALPHA * error_l_2).reshape(HIDDEN_SIZE,)

    # import ipdb; ipdb.set_trace()
    # if r != 0: print(f"W Post: {model.w_2_3.sum(axis=0) + model.w_2_3_b }")
    # if r != 0: print(f"Diff  : {model.w_2_3.sum(axis=0) + model.w_2_3_b - theta_sum}\n")

    s_t1 = s_t2

    if step_counter and not step_counter % SHOW_EVERY:
        target_model.w_1_2 = model.w_1_2
        target_model.w_1_2_b = model.w_1_2_b
        target_model.w_2_3 = model.w_2_3
        target_model.w_2_3_b = model.w_2_3_b

    # Display
    if step_counter and not step_counter % SHOW_EVERY:
        print(step_counter, sum(episode_rewards) / len(episode_rewards), len(episode_rewards))
        episode_rewards = [0]
    # Draw Environment
    allsprites.draw(screen) 
    # Flip the screen and show what we've drawn
    pygame.display.flip()

    display = np.zeros((3, len(Q_t1), 3), dtype=np.uint8)            
    min_q = np.min(Q_t1)
    max_q = np.max(Q_t1)

    # output = hidden_layers(s_t1.reshape(1,INPUT_SIZE))[0][0]
    # min_q = np.min(output)
    # max_q = np.max(output)
    left = ((Q_t1[0] - min_q)*255)/(max_q - min_q)  if max_q != min_q else 0
    nothing = ((Q_t1[1] - min_q)*255)/(max_q - min_q)  if max_q != min_q else 0
    right = ((Q_t1[2] - min_q)*255)/(max_q - min_q) if max_q != min_q else 0
    display[0][0] = (left, 0, 0)
    display[0][1] = (0, nothing, 0)
    display[0][2] = (0, 0, right)

    theta_sum = model.w_2_3.sum(axis=0) + model.w_2_3_b    

    display[2][0] = ((theta_sum[0] - theta_sum.min())*255 / (theta_sum.max() - theta_sum.min()), 0, 0)
    display[2][1] = (0, (theta_sum[1] - theta_sum.min())*255 / (theta_sum.max() - theta_sum.min()), 0)
    display[2][2] = (0, 0, (theta_sum[2] - theta_sum.min())*255 / (theta_sum.max() - theta_sum.min()))

    img = Image.fromarray(display, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
    img = img.resize((300, 50), Image.NONE)  # resizing so we can see our agent in all its glory.
    cv2.imshow("Q", np.array(img))  # show it!
 
pygame.quit()