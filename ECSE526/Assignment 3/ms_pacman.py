import cv2
import numpy as np
from collections import deque
import sys
import random
from ale_py import ALEInterface
import pickle
import random
import argparse

class PM(object):
    """
    class of MsPacman for intializing the ale interface and showing the game progress 
    It works for changing the position of pacman, reading the ram values at different frames of the game
    according to ram values it gives the values of the x,y coordiantes of the different ghosts, pacman and fruit
    it also takes input of actions and changes the position of the pacman according to the map of the game.

    """

    def __init__(self, seed, display):
        
        self._ale = ALEInterface()

        if seed is None:
            seed = random.randint(0, 255)
        self._ale.setInt("random_seed", seed)

        if display:
            if sys.platform == "darwin":
                # Use PyGame in macOS.
                import pygame
                pygame.init()

                # Sound doesn't work on macOS.
                self._ale.setBool("sound", False)
            elif sys.platform.startswith("linux"):
                self._ale.setBool("sound", True)

            self._ale.setBool("display_screen", True)

        self._ale.loadROM("MS_PACMAN.BIN")

        self._reward = 0
        self._raw_ms_pacman_position = (0, 0)

        self.__screen = self._ale.getScreen()
        self.__ram = self._ale.getRAM()

        self._lives = self._ale.lives()

        self._update_state()

        self._go_to((94, 98), 3)

    @property
    def lives(self):
        return self._lives

    @property
    def reward(self):
        return self._reward

    @property
    def map(self):
        return self._map

    @property
    def sliced_map(self): #sliced map is useful for our pacman position and checking the possible action of the pacman
        return self._sliced_map

    @property
    def ms_pacman_position(self): #pacman position taken from the ram and in the it converts into x,y cordinates of the system
        return self._ms_pacman_position

    @property
    def fruit(self):
        return self._fruit

    @property
    def ghosts(self):
        return self._ghosts

    def available_actions(self):
        actions = []  #take only four actions which are possible in pacman 

        for action, move in [
            (2, (-1, 0)),  # up
            (3, (0, 1)),   # right
            (4, (0, -1)),  # left
            (5, (1, 0))    # down
        ]:
            new_pos = self.get_next_position(self._ms_pacman_position, move)
            if 0 <= new_pos[0] < Map_matrix.HEIGHT:
                if self._map.map[new_pos] != Game_objects.WALL:
                    actions.append(action)
        return actions

    def action_to_move(self, action):  #converting actions to start from zero so that I can interate through for loop
        return [(-1, 0), (0, 1), (0, -1), (1, 0)][action - 2]

    def get_next_position(self, curr_position, move):
        new_pos = (
            curr_position[0] + move[0],
            curr_position[1] + move[1]
        )
        if new_pos[1] < 0:
            new_pos = (new_pos[0], new_pos[1] + Map_matrix.WIDTH)
        elif new_pos[1] >= Map_matrix.WIDTH:
            new_pos = (new_pos[0], new_pos[1] - Map_matrix.WIDTH)
        return new_pos

    def act(self, action):
        
        m = self.action_to_move(action)
        next_pos = self.get_next_position(self._ms_pacman_position, m)
        old_reward = self._reward
        old_lives = self._lives

        expected_reward = Game_objects.to_reward(self._map.map[next_pos])

        MAX_ACTION_COUNT = 20
        for _ in range(MAX_ACTION_COUNT):
            if expected_reward <= 0:
                if self._ms_pacman_position == next_pos:
                    break
            elif self._reward != old_reward:
                break

            if self.game_over() or self._lives < old_lives:
                return Game_objects.to_reward(Game_objects.BAD_GHOST)

            self._reward += self._ale.act(action)
            self._update_state()

        self._update_map()
        return self._reward - old_reward

    def _go_to(self, raw_pos, action):
        
        while (abs(self._raw_ms_pacman_position[0] - raw_pos[0]) > 1 or
                abs(self._raw_ms_pacman_position[1] - raw_pos[1]) > 1):
            self._ale.act(action)
            self._update_state()
        self._update_map()

    def game_over(self):
        
        return self._ale.game_over()

    def reset_game(self):
       
        self._reward = 0
        return self._ale.reset_game()

    def _to_map_position(self, pos):
        #for converting ram coordinates into map coordinates of i,j which I can iterate through
        x, y = pos
        i = round((y - 2) / 12.0)
        if x < 83:
            j = round((x - 18) / 8.0 + 1)
        elif 93 < x < 169:
            j = round((x - 22) / 8.0 + 1)
        elif x > 169:
            j = 0
        elif x < 88:
            j = 9
        else:
            j = 10
        return i, j

    def _to_raw_position(self, pos):
        i, j = pos
        y = i * 12 + 2
        if j == 0:
            x = 12
        elif j <= 9:
            x = (j - 1) * 8 + 18
        else:
            x = (j - 1) * 8 + 22
        return x, y

    def _update_state(self):
        #these are ram values from which we can identify the position of the pacman and different ghosts
        self._ale.getRAM(self.__ram)
        new_ms_pacman_position = (int(self.__ram[10]), int(self.__ram[16]))
        new_ghosts_ram = [
            ((int(self.__ram[6]), int(self.__ram[12])), int(self.__ram[1])),
            ((int(self.__ram[7]), int(self.__ram[13])), int(self.__ram[2])),
            ((int(self.__ram[8]), int(self.__ram[14])), int(self.__ram[3])),
            ((int(self.__ram[9]), int(self.__ram[15])), int(self.__ram[4]))
        ]
        fruit = (int(self.__ram[11]), int(self.__ram[17])), int(self.__ram[5])
        self._fruit = Fruit.to_ram(self._to_map_position(fruit[0]), fruit[1],
                                     fruit[0][0] != 0)

        self._raw_ms_pacman_position = new_ms_pacman_position
        self._ms_pacman_position = self._to_map_position(
            new_ms_pacman_position)
        self._ghosts = [
            Ghost.to_ram(self._to_map_position(pos), ram)
            for pos, ram in new_ghosts_ram
        ]

        
        self._lives = self._ale.lives()

    def _update_map(self):
        
        self._ale.getScreen(self.__screen)
        self._map = Map_matrix(self.__screen.reshape(210, 160))
        self._blank_map = Map_matrix.from_map(self._map.map.copy())
        self._map.map[self._ms_pacman_position] = Game_objects.MS_PACMAN
        if self._fruit.exists:
            self._map.map[self._fruit.position] = Game_objects.FRUIT
        for ghost in self._ghosts:
            if ghost.state == Ghost.GOOD:
                self._map.map[ghost.position] = Game_objects.GOOD_GHOST
            elif ghost.state == Ghost.BAD:
                self._map.map[ghost.position] = Game_objects.BAD_GHOST
        self._sliced_map = Sliced_Map_matrix(self._map,
                                         self._ms_pacman_position)

class ghost_fruit(object):

    """"
    This is a class for defining ghost and fruit which are dynamic in game
    """
    def __init__(self, position, direction):
        self.position = position
        self.direction = direction

    @classmethod
    def to_ram(cls, position, ram):
        direction = cls.directions(ram)
        return ghost_fruit(position, direction)

    @classmethod
    def directions(cls, ram):  #This function returns the direction of the pacman, which are up, down, right, left
        possible = ram & 3
        second_stage = \
            [-1, 0] if possible  == 0 else \
            [0, 1] if possible  == 1 else \
            [1, 0] if possible  == 2 else \
            [0, -1]
        return second_stage


class Ghost(ghost_fruit):

    """There are two types of ghost in the game, after eating the power up,
    pacman can eat ghosts so, for that I've decided to change the state of the ghosts
    after pacman get the power up points."""

    BAD = 0
    GOOD = 1

    def __init__(self, position, direction, state):
        super(Ghost, self).__init__(position, direction)
        self.state = state

    @classmethod
    def to_ram(cls, position, ram):
        direction = cls.directions(ram)
        edible = (ram >> 7) & 1
        state = \
            cls.GOOD if edible == 1 else \
            cls.BAD

        return cls(position, direction, state)


class Fruit(ghost_fruit):

    def __init__(self, position, direction, exists):
        super(Fruit, self).__init__(position, direction)
        self.exists = exists

    @classmethod
    def to_ram(cls, position, ram, exists):
        direction = cls.directions(ram)
        return cls(position, direction, exists)


class Game_objects(object):

    """
    Convert game maps into different types of objects, which constitute the different parts of the game.
    """
    EMPTY = 0
    WALL = 1
    PELLET = 2
    POWER_UP = 3
    GOOD_GHOST = 4
    BAD_GHOST = 5
    FRUIT = 6
    MS_PACMAN = 7

    @classmethod
    def to_reward(cls, classification):
        """Converts a GameMapObject to a reward.

        Args:
            classification: GameMapObject.

        Returns:
            Reward.
        """
        reward = 0
        if classification == Game_objects.WALL:
            reward = 0
        elif classification == Game_objects.PELLET:
            reward = 10
        elif classification == Game_objects.POWER_UP:
            reward = 200
        elif classification == Game_objects.GOOD_GHOST:
            reward = 200
        elif classification == Game_objects.BAD_GHOST:
            reward = -100
        elif classification == Game_objects.FRUIT:
            reward = 100
        elif classification == Game_objects.MS_PACMAN:
            reward = 0
        return reward

    @classmethod
    def to_color(cls, classification):
        """
        This converts objects into RGB, different objects of the game are differentialted into different parts based on color 
        of the object, which have unique RGB values and based on that we can create a map of different things.
        """
        color = [136, 28, 0]  
        if classification == Game_objects.WALL:
            color = [111, 111, 228]  
        elif classification == Game_objects.PELLET:
            color = [255, 255, 255]  
        elif classification == Game_objects.POWER_UP:
            color = [255, 255, 0]  
        elif classification == Game_objects.GOOD_GHOST:
            color = [0, 255, 0]  
        elif classification == Game_objects.BAD_GHOST:
            color = [0, 0, 255]  
        elif classification == Game_objects.FRUIT:
            color = [255, 0, 255]  
        elif classification == Game_objects.MS_PACMAN:
            color = [0, 255, 255] 
        return color

class Map_matrix(object):
    """
    consturcts a matrix of an given image for better understanding of the game.
    """
    WIDTH = 20
    HEIGHT = 14

    PRIMARY_COLOR = 74

    def __init__(self, image, game_map=None):
        if game_map is not None:
            self._map = game_map
            return

        self._image = image[2:170]

        height, width = self._image.shape
        self._width_step = width / self.WIDTH
        self._height_step = height / self.HEIGHT

        self._classify()

    @classmethod
    def from_map(cls, game_map):
        return cls(None, game_map)

    @property
    def map(self):
        return self._map

    def _classify_histogram(self, histogram):
        #for the classification of the given image into wall, pellete, power_up, empty
        primary_count = histogram[self.PRIMARY_COLOR]
        total_count = self._width_step * self._height_step
        primary_ratio = primary_count / total_count

        
        if primary_ratio >= 0.40:
            return Game_objects.WALL

       
        if primary_ratio >= 0.25:
            return Game_objects.POWER_UP

       
        if primary_ratio >= 0.05:
            return Game_objects.PELLET

        
        return Game_objects.EMPTY

    def _classify_partition(self, partition):
        
        histogram = cv2.calcHist([partition], [0], None, [256], [0, 256])
        return self._classify_histogram(histogram)

    def _classify(self):
        self._map = np.zeros((self.HEIGHT, self.WIDTH), dtype=np.uint8)
        for i in range(self.WIDTH):
            for j in range(self.HEIGHT):
                curr_width = int(i * self._width_step)
                curr_height = int(j * self._height_step)

                next_width = int(curr_width + self._width_step)
                next_height = int(curr_height + self._height_step)

                partition = self._image[curr_height:next_height,
                                        curr_width:next_width]

                self._map[j, i] = self._classify_partition(partition)

    def to_image(self):
        image = np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        for i in range(self.WIDTH):
            for j in range(self.HEIGHT):
                classification = self._map[j, i]
                image[j, i] = Game_objects.to_color(classification)

        upscaled_image = cv2.resize(image, (160, 168),
                                    interpolation=cv2.INTER_NEAREST)
        return upscaled_image


class Sliced_Map_matrix(object):
    """
    convert the whole image into smaller image centered around the pacman
    """
    RADIUS = 2

    def __init__(self, game_map, ms_pacman_position):
        
        self._map = get_slice(game_map, ms_pacman_position, self.RADIUS)

    @property
    def map(self):
        return self._map

    def to_image(self):

        height, width = self._map.shape
        image = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(width):
            for j in range(height):
                classification = self._map[j, i]
                image[j, i] = Game_objects.to_color(classification)

        upscaled_image = cv2.resize(image, (100, 100),
                                    interpolation=cv2.INTER_NEAREST)
        return upscaled_image

def get_slice(game_map, pac_pos, radius):
    """
    This function converts the whole given image into readable matrix for the user 
    specified by the values given in the prior classes.
    """
    min_i = pac_pos[0] - radius
    max_i = pac_pos[0] + radius + 1
    min_j = pac_pos[1] - radius
    max_j = pac_pos[1] + radius + 1

    vertical_slice = slice(max(min_i, 0), min(max_i, game_map.HEIGHT))
    horizontal_slice = slice(max(min_j, 0), min(max_j, game_map.WIDTH))
    map_slice = game_map.map[vertical_slice, horizontal_slice]

    # Concatenate the opposite side of the board for a horizontal overflow.
    if min_j < 0:
        map_slice = np.hstack((
            game_map.map[vertical_slice, min_j - 1:-1],
            map_slice
        ))
    elif max_j >= game_map.WIDTH:
        map_slice = np.hstack((
            map_slice,
            game_map.map[vertical_slice, 0:max_j - game_map.WIDTH]
        ))

    # Concatenate walls for any vertical overflow.
    height, width = map_slice.shape
    if min_i < 0:
        map_slice = np.vstack((
            np.ones((abs(min_i), width), dtype=np.uint8),
            map_slice
        ))
    elif max_i >= game_map.HEIGHT:
        map_slice = np.vstack((
            map_slice,
            np.ones((max_i - game_map.HEIGHT, width), dtype=np.uint8)
        ))

    return hide_cells_behind_wall(map_slice)


def hide_cells_behind_wall(map_slice):
    
    height, width = map_slice.shape
    center = int((height - 1) / 2)

    shadowed_map = np.ones((height, width))
    visited = np.zeros((height, width))
    neighbor_queue = deque()
    neighbor_queue.append((center, center))

    while neighbor_queue:
        cell = neighbor_queue.popleft()
        visited[cell] = 1
        shadowed_map[cell] = map_slice[cell]
        if map_slice[cell] == Game_objects.WALL:
            continue
        for neighbor in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            i = cell[0] + neighbor[0]
            j = cell[1] + neighbor[1]
            if 0 <= i < height and 0 <= j < width and not visited[(i, j)]:
                neighbor_queue.append((i, j))

    return shadowed_map

def get_next_state(game, action):
    move = game.action_to_move(action)
    new_pos = game.get_next_position(game.ms_pacman_position, move)
    game_map = Map_matrix.from_map(game._blank_map.map.copy())
    if game.fruit.exists:
        game_map.map[game.fruit.position] = Game_objects.FRUIT
    for ghost in game.ghosts:
        new_ghost_position = game.get_next_position(ghost.position,
                                                    ghost.direction)
        if (not 0 < new_ghost_position[0] < game_map.HEIGHT or
                not 0 < new_ghost_position[1] < game_map.WIDTH or
                ghost.position == new_pos):
            if ghost.state == Ghost.GOOD:
                game_map.map[ghost.position] = Game_objects.GOOD_GHOST
            elif ghost.state == Ghost.BAD:
                game_map.map[ghost.position] = Game_objects.BAD_GHOST
        elif game_map.map[new_ghost_position] == Game_objects.WALL:
            continue
        elif ghost.state == Ghost.GOOD:
            game_map.map[new_ghost_position] = Game_objects.GOOD_GHOST
        elif ghost.state == Ghost.BAD:
            game_map.map[new_ghost_position] = Game_objects.BAD_GHOST
    return get_slice(game_map, new_pos, 2)


class Generalization(object):

    def __init__(self, alpha=0.01, gamma=0.7):
        self.weights = [1] * 30
        self.greed = 0.25
        self.alpha = alpha
        self.gamma = gamma

    def _get_utility(self, state):
        state_rewards = self._get_state(state)
        utility = 0
        for i in range(len(state_rewards)):
            w_index = self._to_weight_index(i)
            utility += self.weights[w_index] * state_rewards[i]
        return utility

    def get_optimal_action(self, game):
        optimal_utility = float("-inf")
        optimal_actions = [0]  # noop.
        available_actions = game.available_actions()

        if random.random() <= self.greed:
            available_actions = [random.choice(available_actions)]

        for a in available_actions:
            next_state = get_next_state(game, a)
            utility = self._get_utility(next_state)
            if utility > optimal_utility:
                optimal_utility = utility
                optimal_actions = [a]
            elif utility == optimal_utility:
                optimal_actions.append(a)

        return (random.choice(optimal_actions), optimal_utility)


    def update_weights(self, prev_state, action, game, guess_utility, reward):
        self.greed = max(0.001, self.greed - 1e-5)
        curr_state = game.sliced_map.map.copy()
        curr_state[2, 2] = \
            prev_state[3, 2] if action == 2 else \
            prev_state[2, 1] if action == 3 else \
            prev_state[2, 3] if action == 4 else \
            prev_state[1, 2]

        state_rewards = self._get_state(curr_state)
        real_utility = reward + self.gamma * self.get_optimal_action(game)[1]
        error = 0.5 * (real_utility - guess_utility) 

        # print("Estimated utility: {}".format(guess_utility))
        # print("Actual utility: {}".format(real_utility))
        # print("Error: {}".format(error))

        for i in range(len(state_rewards)):
            self.weights[self._to_weight_index(i)] += \
                self.alpha * (real_utility - guess_utility) * \
                state_rewards[i] / self._to_weight_norm(i % 25)

    def _get_state(self, game_map):
        all_state = game_map.flatten()
        size = len(all_state)

        total_state = [0] * (5 * size)

        for i in range(size):
            classification = all_state[i]
            if classification == Game_objects.BAD_GHOST:
                total_state[i] = 1
            elif classification == Game_objects.GOOD_GHOST:
                total_state[i + size] = 1
            elif classification == Game_objects.PELLET:
                total_state[i + size * 2] = 1
            elif classification == Game_objects.POWER_UP:
                total_state[i + size * 3] = 1
            elif classification == Game_objects.FRUIT:
                total_state[i + size * 4] = 1

        return total_state

    def _to_weight_index(self, i):
        return self._to_weight_map(i % 25) + int(i / 25) * 6

    def _to_weight_map(self, i):
        return [
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1
        ][i]

    def _to_weight_norm(self, i):
        return [
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1
        ][i]

pacman = PM(None, True)
AI = Generalization(0.05)
for episode in range(1):
    while not pacman.game_over():
        prev_state = pacman.sliced_map.map
        optimal_a, expected_utility = AI.get_optimal_action(pacman)
        reward = pacman.act(optimal_a)
        AI.update_weights(prev_state, optimal_a, pacman,expected_utility, reward)
    pacman.reset_game()
