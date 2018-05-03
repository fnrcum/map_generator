import random
import noise
from renderer import *
import numpy as np
from scipy.misc import toimage
import math


class GenerateMap:

    def __init__(self, size=(50, 50), color_range=10, color_perlin_scale=0.025, scale=350, octaves=6, persistance=0.6,
                 lacunarity=2.0):
        self.scale = scale
        self.octaves = octaves
        self.persistance = persistance
        self.lacunarity = lacunarity

        self.mapSize = size  # size in pixels
        self.mapCenter = (self.mapSize[0] / 2, self.mapSize[1] / 2)

        self.heightMap = np.zeros(self.mapSize)
        # self.colorMap = [[Color() for j in range(self.mapSize)] for i in range(self.mapSize)]

        self.randomColorRange = color_range
        self.colorPerlinScale = color_perlin_scale

        self.lightblue = [0, 191, 255]
        self.blue = [65, 105, 225]
        self.darkblue = [0, 0, 139]
        self.green = [34, 139, 34]
        self.darkgreen = [0, 100, 0]
        self.sandy = [210, 180, 140]
        self.beach = [238, 214, 175]
        self.snow = [255, 250, 250]
        self.mountain = [139, 137, 137]

        self.threshold = -0.01

    def return_initial_blank_map(self):
        return self.heightMap

    def get_map_corners(self):
        nw = self.heightMap[0][0]
        ne = self.heightMap[0][len(self.heightMap[0])-1]
        sw = self.heightMap[len(self.heightMap)-1][0]
        se = self.heightMap[len(self.heightMap)-1][len(self.heightMap[0])-1]
        return nw, ne, sw, se

    def get_map_start_position(self, start_position):
        pass

    def generate_map(self, map_type=""):
        random_nr = random.randint(0, self.mapSize[0])
        for i in range(self.mapSize[0]):
            for j in range(self.mapSize[1]):
                self.heightMap[i][j] = noise.pnoise3(i / self.scale, j / self.scale, random_nr, octaves=self.octaves,
                                                     persistence=self.persistance, lacunarity=self.lacunarity,
                                                     repeatx=1024, repeaty=1024, base=0)
        print("monochrome map created")
        if map_type is "island":
            gradient = self.create_circular_gradient(self.heightMap)
            color_map = self.add_color(gradient)
        else:
            color_map = self.add_color(self.heightMap)
        return color_map

    def add_color(self, world):
        color_world = np.zeros(world.shape + (3,))
        # print(color_world)
        for i in range(self.mapSize[0]):
            for j in range(self.mapSize[1]):
                if world[i][j] < self.threshold + 0.02:
                    color_world[i][j] = self.darkblue
                elif world[i][j] < self.threshold + 0.03:
                    color_world[i][j] = self.blue
                elif world[i][j] < self.threshold + 0.058:
                    color_world[i][j] = self.sandy
                elif world[i][j] < self.threshold + 0.1:
                    color_world[i][j] = self.beach
                elif world[i][j] < self.threshold + 0.25:
                    color_world[i][j] = self.green
                elif world[i][j] < self.threshold + 0.6:
                    color_world[i][j] = self.darkgreen
                elif world[i][j] < self.threshold + 0.7:
                    color_world[i][j] = self.mountain
                elif world[i][j] < self.threshold + 1.0:
                    color_world[i][j] = self.snow
        print("color map created")
        return color_world

    def create_circular_gradient(self, world):
        center_x, center_y = self.mapSize[1] // 2, self.mapSize[0] // 2
        circle_grad = np.zeros_like(world)

        for y in range(world.shape[0]):
            for x in range(world.shape[1]):
                distx = abs(x - center_x)
                disty = abs(y - center_y)
                dist = math.sqrt(distx * distx + disty * disty)
                circle_grad[y][x] = dist

        # get it between -1 and 1
        max_grad = np.max(circle_grad)
        circle_grad = circle_grad / max_grad
        circle_grad -= 0.5
        circle_grad *= 2.0
        circle_grad = -circle_grad

        # shrink gradient
        for y in range(world.shape[0]):
            for x in range(world.shape[1]):
                if circle_grad[y][x] > 0:
                    circle_grad[y][x] *= 20

        # get it between 0 and 1
        max_grad = np.max(circle_grad)
        circle_grad = circle_grad / max_grad
        grad_world = self.apply_gradient_noise(world, circle_grad)
        return grad_world

    def apply_gradient_noise(self, world, c_grad):
        world_noise = np.zeros_like(world)

        for i in range(self.mapSize[0]):
            for j in range(self.mapSize[1]):
                world_noise[i][j] = (world[i][j] * c_grad[i][j])
                if world_noise[i][j] > 0:
                    world_noise[i][j] *= 20

        # get it between 0 and 1
        max_grad = np.max(world_noise)
        world_noise = world_noise / max_grad
        return world_noise


map_data = GenerateMap((1024, 1024))
print("Generator initiallized")
mono_map = map_data.generate_map("island")
print("map generated")
toimage(mono_map).show()
print("map displayed")