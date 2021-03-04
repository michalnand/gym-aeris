import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy
import torch

import cv2



class SwarmFoodGatheringEnv(gym.Env):

    def __init__(self, envs_count = 16, robots_count = 256, foods_count = 1024):
        gym.Env.__init__(self) 
        
        self.device       = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.dt                     = 0.01

        self.envs_count             = envs_count
        self.robots_count           = robots_count
        self.foods_count            = foods_count
        
        self.internal_state_size    = 4
        self.closest_count          = 4

        self.actions_shape          = (self.envs_count*self.robots_count*(2 + self.internal_state_size), )
        self.observation_shape      = (self.envs_count, self.robots_count, self.closest_count*2 + (self.closest_count+1)*(2 + self.internal_state_size))
        
        self.action_space           = spaces.Box(low=-1.0, high=1.0, shape=self.actions_shape, dtype=numpy.float32)
        self.observation_space      = spaces.Box(low=-1.0, high=1.0, shape=self.observation_shape, dtype=numpy.float32)

        self.reset()

    def reset(self, env_id = -1):
        if env_id == -1:
            self.steps              = numpy.zeros(self.envs_count)
            self.robots_positions   = 0.95*(2.0*torch.rand((self.envs_count, self.robots_count, 2)) - 1.0).to(self.device)
            self.foods_positions    = 0.95*(2.0*torch.rand((self.envs_count, self.foods_count, 2))  - 1.0).to(self.device)
            self.internal_state     = (torch.zeros((self.envs_count, self.robots_count, self.internal_state_size))).to(self.device)
        else:
            self.steps[env_id]              = 0
            self.robots_positions[env_id]   = 0.95*(2.0*torch.rand((self.robots_count, 2)) - 1.0).to(self.device)
            self.foods_positions[env_id]    = 0.95*(2.0*torch.rand((self.foods_count, 2))  - 1.0).to(self.device)
            self.internal_state[env_id]     = torch.zeros((self.robots_count, self.internal_state_size)).to(self.device)

        self._update_distances()
        return self._update_observation()
        
    def step(self, actions):
        self.steps+= 1

        action_  = actions.reshape((self.envs_count, self.robots_count, 2 + self.internal_state_size))
        action_  = torch.from_numpy(action_).to(self.device)

        velocity = action_[:,:,0:2]

        self.robots_positions+= velocity*self.dt
        self.robots_positions = torch.clamp(self.robots_positions, -1.0, 1.0)
        
        self.internal_state = action_[:, :, 2:2+self.internal_state_size].clone()

        self._update_distances()

        rewards = self._update_rewards()
        dones   = self._update_dones()
   
        return self._update_observation(), rewards, dones, None

    def render(self, env_id = 0):
        
        robots_positions_np = self.robots_positions[env_id].detach().to("cpu").numpy()
        foods_positions_np  = self.foods_positions[env_id].detach().to("cpu").numpy()
        
        closest_robots_positions_np = self.closest_robots_positions[env_id].detach().to("cpu").numpy()
        closest_foods_positions_np  = self.closest_foods_positions[env_id].detach().to("cpu").numpy()

        height = 800
        width  = 800

        image = numpy.zeros((height, width,3), numpy.uint8)


        for j in range(len(robots_positions_np)):
            sx = int(width*(robots_positions_np[j][0] + 1.0)/2.0)
            sy = int(width*(robots_positions_np[j][1] + 1.0)/2.0)

            for i in range(self.closest_count+1):
                dx = int(width*(closest_robots_positions_np[j][i][0] + 1.0)/2.0)
                dy = int(width*(closest_robots_positions_np[j][i][1] + 1.0)/2.0)

                image = cv2.line(image, (sx, sy), (dx, dy), (120, 64, 64), 1) 

            for i in range(self.closest_count):
                dx = int(width*(closest_foods_positions_np[j][i][0] + 1.0)/2.0)
                dy = int(width*(closest_foods_positions_np[j][i][1] + 1.0)/2.0)

                image = cv2.line(image, (sx, sy), (dx, dy), (64, 64, 120), 1) 
        
        for j in range(len(robots_positions_np)):
            cx = int(width*(robots_positions_np[j][0] + 1.0)/2.0)
            cy = int(width*(robots_positions_np[j][1] + 1.0)/2.0)

            image = cv2.circle(image, (cx, cy), 1, (255, 0, 0), 2)
 
        for j in range(len(foods_positions_np)):
            cx = int(width*(foods_positions_np[j][0] + 1.0)/2.0)
            cy = int(width*(foods_positions_np[j][1] + 1.0)/2.0)

            image = cv2.circle(image, (cx, cy), 1, (0, 0, 255), 2)

        window_name = "Aeris - " + self.__class__.__name__ + " " + str(env_id)
        
        cv2.imshow(window_name, image) 
        cv2.waitKey(1)

    def close(self):
        pass

    def _update_distances(self):
        self.robots_robots_distances = torch.cdist(self.robots_positions, self.robots_positions)
        self.robots_foods_distances  = torch.cdist(self.robots_positions, self.foods_positions)

        self.closest_robots_indices  = self.robots_robots_distances.argsort(dim=2, descending=False)[:, :, 0:self.closest_count+1]
        self.closest_foods_indices   = self.robots_foods_distances.argsort(dim=2, descending=False)[:,  :, 0:self.closest_count]
    
        self.closest_robots_positions  = torch.zeros((self.envs_count, self.robots_count, self.closest_count+1, 2), device=self.device)
        self.closest_foods_positions   = torch.zeros((self.envs_count, self.robots_count, self.closest_count, 2), device=self.device)
        self.closest_internal_states   = torch.zeros((self.envs_count, self.robots_count, self.closest_count+1, self.internal_state_size), device=self.device)

        #TODO optimize this loop
        for e in range(self.envs_count):
            self.closest_robots_positions[e] = self.robots_positions[e][self.closest_robots_indices[e]]
            self.closest_foods_positions[e]  = self.foods_positions[e][self.closest_foods_indices[e]]
            self.closest_internal_states[e]  = self.internal_state[e][self.closest_robots_indices[e]]
        
    def _update_observation(self):

        closest_robots_positions_ = self.closest_robots_positions.reshape((self.envs_count, self.robots_count, (self.closest_count+1)*2))
        closest_foods_positions_  = self.closest_foods_positions.reshape((self.envs_count, self.robots_count, self.closest_count*2))
        closest_internal_states_  = self.closest_internal_states.reshape((self.envs_count, self.robots_count, (self.closest_count+1)*self.internal_state_size))

        obs = torch.cat([closest_robots_positions_, closest_foods_positions_, closest_internal_states_], dim=2)

        return obs.detach().to("cpu").numpy()

    #TODO
    def _update_rewards(self):
        rewards = numpy.zeros(self.envs_count)
        return rewards

    def _update_dones(self):
        dones = numpy.zeros(self.envs_count, dtype=bool)
 
        for i in range(self.envs_count):
            if self.steps[i] >= 1000:
                dones[i] = True

        return dones
