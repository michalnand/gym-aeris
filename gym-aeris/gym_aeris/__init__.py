from gym.envs.registration import register

register(
    id='basic-v0',
    entry_point='gym_aeris.envs:BasicEnv',
)

register(
    id='goal-v0',
    entry_point='gym_aeris.envs:GoalEnv',
)
