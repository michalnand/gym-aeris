from gym.envs.registration import register

register(
    id='TargetNavigate-v0',
    entry_point='gym_aeris.envs:TargetNavigateEnv'
)

register(
    id='AvoidHazards-v0',
    entry_point='gym_aeris.envs:AvoidHazardsEnv'
)

register(
    id='AvoidFragiles-v0',
    entry_point='gym_aeris.envs:AvoidFragilesEnv'
)

register(
    id='FoodGathering-v0',
    entry_point='gym_aeris.envs:FoodGatheringEnv'
) 

register(
    id='FoodGatheringAdvanced-v0',
    entry_point='gym_aeris.envs:FoodGatheringAdvancedEnv'
)


register(
    id='SwarmFoodGathering-v0',
    entry_point='gym_aeris.envs:SwarmFoodGatheringEnv'
)


register(
    id='GridTargetSearchAEnv-v0',
    entry_point='gym_aeris.envs:GridTargetSearchAEnv'
)

register(
    id='GridTargetSearchBEnv-v0',
    entry_point='gym_aeris.envs:GridTargetSearchBEnv'
)



register(
    id='GridTargetSearchADiscreteEnv-v0',
    entry_point='gym_aeris.envs:GridTargetSearchADiscreteEnv'
)

register(
    id='GridTargetSearchBDiscreteEnv-v0',
    entry_point='gym_aeris.envs:GridTargetSearchBDiscreteEnv'
)
