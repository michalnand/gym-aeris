import numpy



class AgentRandom:
    def __init__(self, env):
        self.env            = env
        self.actions_count  = self.env.action_space.shape[0]

        self.iterations = 0

    def main(self, verbose=False):
        self.iterations+= 1

        action = 2.0*numpy.random.randn(self.actions_count)
        action = numpy.tanh(action)

        state, reward, done, info = self.env.step(action)

        if verbose:
            print("action      = ", action)
            print("state shape = ", state.shape)
            print("reward      = ", reward)
            print("done        = ", done)
            print("state = \n", state)
            print("\n\n\n")
        
        if done:
            state = self.env.reset()

        return reward, done
    
   
