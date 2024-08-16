
import gymnasium as gym

def make_env(args):
    """
    Create and return an instance of the specified environment.

    Args:
        args (object): An object containing the environment name.

    Returns:
        object: An instance of the specified environment.

    """
    env = gym.make(args.env.name)
    return env