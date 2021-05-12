from .env import PendulumEnv

def make_env(args):
    return PendulumEnv()