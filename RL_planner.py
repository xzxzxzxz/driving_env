from config_ppo import Config_PPO
from run_ppo import Run_PPO
import driving


def main(sim_steps):
    # get the expert ready
    config = Config_PPO(log_name='model')
    expert = Run_PPO(config)
    expert.restore('model70.ckpt')

    # get the sim_env ready
    env = driving.LaneKeepingChanging()
    ob = env.reset()

    R = 0
    for i in range(sim_steps):
        ac = expert.obs_to_dyn_act(ob)
        ob, r, done = env.step(ac, i)
        R += r
        if done:
            print(i)
            break

    print(R)
    env.render_traj_history()


if __name__ == '__main__':
    main(1000)