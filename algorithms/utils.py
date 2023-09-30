def __init__():
    pass


def evaluate_meta_policy(policy, env, max_steps=200):

    state = env.reset()

    print(state)

    acc_reward = 0
    success = False

    for _ in range(max_steps):

        action = policy[state]
        state, reward, done, _ = env.step(action)
        acc_reward += reward

        if done:
            success = True
            break


    return success, acc_reward