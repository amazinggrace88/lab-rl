"""
PG(Policy Gradient): 최대 보상을 받을 수 있도록 정책(policy)을 변화시킴.
신경망의 파라미터들을 즉각적으로 업데이트하는 대신에,
여러 에피소드를 진행시킨 후, 더 좋은 결과를 준 action이 더 많은 확률로 나올 수
있도록 변경함.
"""
import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from rl.ex04_cartpole_n_neuralnet import render_policy_net


def play_one_step(env, obs, model, loss_fn):
    """주어진 신경망 모델을 사용해서 게임을 1 step 진행.
    1 step 진행 후 바뀐 observation, reward, done, gradients들을 리턴.
    """
    with tf.GradientTape() as tape:
        left_prob = model(obs[np.newaxis])  # 1D -> 2D : 0~1사이 값들로 만들었다.
        action = (tf.random.uniform([1, 1]) > left_prob)  # boolean(T, F)
        y_target = tf.constant([[1.0]]) - tf.cast(action, tf.float32)
        # y_target: 신경망의 선택이 최적이라고_즉, 옳다고(optimal) 가정하는 것! (정책을 모르기 때문에 random 하게 가정)
        loss = tf.reduce_mean(loss_fn(y_target, left_prob))
    grads = tape.gradient(loss, model.trainable_variables)  # 신경망이 가지고 있는 w에 대한 gradient
    obs, reward, done, info = env.step(int(action[0, 0].numpy()))
    return obs, reward, done, grads


def play_multiple_episodes(env, n_episodes, max_steps, model, loss_fn):
    """여러번 에피소드를 플레이하는 함수.
    모든 보상들(reward)과 gradients 들을 리턴.
    """
    all_rewards = []  # 에피소드가 끝날 때마다 총 보상(reward)를 추가할 리스트
    all_grads = []  # 에피소드가 끝날 때마다 계산된 gradinet들을 추가할 리스트
    for episode in range(n_episodes):
        current_rewards = []  # 각 스텝마다 받은 보상을 추가할 리스트
        current_grads = []  # 각 스텝마다 계산된 gradient를 추가할 리스트
        obs = env.reset()  # 게임 환경 초기화
        for step in range(max_steps):
            obs, reward, done, grads = play_one_step(env, obs, model, loss_fn)
            current_rewards.append(reward)
            current_grads.append(grads)
            if done:
                break
        all_rewards.append(current_rewards)
        all_grads.append(current_grads)
    return all_rewards, all_grads


def discount_rewards(rewards, discount_rate):
    """
    gamma: 할인율. 0 <= gamma <= 1.
           미래의 보상을 현재 보상에 얼만큼 반영할 지를 결정하는 하이퍼 파라미터.
    R(t): 현재(t 시점)에서의 예상되는 미래 수익
    R(t) = r(t) + gamma * r(t+1) + gamma^2 * r(t+2) + gamma^3 * r(t+3) + ...
    gamma = 1 인 경우, 미래의 모든 수익을 동등하게 고려.
    gamma = 0 인 경우, 미래의 수익은 고려하지 않고, 단지 현재 수익만 고려.
    0 < gamma < 1 인 경우, 미래의 몇 단계까지만 중요하게 고려
    R(t) = r(t) + gamma * {r(t+1) + gamma * r(t+2) + gamma^2 * r(t+3) +...}
    R(t) = r(t) + gamma * R(t+1)
    """
    # 게임 진행 하고 나서, 미래에 얻을 수 있는 보상을 계산하기. (미래에 얻어질 보상들 = 지연보상 ~ 미래가치)
    # discount rate 를 몇 스텝까지 고려할 것인지 정한다.
    # (r)^x ~ 0.5(reward) : x는 몇 스텝까지 고려대상에 넣을건지가 할인율(r)에 의해 결정된다.
    discounted = np.array(rewards)
    for step in range(len(rewards) - 2, -1, -1):
        discounted[step] += discount_rate * discounted[step + 1]  # 현재 수익 = 현재 상태 + r * 미래 수익
    return discounted


def discount_normalize_rewards(all_rewards, discount_rate):
    #
    all_dc_rewards = [discount_rewards(rewards, discount_rate)
                      for rewards in all_rewards]
    # z = (x - mu) / sigma
    flat_rewards = np.concatenate(all_dc_rewards)  # 2D array -> 1D array : 2D array 가 직사각형이 아니기 때문에
    rewards_mean = flat_rewards.mean()
    rewards_std = flat_rewards.std()
    return [(x - rewards_mean) / rewards_std
            for x in all_dc_rewards]


if __name__ == '__main__':
    rewards = [10, 0, -50]
    discounted = discount_rewards(rewards, discount_rate=0.8)
    print(discounted)

    # ragged matrix : 끝이 왔다갔다 하는 .. 행렬
    all_rewards = [
        [10, 0, -50],
        [10, 20]
    ]
    dc_normalized = discount_normalize_rewards(all_rewards, 0.8)
    print(dc_normalized)

    # Policy Gradient에서 사용할 신경망 생성
    model = keras.Sequential()
    model.add(keras.layers.Dense(4, activation='elu', input_shape=(4,)))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    loss_fn = keras.losses.binary_crossentropy

    # 학습에 필요한 상수들
    n_iterations = 150  # 전체 반복 회수
    # 신경망 모델을 업데이트하기 전에 실행할 에피소드 회수
    n_episode_per_update = 10
    # 1 에피소드에서 실행할 최대 스텝
    max_steps = 200
    # 할인율: 각 스텝에서의 보상(reward)의 할인값을 계산하기 위해서
    discount_rate = 0.95

    env = gym.make('CartPole-v1')  # 게임 환경

    for iteration in range(n_iterations):
        all_rewards, all_grads = play_multiple_episodes(env,
                                                        n_episode_per_update,
                                                        max_steps,
                                                        model,
                                                        loss_fn)
        total_rewards = sum(map(sum, all_rewards))
        mean_rewards = total_rewards / n_episode_per_update
        print(f'Iteration #{iteration}: mean_rewards={mean_rewards}')
        all_final_rewards = discount_normalize_rewards(all_rewards, discount_rate)  # discount(보상을 discount), normalize
        all_mean_grads =[]
        for idx in range(len(model.trainable_variables)):
            mean_grads = tf.reduce_mean(
                # 보상이 고려가 된 가중치(중요) : 보상과 gradient 를 곱하여 weight 를 변경시킴
                [final_reward * all_grads[episode_index][step][idx]  # 두번째 for 문에서 가져온 final reward 를 all_grads 에 곱(3)
                 for episode_index, final_rewards in enumerate(all_final_rewards)  # index 번째 episode 의 reward 리스트 꺼냄(1)
                 for step, final_reward in enumerate(final_rewards)],  # reward 리스트에서 step 번째 final reward 를 꺼냄(2)
                axis=0
            )
            all_mean_grads.append(mean_grads)
        optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))

    env.close()

    render_policy_net(model, max_steps=1000)




