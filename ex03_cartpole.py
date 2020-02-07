"""
cartpole - policy 적용
"""
import gym
import random
import numpy as np


def random_policy():
    """policy - 무작위로 카트를 움직임"""
    return random.randint(0, 1)  # 0(-) or 1(+)


def basic_policy(observation):
    """
    막대가 기울어진 각도 : theta
    theta > 0 : 왼쪽으로 카트가 이동하므로 오른쪽으로 카트를 친다.(action = 1)
    theta < 0 : 오른쪽으로 카트가 이동하므로 왼쪽으로 카트를 친다. (action = 0)
    """
    theta = observation[2]  # theta
    if theta > 0:
        action = 1
    else:
        action = 0
    return action


def custom_policy(observation):
    """
    막대가 기울어진 각도 : theta
    theta > 0 -> action = 1
    theta < 0 -> action = 0
    막대가 기울어지는 각의 속도 : w
    w > 0 -> v = 1
    w < 0 -> v = 0
    """
    theta = observation[2]
    w = observation[3]
    action = 0
    v = 0
    if theta > 0 or w > 0:
        action = 1
    if theta < 0 and w < 0:
        action = 0
    return action


if __name__ == '__main__':
    env = gym.make('CartPole-v1')  # environment 생성

    max_episodes = 100  # 게임 실행 횟수 (pole 이 넘어지기 전 : 1 episodes)
    max_step = 1_000  # 1 에피소드에서 최대 반복 횟수 (카트를 움직이는 횟수)
    total_rewards = []  # 에피소드 끝날 때마다 얻은 보상을 저장할 리스트

    for episode in range(max_episodes):
        print(f'----- Episode #{episode + 1} -----')
        obs = env.reset()  # 게임 환경 초기화 : pole 이 넘어지고 나서 게임이 시작하지 않도록 !
        episode_reward = 0  # 1개의 episode 에서 얻은 보상점수
        for step in range(max_step):
            env.render()  # 초기화 화면 출력
            action = custom_policy(obs)  # policy 선택 가능 - w 조건 추가한 공식 max = 262
            obs, reward, done, info = env.step(action)
            episode_reward += reward  # reward 를 더한다.
            if done:
                print(f'finished ! ---- # {step + 1} reward : {episode_reward}')
                break
        total_rewards.append(episode_reward)

    # 보상(점수)들의 리스트의 평균, 표준편차, 최댓값, 최솟값
    print(f'mean : {np.mean(total_rewards)}')
    print(f'std : {np.std(total_rewards)}')
    print(f'max : {np.max(total_rewards)}')
    print(f'min : {np.min(total_rewards)}')

    env.close()
