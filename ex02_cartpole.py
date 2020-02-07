"""
cartpole - step for 문으로 묶기
"""
import gym
import numpy as np


if __name__ == '__main__':
    # 게임 environment 생성
    env = gym.make('CartPole-v1')
    # 게임 환경 초기화
    obs = env.reset()
    # 초기화 화면 출력
    env.render()
    print(obs)

    max_steps = 100  # 반복횟수
    for t in range(max_steps):
        action = 1  # action : 오른쪽으로 이동하기
        obs, reward, done, info = env.step(action)  # 게임 진행
        env.render()  # 게임 환경 시각화하여 출력
        print(obs)
        print(f'reward : {reward}, done : {done}, info : {info}')

    env.close()  # 게임 환경 종료

    """
    env.step(action) 설명
    obs : 위치 [x 카트 위치, v 카트 속도, theta 막대의 기울어져 있는 각도, w 각속도]
    reward : 1.0씩 보상 (if done = False) / 0.0 (if done = True)
    done : 게임이 끝나면 True / 진행된다면 False
    info : 정보 -> 복잡한 게임은 여러가지 정보가 들어가 있을 수 있음.
    """

    # action random 으로 만들기
    obs = env.reset()
    env.render()

    max_steps = 100
    reward_sum = 0
    for t in range(max_steps):
        action = np.random.randint(0, 1)  # int(np.random.randn()) ok.
        obs, reward, done, info = env.step(action)
        env.render()
        print(obs)
        reward_sum += reward
        if done:  # if done == True
            print(f'finished after {t+1} steps')
            print(reward_sum)
            break

    env.close()

