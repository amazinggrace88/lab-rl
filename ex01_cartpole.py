"""
강화학습 1 -> gym 사용하는 방법 알아보기
>>> cmd pip install gym (OpenAI Gym)
"""
import gym
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # gym package version 확인
    print(gym.__version__)  # 0.15.6

    # gym package 의 환경 (environment - 게임이 실행되는 환경) 리스트 출력
    print(gym.envs.registry.all())  # all 메소드

    # CartPole-v1 게임 환경을 생성 [make]
    # : 게임 실행환경을 만든다.
    env = gym.make('CartPole-v1')  # id : CartPole-v1

    # 환경 초기화 [reset]
    # : 게임 환경 초기화(random화)
    obs = env.reset()  # observation(관찰) : 머신러닝이 게임을 봐야 하기 때문에 -> reset 된 것을 본다.
    print(obs)  # [0.01916787 0.0109055  0.01954739 0.01084673]
    # 해석
    # [0.01916787-카트의 정중앙 위치 0.0109055-속도(전진(양수)/후진(음수))
    # 0.01954739-막대기의 각도(시계방향(양수)/반시계(음수)) 0.01084673-막대기의 속도=각속도(단위시간동안 움직인 각도)]

    # 환경 시각화 [render]
    # : 화면 보이게 하기
    env.render()  # rendering : 렌더링 () -> 게임 이미지를 나타낸다.

    # 환경 렌더링 이미지를 이미지 배열로 저장
    # img = env.render(mode='rgb_array')
    # print(img.shape)  # (400, 600, 3) : numpy.ndarray (x, y, 3) - x-by-y pixel image
    # print(img)
    # plt.imshow(img)
    # plt.show()

    # 게임을 실행 [action]
    # : 게임을 실행, 게임 상태 변경 (카트를 움직이면서 막대를 서있게 함)
    # 가능한 액션의 갯수
    print(env.action_space)  # action 들이 모여있는 공간  # Discrete(2) 불연속적 값 2가지 action 존재
    """
    CartPole 게임의 액선
    0 : 왼쪽 방향(-)으로 가속도를 줌
    1 : 오른쪽 방향(+)으로 가속도를 줌
    """

    # 게임을 진행 [step]
    action = 1  # action 종류 설정
    obs, reward, done, info = env.step(action)  # 게임 상태 설정 -> step 함수 : 다음 단계로 action 1 진행한다.
    print(obs)  # [-0.00124276  0.18280879  0.02532078 -0.29406667] : 두번째 관측치
    print(reward)  # 1.0
    print(done)  # False 막대가 쓰러졌다면 True
    print(info)  # {}

    action = 1
    obs, reward, done, info = env.step(action)
    print(obs)  # [ 0.05061262  0.41328121 -0.02303044 -0.63545019]

    action = 0
    obs, reward, done, info = env.step(action)
    print(obs)  # [ 0.03061994  0.23592278 -0.06437479 -0.36999589]

    """
    관성에 대하여
    - 각속도는 관성에 따라 움직인다. (카트의 속도는 막대의 속도와 반비례한다.)
    """

    # 사용했던 게임 환경 종료 [close]
    env.close()