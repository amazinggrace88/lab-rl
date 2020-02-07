"""
CartPole with Neuralnet (basic 적용)
"""
import gym
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
# keras 가 tensorflow 안에 있음

def render_policy_net(model, max_steps=200):
    """
    게임 화면을 출력한다.
    :param model: 신경망 모델
    :param max_steps: 한 episode에서 최대 step 횟수
    :return: 게임 화면 출력
    """
    env = gym.make('CartPole-v1')  # 게임 환경 생성
    obs = env.reset()  # 게임 환경 초기화
    env.render()  # 초기 화면 렌더링

    for step in range(max_steps):
        # 모델에 obs 을 적용하여 예측값 알아냄
        p = model.predict(obs.reshape(1, -1))
        # 예측값 p 을 이용하여 action 을 결정 (0: 왼쪽 , 1: 오른쪽)
        action = int(np.random.random() > p)
        # action 을 바뀐 환경에 적용(step) -> 다음 스텝(환경)으로 변화
        obs, reward, done, info = env.step(action)
        env.render()  # 바뀐 환경을 렌더링
        if done:
            print(f'finished! #{step + 1}')
            break
    if not done:  # max_step 반복하는 동안 막대가 쓰러지지 않았을 경우
        print('Still Alive !! ')
    
    env.close()  # 게임 환경 종료
    

if __name__ == '__main__':
    # Sequential 모델
    model = keras.Sequential()  # =keras.model.Sequential()
    # fully connected hidden layer 를 추가
    model.add(keras.layers.Dense(4, activation='elu', input_shape=(4,)))  # () : 생성자
    # fully connected 출력층 추가
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    """
    relu 와 elu 의 차이
    relu(Rectified Linear Unit) -> 원점에서 불연속인 함수
    : f(x) = x (x>0), 0 (otherwise)
    elu(Exponential Linear Unit) -> relu 와 달리 원점에서도 미분가능한 함수 
    : 지수함수 f(x) = x (x>0), alpha * (exp(x) - 1) (otherwise)
    0 <= alpha < 1
    """
    # 신경망 요약정보
    model.summary()
    # dense - param (4 * 4) + 4 = 20
    # Trainable params: 25 학습을 시켜야 할 파라미터.

    # CartPole game 환경 생성
    env = gym.make('CartPole-v1')
    obs = env.reset()
    print(obs)  # [-0.02590488 -0.00990675 -0.00409705(막대 왼쪽으로 내려감)  0.03518737(오른쪽으로 돌려고 함)]

    # obs 을 신경망 모델에 forward 시켜 예측값을 알아냄
    print('obs shape = ', obs.shape)  # (4,) -> (1, 4) 로 만든다.
    p = model.predict(obs.reshape(1, -1))  # obs.reshape(1, -1) 행 1개, 열은 알아서 맞추기.
    print('p = ', p)

    action = int(np.random.random() > p)  # random 값을 한 번 더 준다.
    print('action = ', action)
    # 0~1 균일분포 > p (왼쪽으로 가야 한다는 확률)
    # p 가 1에 가까울 수록 np.random.random() > p 은 False 가 될 확률이 높아짐 -> int(False) = 0 -> 왼쪽으로 cart 움직이겠다는 의미
    # p 가 0에 가까울 수록 np.random.random() > p 은 True 가 될 확률이 높아짐 -> int(True) = 1 -> 오른쪽으로 cart 움직이겠다는 의미
    # 당연히, 왼쪽으로 가야 한다는 확률이 높아질 수록  action 또한 왼쪽으로 갈 확률이 높아짐.

    env.close()  # 게임 환경 종료
    
    # 함수 이용하여 신경망 모델을 전달하여 게임 실행
    render_policy_net(model)

    """
    신경망의 학습 가능한 파라미터 (W, b) 을 Gradient Descent 방법으로 학습시키자.
    mini-batch : 게임 환경을 여러 개(50개) 만들어서 신경망에서 input 으로 사용 
    """
    n_envs = 50  # 학습에 사용할 게임 환경(environments)의 갯수
    n_iterations = 1_000  # 학습 횟수 - 파라미터 갱신 횟수

    # 게임 환경 병렬로 50개 생성
    environments = [gym.make('CartPole-v1') for _ in range(n_envs)]
    # 게임 환경 50개 초기화 -> (row=50개, col=4개) 인 미니배치처럼 행렬화 되었다.
    observations = [env.reset() for env in environments]  # env 가 environment list 안에 여러개 있으므로 -> 2차원 배열 되었다.
    # Gradient 를 업데이트하는 방법 선택
    optimizer = keras.optimizers.RMSprop()
    # 손실함수(loss function) 선택
    loss_fn = keras.losses.binary_crossentropy

    # 학습
    for iteration in range(n_iterations):
        # 문제점 : 강화학습은 label(정답) 이 없어 오차 계산 불가
        # 해결책 : 가상의 label 을 정의한다. -> label 정답 = target 정답 = policy 정책
        # label 정답 = target 정답 을 정의하기 위한 정책 세우기
        # angle > 0 이면 target = 0
        # angle < 0 이면 target = 1
        target_probs = np.array([
            ([0.] if obs[2] < 0 else [1.])
            for obs in observations
        ])  # 2 차원 배열

        with tf.GradientTape() as tape:  # Gradient 계산
            loss_probs = model(np.array(observations))  # np.array 로 obs 만들어 model 에 대입
            loss = tf.reduce_mean(loss_fn(target_probs, loss_probs))  # target, loss 값을 가지고 손실함수를 계산
            print(f'iteration #{iteration}, loss : {loss.numpy()}')
            grads = tape.gradient(loss, model.trainable_variables)  # gradient 를 계산
            optimizer.apply_gradients(zip(grads, model.trainable_variables)) 
            actions = (np.random.rand(n_envs, 1) < loss_probs.numpy()).astype(np.int32)  # 50 개 np.array 로 각각 비교하여 숫자로 만들자.
            for idx, env in enumerate(environments):
                obs, reward, done, info = env.step(actions[idx][0])
                observations[idx] = obs if not done else env.reset()  # 중간에 게임이 끝나면 다시 리셋하고 다시 움직여야 함. -> 가중치 갱신 n_iterations 숫자만큼 해야함.
            
    # 게임 환경 50개 close
    for env in environments:
        env.close()

    # 학습이 끝난 모델을 이용해서 게임을 실행
    render_policy_net(model)

