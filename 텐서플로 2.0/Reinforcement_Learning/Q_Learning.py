# 앞 절에서 시도한 방법은 신경망 네트워크를 사용하긴 했지만 강화학습의 이론을 사용하지는 않았다.
# 강화학습의 대표적인 방법론인 큐러닝(Q-Learning)
# 행동 공간이 이산적이지 않고 연속적인 MountainCar-v0 환경 만듬
import gym
import random
env = gym.make('MountainCarContinuous-v0')

print(env.observation_space) # (-1.2~, 0.6~, (2,), float32) x 축 범위 
print(env.observation_space.low) # (x 좌표, 속도)의 최솟값
print(env.observation_space.high) # (x 좌표, 속도)의 최댓값
print()
print(env.action_space),
print(env.action_space.low) # 최솟값 -1, 이 사이의 값을 지정해서 에이전트를 왼쪽이나 오른쪽으로 움직일 수 있다.
print(env.action_space.high) # 최댓값 1, 행동의 절대값이 클수록 에이전트는 큰 힘을 받게된다.
print()
print(env._max_episode_steps) # 최대 시간 단위 = 999
< -------------- -------------- >
env.reset()
score = 0
step = 0
    
for i in range(999):
    action = env.action_space.sample() 
    obs, reward, done, info = env.step(action)

    previous_obs = obs
    score += reward
    step += 1

    if done:
        break
        
print(score, step)
# 각 스텝마다 얻는 보상은 obs^2 * 0.1 * -1, 움직이지 않으면 0의 보상, 깃발을 획득하면 +100
# 중요한건 큰 힘으로 움직일 수록 보상은 적어진다....(보상 함수는 MountainCar-v0보다 까다롭다.)
# score가 작을 수록(더 큰 음수일 수록) 성공적인 에피소드??
# 해당 env에서는 연속된 100회의 에피소드에서 +90 이상의 누적 보상을 얻는다면 환경을 풀었다고 판단.
# p.412 글부터 읽기 (중요!!!)
< -------------- -------------- >
# 분류가 아닌 회귀 신경망을 사용, 관찰 사애에 따른 하나의 행동값을 추측하는 문제
scores = []
training_data = [] # (-1, 2)
accepted_scores = []
required_score = -198

for i in range(5000):
    if i % 100 == 0:
        print(i)
    env.reset()
    score = 0
    game_memory = []
    previous_obs = []
    
    for i in range(200): # 999 아닌가
        action = env.action_space.sample() # 3가지 액션 중에 한 가지 액션 랜덤으로 가져옴
        obs, reward, done, info = env.step(action)
        
        if len(previous_obs) > 0:
            game_memory.append([previous_obs, action])
        
        previous_obs = obs
        if obs[0] > -0.2: # obs = [현재 위치, 현재 속도](예측)
            reward = 1
        else:
            reward = -1
        
        score += reward
        
        if done:
            break
        
    scores.append(score)
    if score > required_score:
        accepted_scores.append(score)
        for data in game_memory:
            training_data.append(data)

import numpy as np

scores = np.array(scores)
print(scores.mean())
print(accepted_scores)

import seaborn as sns
sns.distplot(scores, rug=True) 
< -------------- -------------- >
# 회귀 모델 정의
import tensorflow as tf
train_X = np.array([i[0] for i in training_data]).reshape(-1, 2)
train_Y = np.array([i[1] for i in training_data]).reshape(-1, 1)
print(train_X.shape)
print(train_Y.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(2,), activation='elu'), # elu는 0이하일 때 -1로 수렴, relu는 0이하일 때 0 고정
    tf.keras.layers.Dense(32, activation='elu'),
    tf.keras.layers.Dense(1, activation='linear') # 출력을 그대로 내보내는 linear 활성화 함수
])
model.compile(optimizer=tf.optimizers.Adam(), loss='mse', metrics=['accuracy']) # categorical_crossentropy가 아닌 mse를 사용

history = model.fit(train_X, train_Y, epochs=10, validation_split=0.25)
< -------------- -------------- >
scores = []
steps = []
actions = []

for i in range(500):
    if i % 100 == 99:
        print(i, 'mean score: {}, mean step: {}'.format(np.mean(scores[-100:]), np.mean(steps[-100:])))
    score = 0
    step = 0
    previous_obs = []
    env.reset()

    while 200:
        if len(previous_obs) == 0:
            action = env.action_space.sample()
        else:
            action = model.predict(np.expand_dims(previous_obs, axis=0))[0] # 실행만 계속됌 
            actions.append(action)
        
        obs, reward, done, info = env.step(action)
        previous_obs = obs
        score += reward
        step += 1

        if done:
            break
    
    scores.append(score)
    steps.append(step)
< -------------- -------------- >
# 10.22 score, step 분포 확인
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].set_title('score')
ax[1].set_title('step')
sns.distplot(scores, rug=True, ax=ax[0])
sns.distplot(steps, rug=True, ax=ax[1])

print(np.mean(scores))
< -------------- -------------- >
# 10.23 행동 분포 확인
sns.distplot(actions)
< -------------- -------------- >
# 10.24 회귀 신경망 행동 에이전트 실행 결과 확인
env.close()
env = wrap_env(gym.make('MountainCarContinuous-v0'))
env.reset()

score = 0
step = 0
previous_obs = []
while True:
    if len(previous_obs) == 0:
        action = env.action_space.sample()
    else:
        action = model.predict(np.expand_dims(previous_obs, axis=0))[0]
    
    obs, reward, done, info = env.step(action)
    previous_obs = obs
    score += reward
    step += 1
    
    if done:
        break

print('score:', score)
print('step:', step)
env.close()
show_video()
< -------------- -------------- >
# 지금까지 본 결과 회귀 신경망으로 이 문제를 푸는 것에는 한계가있다.
# 큐러닝을 사용해 풀어보자. 큐러닝은 특정 상태에서 어떤 행동의 Q값이 다른 행동보다 높으면 
# Q값이 높은 행동을 우선적으로 선택할 수 있게 해주는 방법
# 소프트맥스 함수로 각 Q값을 입력으로 삼아 확률을 기반으로한 행동을 선택할 수도 있다.
# 10.25 관찰 공간과 행동 공간을 격자화
state_grid_count = 10
action_grid_count = 6

q_table = []
for i in range(state_grid_count):
    q_table.append([])
    for j in range(state_grid_count):
        q_table[i].append([])
        for k in range(action_grid_count):
            q_table[i][j].append(1e-4)
            
actions = range(action_grid_count)
actions = np.array(actions).astype(float)
actions *= ((env.action_space.high - env.action_space.low) / (action_grid_count - 1))
actions += env.action_space.low

print(actions)
< -------------- -------------- >
# 10.26 obs_to_state, softmax 함수 정의
import random
def obs_to_state(env, obs): # 해당 상태에 맞는 격자 반환
    obs = obs.flatten()
    low = env.observation_space.low
    high = env.observation_space.high
    idx = (obs - low) / (high - low) * state_grid_count
    idx = [int(x) for x in idx]
    return idx

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    sum_exp_logits = np.sum(exp_logits)
    return exp_logits / sum_exp_logits
< -------------- -------------- >
# 10.27 큐러닝 에이전트 학습
max_episodes = 10000
scores = []
steps = []
select_actions = []

learning_rate = 0.05
gamma = 0.99 

# 강화학습에서 널리 쓰이는 방법 중 하나인 입실론-탐욕(epsilon-greedy)이라는 정책 사용
# p.425
epsilon = 1.0 # ?
epsilon_min = 0.01 # ?

for i in range(max_episodes):
    epsilon *= 0.9 
    epsilon = max(epsilon_min, epsilon)
    
    if i % 100 == 0 and i != 0:
        print(i, 'mean score: {}, mean step: {}, epsilon: {}'.format(np.mean(scores[-100:]), np.mean(steps[-100:]), epsilon))
        
    previous_obs = env.reset()
    score = 0
    step = 0
    
    while True:
        state_idx = obs_to_state(env, previous_obs) # 격자 위치 받음
        if random.random() < epsilon: # 0 이상 1 미만의 랜덤 숫자
            action_idx = random.randint(0, action_grid_count-1) # (a, b) a, b를 포함하는 범위 내 정수 반환
            action = actions[action_idx] # ?
        else:
            logits = q_table[state_idx[0]][state_idx[1]]
            action_idx = np.argmax(softmax(logits))
            action = actions[action_idx]
        
        obs, reward, done, info = env.step([action])
        previous_obs = obs
        score += reward
        reward -= 0.05
        step += 1
        
        select_actions.append(action)
        
        new_state_idx = obs_to_state(env, obs)
        
        q_table[state_idx[0]][state_idx[1]][action_idx] = \
            q_table[state_idx[0]][state_idx[1]][action_idx] + \
            learning_rate * (reward + gamma * np.amax(q_table[new_state_idx[0]][new_state_idx[1]]) - q_table[state_idx[0]][state_idx[1]][action_idx])
        
        if done:
            break
    
    scores.append(score)   
    steps.append(step)
    
    if np.mean(scores[-100:]) >= 90:
        print('Solved on episode {}!'.format(i))
        break
