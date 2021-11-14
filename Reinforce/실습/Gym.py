# 텐서플로 2 버전 선택
try:
    # %tensorflow_version only exists in Colab.
    %tensorflow_version 2.x
except Exception:
    pass
import tensorflow as tf
print(tf.__version__)
# 현재 model.predict() 의 속도가 느린 문제가 있어서 eager_execution을 끕니다.
# 관련 버그 이슈 링크: https://github.com/tensorflow/tensorflow/issues/32104
tf.compat.v1.disable_eager_execution()
import numpy as np


!pip install gym pyvirtualdisplay
!apt-get install -y xvfb python-opengl ffmpeg

!apt-get update
!apt-get install cmake
!pip install --upgrade setuptools
!pip install ez_setup
!pip install gym[atari]

!pip install box2d-py
!pip install gym[Box_2D]

!pip install pyglet
< ----------------- ----------------- > 
# Reinforcement Learning(강화학습)은 실수와 보상을 통해 배우는 알고리즘
# Gym은 전통적인 알고리즘 흉내 내기, Box2D를 사용한 간단한 물리 조작계, 아타리(Atari)게임, 로봇 시뮬레이션
# Gym의 구조는 강화학습에서 요구하는 표준적인 구조입니다. 
# 문제가 주어진 환경이 있고, 강화학습 문제를 풀기 위한 에이전트(agent)가 존재
# 에이전트는 해옹으로 환경에 영향을 주고, 그 결과에 따라 보상을 받는다.
# 좋은 보상을 받으면 에이전트는 그 행동을 더 많이 하게 되고, 나쁜 보상을 받으면 그 행동을 덜 하도록 학습하는 것이 강화학습의 기본
# MountainCar-v0 환경을 불러옴
import gym 
import random
env = gym.make('MountainCar-v0')
# 환경 정보 파악을 위해 env의 속성을 출력
print(env.observation_space) # observation_space는 에이전트가 환경을 볼 수 있는 작은 창
print(env.observation_space.low) # [x좌표, 속도]의 최솟값
print(env.observation_space.high) # [x좌표, 속도]의 최대값
print()
print(env.action_space) # Discrete(3) = 3가지 행동 가능 0 : 왼, 1 : 정지, 2 : 오
print()
print(env._max_episode_steps) # 이 에피소드는 200번째 시간 단위에서 종료 
< ----------------- ----------------- > 
# env.render() 함수의 결과를 mp4 동영상으로 보여주기 위한 코드
# from https://colab.research.google.com/drive/1flu31ulJlgiRL1dnN2ir8wGh9p7Zij2t
from gym import logger as gymlogger
from gym.wrappers import Monitor
gymlogger.set_level(40) #error only
import glob
import io
import base64
from IPython.display import HTML
from IPython import display as ipythondisplay

"""
Utility functions to enable video recording of gym environment and displaying it
To enable video, just do "env = wrap_env(env)""
"""

def show_video():
    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else: 
        print("Could not find video")
    

def wrap_env(env):
    env = Monitor(env, './video', force=True)
    return env
  
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()
< ----------------- ----------------- > 
env = wrap_env(gym.make('MountainCar-v0'))
env.reset() # 가상환경 초기화(꼭 필요함) 

step = 0
score = 0 

while True:
  action = env.action_space.sample() # 랜덤한 행동을 선택 0, 1, 2 중 하나가 랜덤하게 들어감
  obs, reward, done, info = env.step(action) # env.step에 action을 전달함과 동시에 시간이 흘러감, 관찰상태(obs), 보상(reward), 종료 여부(done), 기타(info)
  print(score)
  score += reward # step()함수에서 보상을 받는데 보상은 누적해서 계산함 score 변수에 보상 저장
  step += 1 # 시간이 얼마나 흘렀는지 step도 저장 

  if done: # 최대 시간 단위에 도달하기 전에 에피소드가 끝나면 done = True 
    break
  
print('score:', score)
print('step:', step)
env.close()
show_video()
< ----------------- ----------------- > 
import numpy as np
# 200의 단위 시간내에 문제를 해결하기 위해서는 여러 번의 에피소드 중 성공적인 에피소드를 저장한 후
# 그때 행동했던 데이터를 신경망에 학습시키는 방법을 사용할 수 있다!!
# 랜덤한 행도을 하는 에이전트로 10,000번의 에피소드를 실행하고 성공적인 에피소드의 데이터를 저장하겠습니다.
env = gym.make('MountainCar-v0')

scores = []
training_data = []
accepted_scores = []
required_score = -198

for i in range(5000):
  if i % 100 == 0:
    print(i)
  env.reset()
  score = 0
  game_memory = [] # 입력 데이터와 출력 데이터를 저장하기 위한 변수
  previous_obs = [] # 변수와 이전 스텝의 관찰 상태를 저장하기 위한 변수

  while True:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action) # obs는 행동

    if len(previous_obs) > 0:
      game_memory.append([previous_obs, action]) # 이전 상태가 존재할 경우 저장, 아직 이 에피소드가 성공적일지는 모름, action에는 뭐가 들어가지?
    
    previous_obs = obs
    if obs[0] > -0.2: # 지금 여기서는 가속도를 이용해 약간이라도 앞으로 간 에피소드라면 보상 +1을 줌
      reward = 1
    
    score += reward

    if done: # 200 step을 가거나 그전에 완료하면 break
      break
    
  scores.append(score) # 해당 에피소드의 보상 합계를 입력
  if score > required_score: # -198 보다 score가 크다면 즉, 가속도로 인한 보상 +1을 3번 이상 받았다면
    accepted_scores.append(score) 
    for data in game_memory: # 성공적이라고 판단된 에피소드의 순서(움직이는)를 training_data에 입력 
      training_data.append(data)

scores = np.array(scores) # 모든 에피소드의 보상 합계를 numpy 배열로 바꾼다.
print(scores.mean())
print(accepted_scores)

import seaborn as sns # seaborn : 데이터 시각화 라이브러리, matpotlib를 기반으로 좀 더 확장된 기능을 사용가능
sns.distplot(scores, rug=True) # distplot() 함수는 데이터의 분포를 빠르게 확인가능, rug 인수를 True로 설정하면 데이터 포인트 하나하나를 바닥에 깔린 작은 선으로 표시
< ----------------- ----------------- > 
# 첫 번째 값은 관찰 상태([이전 내 x좌표, 이후 내 x좌표], 행동), 두 번째 값은 행동
# array([-0.~, -0.~], 0])
print(training_data[:5])
# 이 값들을 신경망이 학습할 수 있도록 X와 Y로 분리해서 저장
train_X = np.array([i[0] for i in training_data]).reshape(-1, 2) # 관찰 상태
train_Y = np.array([i[1] for i in training_data]).reshape(-1, 1) # 행동

# 관찰 상태에 대한 행동을 학습할 분류 신경망 정의
import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(2,), activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_X, train_Y, epochs=30, batch_size=16, validation_split=0.25)
< ----------------- ----------------- > 
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], 'b-', label='accuracy')
plt.plot(history.history['val_accuracy'], 'r--', label='val_accuracy')
plt.legend()
plt.show()
# 약 40%의 정확도를 보이는데, 행동이 3가지이기 때문에 랜덤한 행동을 할 때 33.3%의 정확도를 보일 것이라고 가정하면 
# 랜덤 행동보다는 뭔가 의미 있는 지식, 관찰 상태에 대한 적합한 행동이라는 지식을 얻음 
< ----------------- ----------------- > 
# MountainCar-v0 환경에서 이 신경망이 정해주는 행동으로 에이전트를 움직여봄
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

scores = []
steps = []
actions = []

for i in range(500):
    if i % 100 == 0:
        print(i)
    score = 0
    step = 0
    previous_obs = []
    env.reset()

    while True:
        if len(previous_obs) == 0: # 처음에
            action = env.action_space.sample() #  랜덤한 동작으로 시작
        else: # 처음이 아니라면
            logit = model.predict(np.expand_dims(previous_obs, axis=0))[0]  # (2, ) -> (1, 2)(예상, 확인필요)
            action = np.argmax(logit) # 3개의 동작 중 가장 확률이 높은 행동을 가져옴
            actions.append(action) # 내가 이번에 취해야하는 행동을 actions 리스트에 삽입
        
        obs, reward, done, info = env.step(action)
        previous_obs = obs
        score += reward
        step += 1

        if done:
            break
    
    scores.append(score)
    steps.append(step)
< ----------------- ----------------- > 
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].set_title('score')
ax[1].set_title('step')
sns.distplot(scores, rug=True, ax=ax[0])
sns.distplot(steps, rug=True, ax=ax[1])

print(np.mean(scores))
< ----------------- ----------------- > 
sns.distplot(actions) 
< ----------------- ----------------- > 
env.close()
env = wrap_env(gym.make('MountainCar-v0'))
env.reset()

score = 0
step = 0
previous_obs = []
while True:
    if len(previous_obs) == 0:
        action = env.action_space.sample()
    else:
        logit = model.predict(np.expand_dims(previous_obs, axis=0))[0]
#         logit = logit.astype(float)
#         logit = logit / logit.sum()
#         prob = np.random.multinomial(1, logit)
#         action = np.argmax(prob)
        action = np.argmax(logit)
    
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
