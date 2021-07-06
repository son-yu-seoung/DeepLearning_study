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
