# < 목차 >
+ [RNN이란](#RNN(Recurrent-Neural-Network))
+ [RNN 레이어의 종류](#RNN-종류)  
  - [SimpleRNN](#SimpleRNN-Layer)
  - [LSTM](#LSTM-Layer)
  - [GRU](#GRU-Layer)
  - [Embedding](#Embedding-Layer)
  - [U-Net](#U-Net)
  - [ResNet](#ResNet)
+ [Hyperparameter](#Hyperparameter)
+ [Overfitting](#Overfitting)

# RNN(Recurrent Neural Network)
RNN은 은닉층의 노드에서 활성화 함수를 통해 나온 결과값을 출력층 방향으로도 보내면서, 다시 은닉층 노드의 다음 계산의 입력으로 보내는 특징을 가지고 있다. (결과값을 오직 출력층 방향으로 보내는 것을 Feed Forward 신경망이라 함)  

RNN은 입력과 출력의 길이를 다르게 설계할 수 있으므로 다양한 용도로 사용할 수 있다.  

![RNN](https://user-images.githubusercontent.com/65440674/141928350-8a1c2089-e69a-4947-b823-2425faea42e8.png)

- RNN 셀의 각 시점 별 입, 출력의 단위는 사용자가 정의하기 나름이지만 가장 보편적인 단어는 '단어 벡터'이다.  
- 하나에 입력에 대해 여러개의 출력(일 대 다)의 모델은 하나의 이미지 입력에서 사진의 제목을 출력하는 이미지 캡셔닝 작업에 사용할 수 있다. 사진의 제목은 단어들의 나열이므로 시퀀스 출력이다.  
- 또한 단어 시퀀스에 대해서 하나의 출력(다 대 일)을 하는 모델은 입력 문서가 긍정적인지 부정적인지를 감성 분류, 또는 메일이 정상 메일인지 스팸 메일인지 판별하는 스팸 메일 분류에 사용할 수 있다.  
- 다 대 다의 모델의 경우에는 입력 문장으로 부터 대답 문장을 출력하는 챗봇과 입력 문장으로부터 번역된 문장을 출력하는번역기, 또는 개체명 인식이나 품사 태깅과 같은 작업 또한 속한다.  

### 긴 의존 기간으로 인한 문제점  
RNN의 성공의 열쇠는 "Long Short-Term Memory Network"(이하 LSTM)의 사용이다.   
LSTM은 RNN의 굉장히 특별한 종류로, 아까 얘기했던 영화를 frame 별로 이해하는 것과 같은 문제들을 단순 RNN 보다 정말 훨씬 진짜 잘 해결한다.  
기존 RNN도 LSTM만큼 이런 일을 잘 할 수 있다면 RNN은 대단히 유용할 텐데, 아쉽게도 RNN은 그 성능이 상황에 따라 그 때 그 때 다르다.  

![문제](https://user-images.githubusercontent.com/65440674/141972566-9badebad-c1ad-4097-a917-86ac6dd8bb09.png)  
                                                <p align=center>**<긴 기간에 의존하는 RNN>**</p>
  
"I grew up in France... I speak fluent French"라는 문단의 마지막 단어를 맞추고 싶다고 생각해보자. 최근 몇몇 단어를 봤을 때 아마도 언어에 대한 단어가 와야 될 것이라 생각할 수는 있지만, 어떤 나라 언어인지 알기 위해서는 프랑스에 대한 문맥을 훨씬 뒤에서 찾아봐야 한다.    
이렇게 되면 필요한 정보를 얻기 위한 시간 격차는굉장히 커지게 된다.    
**안타깝게도 이 격차가 늘어날 수록 RNN은 학습하는 정보를 계속 이어나가기 힘들어한다.**  



# RNN 종류
### LSTM(참고 - https://dgkim5360.tistory.com/entry/understanding-long-short-term-memory-lstm-kr)
LSTM은 RNN의 특별한 한 종류로, 긴 의존 기간을 필요로 하는 학습을 수행할 능력을 갖고 있다.   
LSTM은 Hochreiter & Schmidhuber (1997)에 의해 소개되었고, 그 후에 여러 추후 연구로 계속 발전하고 유명해졌다. LSTM은 여러 분야의 문제를 굉장히 잘 해결했고, 지금도 널리 사용되고 있다.  
  
+ LSTM은 긴 의존 기간의 문제를 피하기 위해 명시적으로(explicitly) 설계되었다.

# RNN의 종류

### SimpleRNN Layer
  
### LSTM Layer
- SimpleRNN 레이어의 치명적 단점인 장기의존성 문제를 해결할 수 있는 모델
 
(그림)
i : input
f : forget
o : output
c(~) : x(t)와 h(t-1)을 각각 U와 W에 곱한 뒤에 tanh 함수를 취한 값, 셀 상태인 c(t)가 되지 전의 출력 값
c : 셀 상태
h : 셀 상태에 tanh 함수를 취한 값을 Output 게이트의 출력에 곱한다.
  
### GRU Layer
- LSTM 과의 가장 큰 차이점은 셀 상태가 보이지 않는다는 것. GRU에서는 h(t)가 비슷한 역할을 한다.
- GRU 레이어는 LSTM 레이어와 비슷한 역할을 하지만 구조가 더 간단하기 때문에 계산상의 이점이 있다.
- 어떤 문제에서는 LSTM보다 GRU가 좋으며 파라미터 수에서도 이점을 볼 수 있다.
  
(그림)
z(t) : Update 게이트를 통과한 출력
r(t) : Reset 게이트
h(~t) : 
h(t) : 
  
### Embedding Layer
임베딩 레이어는 자연어를 수치화된 정보로 바꾸기 위한 레이어이다.
  

