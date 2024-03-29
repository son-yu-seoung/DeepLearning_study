# < 목차 >
+ [CNN의 종류](#CNN-종류)  
  - [AlexNet](#AlexNet)
  - [LeNet](#LeNet)
  - [ZFNet](#ZFNet)
  - [GoogleNet](#GoogleNet)
  - [U-Net](#U-Net)
  - [ResNet](#ResNet)
+ [Hyperparameter](#Hyperparameter)
+ [Overfitting](#Overfitting)

# CNN 종류
### AlexNet   
: AlexNet은 ImageNet 영상 데이터 베이스를 기반으로 한 화상 인식 대회인 "ILSVRC 2012"에서 우승한 CNN 구조이다. 전체적으로 보면 2개의 GPU를 기반으로 한 병렬 구조인 점을 제외하면, 
LeNet-5와 크게 다르지 않다. AlexNet은 5개의 convolution layers와 3개의 fully-connected layers로 구성되어 있으며 마지막 FC layer는 1000개의 category로 분류하기 위해 활성 함수로 softmax
함수를 사용하고 있다. 이러한 설계로 인해 AlexNet은 약 65만개의 뉴런과 6,000만개의 자유 파라미터, 6.3억개의 connection이라는 방대한 CNN 구조를 가지게 되었다. AlexNet은 2개의 GPU를 사용한
학습 이외에도 성능, 속도 개선을 위하여 아래와 같은 여러가지 방법을 사용한다.  

1. ReLU

2. Local Response Normalization (LRN)
- LRN은 같은 위치의 픽셀에 대해서, 복수의 feature map 간에 정규화를 하는 방법이다. 이 방법은 실제 세포에서 발생하는 측면 억제 현상과 같은 효과가 있으며, 망의 일반화를 돕는 효과가 있다. 
AlexNet에서는 LRN을 첫 번쨰와 두 번째 convolution 다음에 적용

3. Overlapping Pooling
- 일반적으로 max pooling을 할 때는 각각 중복되지 않는 영역에서 pooling한다. 하지만 AlexNet은 3x3 영역을 2픽셀 단위로 pooling하여 조금씩 겹치는 부분이 있도록 pooling하여 overfitting 현상
을 개선했다.

4. Data Augmentation
: AlexNet은 overfitting을 억제하기 위해 학습 데이터를 증가 시키는 방법을 사용
- 256x256 이미지에서 랜덤으로 227x227 이미지를 crop
- RGB 채널 값 변화

5. Dropout

- AlexNet의 블록의 depth는 LeNet에서 feature map의 개수와 같은 의미이고, 각 층의 연산 과정은 동일

### LeNet  
: LeNet은 Convolution Neural Network라는 개념을 최초로 개발한 Yann LeCun이 개발한 구조이다. Convolution과 Subsampling을 반복적으로 거치면서, 마지막에 Fully-connected Multi-layerd Neural Network로 Classification을 수행하고 있다

### ZFNet
: ZFNet은 ILSVRC 2012에서 우승한 AlexNet에 이어 ILSVRC 2013에서 우승한 CNN구조이다. ZFNet의 구조 자체는 AlexNet에서 GPU를 하나만 쓰고 일부 convolution layer의 kernel사이즈와 stride를 
일부 조절한 것뿐이다. ZFNet의 논문의 핵심은, ZFNet의 구조 자체보다도 CNN을 가시화하여 CNN의 중간 과정을 눈으로 보고 개선 방향을 파악할 방법을 만들었다는 것에 있다.

ZFNet 논문의 Visualizing 기법
- CNN의 중간 layer의 feature map은 그 자체로는 의미를 알기 어려우므로, 입력 이미지 공간에 mapping 하여 분석할 수 있다.
- CNN은 convolution 계산 후 활성 함수를 통해 feature amp을 생성하고, pooling 하여 이미지를 축소 시키는 것을 반복한다.
- 그렇다면, 그 동작을 반대로 하면 원본 이미지에 mapping 할 수 있는 이미지를 생성할 수 있을 것이다.
- 가장 문제가 되는 것은 max-pooling 인데, max-pooling은 이미지를 축소 시키는 과정에서 가장 강한 자극만을 전달한다 이것을 반대로 할 때는 그 강한 자극이 어디에서 온 자극인지 알 수 없는 문제가
있다. ZFNet 논문의 Visualizing 기법에서는 switch 라는 개념을 생각해 내어, max-pooling 과정의 가장 강한 자극의 위치를 가지고 있도록 하여, un-pooling 할 때 그 위치를 알 수 있도록 하였다는 
게 핵심이다.


# Hyperparameter

1. Learning rate	 
- Learning rate는 결과의 오차를 학습에 얼마나 반영할 지를 결정하는 변수
- 학습률이 낮으면 local minima에 수렴하게 될 가능성이 있다.
- 학습률이 높으면 결과가 수렴되지 않고 진동할 가능성이 있다.

2. Cost function  
: 입력에 따른 기대 값과 실제 값의 차이를 계산하는 함수
- Mean Square Error (평균 제곱 오차)
- Cross-Entorpy Error (교차 엔트로피 오차)

3. Regularization parameter   
- 정규화를 하지 않았을 때는 overfitting 된 결과를 얻을 수 있다.

4. Mini-batch 크기  
: 모든 데이터의 Cost function의 합을 구하려면 데이터가 많을수록 많은 시간이 걸린다.
그 때문에 데이터 중 일부를 사용하여 가중치를 갱신하여 학습하게 된다. 이때, Mini-batch의 크기가 크면 병렬 연산 구조에서 학습 속도를 높일 수 있으며, 크기가 작으면 더 자주 update 할 수 있다.

5. Training 반복 횟수  
: Training 횟수가 너무 많으면 Overfitting 되어 실제 정확도가 떨어질 수 있다.

6. Hidden unit의 개수  
: hidden layer의 hidden unit의 개수가 많으면 네트워크로 표현력이 넓어져 더 좋은 성능을 낼 수도 있지만, Overfitting될 수 있는 문제가 있다.

7. 가중치 초기화(Weight initialization)  
: 가중치 초기 값을 잘못 설정하면 학습이 효과를 보지 못할 수 있다.

Hypermarameter Optimization, 위의 다양한 Hyperparameter를 설정하는 여러 방법  

1. Grid Search   
: Hyperparameter의 대략적인 범위를 지정하고 일정한 간격으로 값을 선택하여 학습하는 방법, 이 방법은 hyperparameter를 범위 내에서 골고루 테스트를 해보는 방법이지만, Hyperparameter가 늘어날수록 그 배수만큼 학습 시간이 늘어나 비효율적이다.
  
2. Random search     
: 대략적인 범위를 지정하고 그 범위 내에서 랜덤하게 값을 선택하여 학습하는 방법 Grid와는 랜덤이라는 차이점만 가지고 있다.

3. Bayesian optimization  
: 기존 학습 결과로 Hyperparameter의 사전 분포를 가정하고, 최적의 Hyperparameter로 가정되는 값의 학습 결과를 획득하여 사후 분포를 결정하는 작업을 반복하는 방법입니다. 


# Overfitting  
학습 속도 저하 - 원인  
:  일반적으로 사용하는 Activation의 sigmoid 함수의 미분 특성으로 인해 결합되게 되면서 학습이 느려지는 현상이 발생하게 된다. (블라블라 한 이유로)

속도 개선 - ReLU(Rectified Linear Unit) function  
: sigmoid function의 문제점을 해결하기 위해 생겨난 활성화 함수로 가장 큰 장점은 미분이 아주 간단하게 된다는 것이다.

학습 능력의 저하 - Overfitting  
: 학습을 할 때 한정된 데이터에 너무 특화가 되어 새로운 데이터에 대한 결과가 나바지거나 학습효과가 나타나지 않는 경우를 말합니다. 

학습 능력의 개선 - Regularization, DropOut  
: 최고의 학습결과를 위해서는 가능성 있는 모든 데이터에 대해서 학습을 진행하는 것이 좋지만, 그것은 현실적으로 불가능하다. 그래서 학습 결과를 향상시키기 위해 여러가지 대안들이 제시되고 있다. 그 중에서 Regularization, DropOut, 지능적 훈련에 대해 알아보겠다.

1. Regularization  
- Penalty라는 개념을 도입하여 복잡한 쪽보다는 간단한 쪽으로 선택을 유도하는 방식이다. 즉, 특정 가중치 값들이 작아지도록 학습을 진행하게 되고 이는 일반화에 적합한 특성을 갖게 만드는 것이라고 볼 수 있습니다. 

2. Dropout  
- 망 자체를 변화시키는 방법이다. Hidden layer의 수가 많아질 경우 Deep Neural Network가 되면서 문제해결 가능성이 높아지지만 학습 시간이 길어지고, Overfitting에 빠질 가능성이 높아진다. Dropout은 망 내의 모든 layer에 대해 학습을 수행하는 것이 아니라 일부 뉴런을 생략하고 줄어든 신경망을 통해 학습을 수행한다.

< Convolution Neural Networks (CNN) >  

CNN은 feature을 추출하는 convolution layer와 추출된 feature들을 sub-sampling 하는 pooling layer로 구성되어 있습니다.   

Convolution이란?  
: 현재의 위치의 출력 데이터는 인접한 pixel에 convolution filter를 곱해서 얻어진 값입니다. 즉, 인접한 pixel 값의 영향력이 출력 데이터라고 보면 된다. 그 영향력은 convolution fliter가 결정한다. 그래서 이미지 처리 분야에서 특정 feature들을 추출하기 위해 convolution을 사용한다. ( sobel filter는 윤곽선을 feature로 하는 컨볼루션 연산 ) 

Pooling이란?  
: Pooling은 Convolution과 비슷하게 인접한 pixel 값만을 사용하지만, 특별히 곱하거나 더하는 연산은 없다. 대표적인 Pooling 방법은 Max Pooling과 Average Pooling이다. Max Pooling은 인접한 pixel에서 가장 큰 값을 새로운 pixel 값으로 정하고, Average Pooling은 인접한 pixel의 평균 값을 새로운 pixel 값으로 정한다. Pooling layer는 아래와 같은 특징이 있다.
- 학습해야 할 매개 변수가 없다.
- 채널 수가 변하지 않는다.
- 입력의 변화에 영향을 적게 받아 입력 데이터가 조금 변해도 pooling의 결과는 잘 변하지 않는다.

CNN 연산에서 설정해야하는 Parameter  
- Convolution Filter 개수
- Filter Size(Kernel_Size)
- Padding 여부
- Stride

< Batch Normalize & Dropout > 

Batch Normalization?  
: Deep Learning에서 Layer가 많아질 때 학습이 어려워지는 이유는 weight의 미세한 변화들이 누적되어 쌓이게 되면, hiddent layer가 높아질수록 값에 변화가 커지기 때문이다.
즉, Network의 각 층이나 Activation 마다 input의 distribution이 달라지는 현상이 Internal Covariate Shift입니다. 
Internal Covariate Shift를 해결하기 위해서 activation의 변화, 초기 weigh의 설정, 낮은 learning rate 등을 이용하였지만, 이는 근본적인 해결책이 아니고 Training 과정 자체를 전체적으로 안정화시킬 수 있는 방법이 필요하다.
weight에 따른 weighted sum의 변화의 폭이 줄어든다면 학습이 잘될 것이라는 Batch Normalization의 기본 가정이다. 단순히 distirbution을 평균 0, 표준 편차 1인 distribution으로 정규화시키는 방법은 활성 함수의 non-linear를 없앨 수 있기 때문에 nomalize된 값에 scale factor와 shift factor가 추가되고, 이 factor는 Back-propagation 과정에서 Train 된다.
- 실제로 Batch Normalization을 Network에 적용시킬 때에는 특정 hidden layer에 들어가기 전에 Batch Normalization layer를 추가하여 input의 distribution을 바꾼 뒤 activation function으로 넣어주는 방식으로 이용된다.



