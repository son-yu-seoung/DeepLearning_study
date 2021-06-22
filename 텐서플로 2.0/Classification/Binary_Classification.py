# 분류(Classification)이란 가장 기초적인 데이터 분석 방법 중 하나
# 데이터가 어느 범주(Category)에 해당하는지 판단하는 문제

# 와인 데이터셋은 보스턴 주택 데이터세과 달리 외부에서 데이터를 불러오고 정제해야 함
import matplotlib.pyplot as plt
import pandas as pd
red = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')
white = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', sep=';')
# 12개의 속성으로 구성되어 있음
print(red.head())
print(white.head())

# + 와인이 레드 와인인지 화이트 와인인지 표시해주는 속성을 추가한 후 레드, 화이트 와인 데이터를 합쳐야 함
red['type'] = 0
white['type'] = 1
wine = pd.concat([red, white])
print(wine.describe)

# 히스토그램으로 red, white 와인의 개수를 시각화
plt.hist(wine['type'])
plt.xticks([0, 1])
plt.show() 
# 1   4898, 0   1599
print(wine['type'].value_counts())# white wine의 개수가 약 3배 정도는 더 많다.

# info() 함수는 데이터프레임을 구성하는 속성들 의 정보를 알려줌
# 정규화 과정에서 데이터에 숫자가 아닌 값이 들어가면 에러의 원인이 되기 때문에 확인 필요
print(wine.info())

# 각 데이터마다 단위가 다르기 때문에 정규화 과정 필요
wine_norm = (wine - wine.min()) / (wine.max() - wine.min())
print(wine_norm.head())
print(wine_norm.describe)

# 정규화된 데이터를 랜덤하게 섞고 학습을 위해 numpy로 변환하기 
p.113





