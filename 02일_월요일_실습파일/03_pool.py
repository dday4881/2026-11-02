# """실습 03 · 클래스 정의 없이 Sequential로 작은 CNN을 연결한다."""
# import torch
# from torch import nn

# torch.manual_seed(42)
# x = torch.rand(4, 3, 32, 32)
# # MaxPool2d(2)는 2×2에서 최댓값을 선택한다. 기본 stride도 2이다.
# # BEGIN B03
# # TODO B03: 출력 높이·너비를 절반으로 만드는 MaxPool을 작성한다.
# pool = nn.MaxPool2d(1)
# # END B03
# model = nn.Sequential(
#     nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), pool,
#     nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
#     nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 2),
# )
# for layer in model:
#     x = layer(x)
#     print(type(layer).__name__, tuple(x.shape))

"""실습 03 · 클래스 정의 없이 Sequential로 작은 CNN을 연결한다."""
import torch
from torch import nn

# 일관된 실험 결과를 위해 무작위 연산 기준(시드)을 고정합니다.
torch.manual_seed(42)

# 가상의 RGB 이미지 데이터 4장을 만듭니다. (크기: 4장, 3채널, 32가로, 32세로)
x = torch.rand(4, 3, 32, 32)

# MaxPool2d(2)는 2×2 영역에서 가장 큰 값 하나를 선택해 추출하는 기능입니다.
# 기본 보폭(stride)도 2로 설정되므로, 가로와 세로 크기가 각각 딱 '절반'으로 줄어듭니다.
# BEGIN B03

# TODO B03: 출력 높이·너비를 절반으로 만드는 MaxPool을 작성한다.
pool = nn.MaxPool2d(2)

# END B03

# nn.Sequential은 여러 개의 신경망 레이어를 한 줄로 순서대로 연결해 주는 상자입니다.
# 데이터가 이 상자에 들어가면 위에서부터 아래로 차례대로 통과하게 됩니다.
model = nn.Sequential(
    # [1] 3채널 입력 -> 16채널 출력, 3x3 필터, 패딩 1 -> (4, 16, 32, 32)
    nn.Conv2d(3, 16, 3, padding=1), 
    
    # [2] 활성화 함수 활성화 (0보다 작은 값은 0으로 제거, 크기 유지) -> (4, 16, 32, 32)
    nn.ReLU(), 
    
    # [3] 맥스 풀링 (가로세로 절반으로 압축) -> (4, 16, 16, 16)
    pool,
    
    # [4] 16채널 입력 -> 32채널 출력, 3x3 필터, 패딩 1 -> (4, 32, 16, 16)
    nn.Conv2d(16, 32, 3, padding=1), 
    
    # [5] 활성화 함수 활성화 (크기 유지) -> (4, 32, 16, 16)
    nn.ReLU(), 
    
    # [6] 맥스 풀링 (가로세로 또 절반으로 압축) -> (4, 32, 8, 8)
    nn.MaxPool2d(2),
    
    # [7] 가로세로(H, W) 크기를 강제로 지정한 크기(1, 1)로 평균 내어 압축합니다. -> (4, 32, 1, 1)
    nn.AdaptiveAvgPool2d(1), 
    
    # [8] 4차원 텐서를 2차원(행렬)으로 평평하게 펼쳐줍니다. (4, 32*1*1) -> (4, 32)
    nn.Flatten(), 
    
    # [9] 최종 분류를 위한 선형 결합 층 (32개 입력 특징 -> 최종 2개 클래스로 분류) -> (4, 2)
    nn.Linear(32, 2),
)

# 모델 내부의 레이어들을 하나씩 꺼내어 순서대로 연산해 보면서 데이터 크기 변화를 관찰합니다.
for layer in model:
    x = layer(x) # 현재 레이어에 데이터를 통과시킴
    # 레이어의 이름과 통과한 후의 데이터 모양(Shape)을 예쁘게 출력합니다.
    print(type(layer).__name__, tuple(x.shape))

    """ 출력 결과 값
Conv2d (4, 16, 32, 32)
ReLU (4, 16, 32, 32)
MaxPool2d (4, 16, 16, 16)
Conv2d (4, 32, 16, 16)
ReLU (4, 32, 16, 16)
MaxPool2d (4, 32, 8, 8)
AdaptiveAvgPool2d (4, 32, 1, 1)
Flatten (4, 32)
Linear (4, 2)    """