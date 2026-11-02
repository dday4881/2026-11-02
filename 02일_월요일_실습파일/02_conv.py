# """실습 02 · 공간 크기를 유지하고 특징 채널을 8개 만든다."""
# import torch
# from torch import nn

# torch.manual_seed(42)
# x = torch.rand(4, 3, 32, 32)
# # Conv2d의 필수: 입력 채널, 출력 채널, 커널 크기.
# # padding=1은 선택 인자. 3×3, stride=1과 결합하면 높이·너비를 유지한다.
# # BEGIN B02
# # TODO B02: 3입력·8출력·3×3·padding 1의 합성곱 층을 만든다.
# conv = nn.Conv2d(3, 8, 3)
# # END B02
# y = conv(x)
# print("입력:", tuple(x.shape))
# print("출력:", tuple(y.shape))
# print("파라미터:", sum(p.numel() for p in conv.parameters()))

"""실습 02 · 공간 크기를 유지하고 특징 채널을 8개 만든다."""
import torch
from torch import nn

# 무작위 연산(랜덤 값 생성 및 가중치 초기화)의 결과를 항상 같게 고정합니다.
torch.manual_seed(42)

# 1교시와 동일하게 RGB 이미지 4장(4, 3, 32, 32)을 가상으로 생성합니다.
x = torch.rand(4, 3, 32, 32)

# Conv2d의 필수: 입력 채널, 출력 채널, 커널 크기.
# padding=1은 선택 인자. 3×3, stride=1과 결합하면 높이·너비를 유지한다.
# BEGIN B02

# TODO B02: 3입력·8출력·3×3·padding 1의 합성곱 층을 만든다.
# nn.Conv2d(in_channels, out_channels, kernel_size, padding)
# - 3 (in_channels)  : 입력 데이터의 채널 수 (RGB 이미지이므로 3)
# - 8 (out_channels) : 출력할 특징 맵(Feature Map)의 채널 수. 8개의 서로 다른 필터(커널)를 씁니다.
# - 3 (kernel_size)  : 이미지의 특징을 훑는 필터의 가로세로 크기 (3x3 크기)
# - padding = 1      : 이미지 가장자리에 빈 테두리를 1칸씩 둘러서, 연산 후에도 가로세로 크기가 줄어들지 않게 방어합니다.
conv = nn.Conv2d(3, 8, 3, padding = 1)

# END B02

# 위에서 정의한 conv 레이어에 입력 데이터 x를 통과시켜 연산을 수행합니다.
y = conv(x)

# 입력 데이터의 크기를 확인합니다. 출력: (4, 3, 32, 32)
print("입력:", tuple(x.shape))

# 출력 데이터의 크기를 확인합니다. 출력: (4, 8, 32, 32)
# 필터를 8개 썼기 때문에 채널은 3 -> 8로 늘어났지만, padding=1 덕분에 가로세로(32, 32)는 그대로 유지되었습니다.
print("출력:", tuple(y.shape))

# 이 합성곱 층(conv)이 학습해야 할 총 가중치(Weights + Biases) 변수의 개수를 계산해 출력합니다.
# 계산 공식: (필터 크기 3x3x입력채널3 + 바이어스1) * 출력채널8 = (27 + 1) * 8 = 224개
print("파라미터:", sum(p.numel() for p in conv.parameters()))

""" 출력 결과 값 
입력: (4, 3, 32, 32)
출력: (4, 8, 32, 32)
파라미터: 224   """

