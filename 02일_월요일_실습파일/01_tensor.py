# """실습 01 · RGB 이미지의 N,C,H,W를 확인한다. 실제 사진 대신 정상 Tensor 4장."""
# import torch

# torch.manual_seed(42)
# # 해설 B01: rand의 네 크기는 이미지 수, 채널, 높이, 너비이다.
# # 필수: 각 차원의 크기. 선택: dtype, device. 기본 float32·CPU를 사용한다.
# # BEGIN B01
# # TODO B01: RGB 이미지 4장, 높이·너비 32인 Tensor를 만든다.
# images = torch.rand(1, 3, 32, 32)
# # END B01
# print("shape:", tuple(images.shape))
# print("dtype:", images.dtype)
# print("device:", images.device)
# print("첫 이미지:", tuple(images[0].shape))

"""실습 01 · RGB 이미지의 N,C,H,W를 확인한다. 실제 사진 대신 정상 Tensor 4장."""
import torch

# 랜덤 수 생성 시 항상 같은 결과가 나오도록 시드(기준)를 고정합니다. (실험 재현용)
torch.manual_seed(42)

# 해설 B01: rand의 네 크기는 이미지 수, 채널, 높이, 너비이다.
# 필수: 각 차원의 크기. 선택: dtype, device. 기본 float32·CPU를 사용한다.
# BEGIN B01

# TODO B01: RGB 이미지 4장, 높이·너비 32인 Tensor를 만든다.
# torch.rand()는 0~1 사이의 무작위 실수를 생성하는 함수입니다.
# 괄호 안의 숫자는 각각 (이미지 개수 N, 채널 수 C, 높이 H, 너비 W)를 의미합니다.
# - 4: 이미지 총 4장 (Batch Size, N)
# - 3: RGB 3개 색상 채널 (빨강, 초록, 파랑, C)
# - 32: 세로 해상도 32 픽셀 (Height, H)
# - 32: 가로 해상도 32 픽셀 (Width, W)
images = torch.rand(4, 3, 32, 32)

# END B01

#전체 텐서의 모양(크기)을 출력합니다. 출력 결과: (4, 3, 32, 32)
print("shape:", tuple(images.shape))

# 데이터의 타입(종류)을 출력합니다. 기본값은 32비트 실수형인 float32입니다.
print("dtype:", images.dtype)

# 이 텐서가 컴퓨터의 CPU와 GPU(CUDA) 중 어디에 저장되어 있는지 확인합니다.
print("device:", images.device)

# 전체 4장의 이미지 중 '첫 번째 이미지(인덱스 0번)' 한 장의 크기만 잘라서 확인합니다.
# 이미지 수는 제외되므로 채널, 높이, 너비인 (3, 32, 32)만 출력됩니다.
print("첫 이미지:", tuple(images[0].shape))

""" 출력 결과 값
shape: (4, 3, 32, 32)
dtype: torch.float32
device: cpu
첫 이미지: (3, 32, 32)  """