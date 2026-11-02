# """실습 04 · 필요한 하위 폴더는 ImageFolder의 이미지 저장 구조뿐이다."""
# from pathlib import Path
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader

# root = Path(__file__).resolve().parent
# # Resize(size)는 목표 크기가 필수. ToTensor()에는 필수 인자가 없다.
# transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
# # ImageFolder(root, transform=None): root는 필수, transform은 선택이다.
# # BEGIN B04
# # TODO B04: data/train을 주어진 transform으로 읽는다.
# train = datasets.ImageFolder(root / "data" / "train")
# # END B04
# loader = DataLoader(train, batch_size=4, shuffle=False, num_workers=0)
# images, labels = next(iter(loader))
# print("학습 이미지:", len(train))
# print("클래스:", train.class_to_idx)
# print("배치:", tuple(images.shape), tuple(labels.shape))
# print("라벨:", labels.tolist())

"""실습 04 · 필요한 하위 폴더는 ImageFolder의 이미지 저장 구조뿐이다."""
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 현재 실행 중인 파이썬 파일(__file__)의 절대 경로를 구해 상위 폴더(parent) 위치를 변수에 저장합니다.
# 이 위치를 기준으로 데이터 폴더 등 다른 파일들의 경로를 안전하게 찾을 수 있습니다.
root = Path(__file__).resolve().parent

# Resize(size)는 목표 크기가 필수. ToTensor()에는 필수 인자가 없다.
# transforms.Compose는 여러 개의 이미지 전처리 과정을 하나로 묶어주는 상자입니다.
transform = transforms.Compose([
    # 1단계: 어떤 크기의 이미지든 모델이 학습할 수 있도록 (32, 32) 가로세로 해상도로 크기를 똑같이 맞춥니다.
    transforms.Resize((32, 32)), 
    # 2단계: 일반 이미지(0~255 값)를 파이토치가 이해하는 텐서(0~1 사이의 실수형, C x H x W 형태)로 변환합니다.
    transforms.ToTensor()
])

# ImageFolder(root, transform=None): root는 필수, transform은 선택이다.
# BEGIN B04

# TODO B04: data/train을 주어진 transform으로 읽는다.
# datasets.ImageFolder는 폴더 이름이 곧 클래스(정답 라벨)가 되는 구조로 이미지를 자동으로 매칭하여 불러옵니다.
# - root / "data" / "train" : 이미지가 들어있는 실제 컴퓨터 내의 폴더 경로 (Path 객체 활용)
# - transform = transform    : 불러오는 이미지에 위에서 정의한 32x32 크기 변환 및 텐서 변환 규칙을 적용합니다.
train = datasets.ImageFolder(root / "data" / "train", transform = transform) 

# ⚠️ 헷갈렸던 오답 복습용 주석:
# 함수 내부에서 값을 매칭해줄 때는 쉼표(,)를 쓰고 괄호 안에서 인자(transform=transform)로 넣어주어야 합니다. 
# 괄호 밖에서 한 칸 띄우고 적으면 파이썬이 문법 오류(SyntaxError)를 일으킵니다!

# END B04

# DataLoader는 이미지 데이터셋(train)을 우리가 학습할 수 있게 '미니 배치(묶음)' 단위로 쪼개서 공급해주는 장치입니다.
# - batch_size=4 : 한 번에 4장씩 묶어서 가져옵니다.
# - shuffle=False: 데이터를 섞지 않고 순서대로 가져옵니다. (보통 평가/테스트 시 False, 학습 시 True)
# - num_workers=0: 데이터 로딩에 사용할 서브 프로세스 개수입니다. (0은 메인 프로세스만 사용함을 의미)
loader = DataLoader(train, batch_size=4, shuffle=False, num_workers=0)

# next(iter(loader))는 전체 데이터 로더에서 '첫 번째 미니 배치(첫 묶음)'인 이미지 4장과 그에 맞는 정답 라벨 4개를 직접 꺼내오는 문법입니다.
images, labels = next(iter(loader))

# 불러온 학습용 전체 이미지 데이터의 개수를 출력합니다.
print("학습 이미지:", len(train))

# 폴더명을 기반으로 자동으로 생성된 클래스 이름과 매칭된 숫자(인덱스) 구조를 확인합니다. 
# 예: {'cat': 0, 'dog': 1} 처럼 딕셔너리 형태로 출력됩니다.
print("클래스:", train.class_to_idx)

# 첫 배치의 형태를 확인합니다. 
# 이미지 크기는 (4, 3, 32, 32) -> 4장, 3채널, 32x32 크기이며, 라벨 크기는 (4,) -> 정답 4개 데이터가 됩니다.
print("배치:", tuple(images.shape), tuple(labels.shape))

# 텐서 상태인 정답 라벨을 보기 편하게 일반 파이썬 리스트([0, 0, 1, 1] 등) 형태로 바꾸어 출력합니다.
print("라벨:", labels.tolist())


""" 출력 결과 값
    학습 이미지: 12
클래스: {'horizontal': 0, 'vertical': 1}
배치: (4, 3, 32, 32) (4,)
라벨: [0, 0, 0, 0]  """