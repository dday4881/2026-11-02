# """실습 05 · 제공된 CNN으로 5 epoch 학습한다. test는 읽지 않는다."""
# from pathlib import Path
# import copy
# import torch
# from torch import nn
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader

# root = Path(__file__).resolve().parent
# torch.manual_seed(42)
# torch.set_num_threads(1)
# transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
# train = datasets.ImageFolder(root / "data" / "train", transform=transform)
# valid = datasets.ImageFolder(root / "data" / "valid", transform=transform)
# # 작은 정상 데이터는 한 배치로 읽어 학습 순서를 단순하게 관찰한다.
# x_train, y_train = next(iter(DataLoader(train, batch_size=len(train), shuffle=False)))
# x_valid, y_valid = next(iter(DataLoader(valid, batch_size=len(valid), shuffle=False)))

# model = nn.Sequential(
#     nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
#     nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
#     nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 2),
# )
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
# loss_fn = nn.CrossEntropyLoss()
# best_loss = float("inf")
# for epoch in range(1, 6):
#     model.train()
#     optimizer.zero_grad()
#     logits = model(x_train)
#     loss = loss_fn(logits, y_train)
#     # 해설 B05: backward는 미분값 계산, step은 그 값을 이용한 갱신이다.
#     # BEGIN B05
#     # TODO B05: loss를 역전파하고 optimizer로 한 번 갱신한다.
#     pass
#     # END B05
#     model.eval()
#     with torch.no_grad():
#         train_loss = loss_fn(model(x_train), y_train).item()
#         valid_logits = model(x_valid)
#         valid_loss = loss_fn(valid_logits, y_valid).item()
#         valid_acc = (valid_logits.argmax(1) == y_valid).float().mean().item()
#     print(f"epoch={epoch} train_loss={train_loss:.4f} valid_loss={valid_loss:.4f} valid_acc={valid_acc:.2f}")
#     if valid_loss < best_loss:
#         best_loss, best_epoch = valid_loss, epoch
#         best_state = copy.deepcopy(model.state_dict())
# model.load_state_dict(best_state)
# print("선택한 epoch:", best_epoch)

"""실습 05 · 제공된 CNN으로 5 epoch 학습한다. test는 읽지 않는다."""
from pathlib import Path
import copy
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 현재 파일의 상위 폴더 경로를 기준점으로 잡습니다.
root = Path(__file__).resolve().parent

# 무작위 연산(시드) 고정 및 CPU 연산에 사용할 스레드 수를 1개로 제한하여 결과를 항상 일정하게 만듭니다.
torch.manual_seed(42)
torch.set_num_threads(1)

# 이미지 전처리 규칙 정의 (32x32 크기 조절 및 파이토치 텐서 변환)
transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

# 학습용(train) 데이터셋과 검증용(valid) 데이터셋을 각각 폴더 구조에 맞춰 불러옵니다.
train = datasets.ImageFolder(root / "data" / "train", transform=transform)
valid = datasets.ImageFolder(root / "data" / "valid", transform=transform)

# 작은 정상 데이터는 한 배치로 읽어 학습 순서를 단순하게 관찰한다.
# 데이터셋 전체 크기(len)만큼을 batch_size로 설정해, 데이터 전체를 통째로(한 번에) 뽑아냅니다.
# x_train, x_valid: 이미지 데이터들 | y_train, y_valid: 해당 이미지들의 정답 라벨들
x_train, y_train = next(iter(DataLoader(train, batch_size=len(train), shuffle=False)))
x_valid, y_valid = next(iter(DataLoader(valid, batch_size=len(valid), shuffle=False)))

# 3교시 때 설계했던 CNN 신경망 구조를 그대로 모델로 정의합니다.
model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 2),
)

# 최적화 도구(AdamW)를 설정합니다. 모델의 파라미터들을 학습률(lr) 0.01의 보폭으로 업데이트합니다.
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)

# 다중 분류 문제(여기서는 2가지 클래스 분류)의 손실(오차)을 계산하는 크로스 엔트로피 손실 함수입니다.
loss_fn = nn.CrossEntropyLoss()

# 가장 다이어트가 잘 된(손실이 가장 낮은) 모델을 저장하기 위해 초기 기준값을 무한대(inf)로 설정합니다.
best_loss = float("inf")

# 전체 데이터를 총 5번 반복해서 학습(5 Epoch) 시킵니다.
for epoch in range(1, 6):
    # [학습 모드 시작] 모델에게 지금부터 '학습(가중치 수정 가능)'할 것이라고 선언합니다.
    model.train()
    
    # 지난번에 계산해둔 가중치의 기울기(오답 노트 기록)를 깨끗하게 청소(0으로 초기화)합니다.
    optimizer.zero_grad()
    
    # 1. 순전파(Forward): 학습용 데이터를 모델에 넣어 예측값(logits)을 뽑아냅니다.
    logits = model(x_train)
    
    # 2. 채점: 예측값(logits)과 실제 정답(y_train)을 비교하여 오차(loss)를 구합니다.
    loss = loss_fn(logits, y_train)
    
    # 해설 B05: backward는 미분값 계산, step은 그 값을 이용한 갱신이다.
    # BEGIN B05
    # TODO B05: loss를 역전파하고 optimizer로 한 번 갱신한다.
    
    # 3. 역전파(Backward): 오차를 바탕으로 각 가중치들이 '얼마나 잘못했는지(기울기)' 역추적해 계산합니다.
    loss.backward() 
    
    # 4. 업데이트(Step): 계산된 가중치들의 오답 원인을 바탕으로 값을 '실제로 갱신'합니다.
    optimizer.step()
    # END B05
    
    # [평가 모드 시작] 검증을 위해 모델에게 드롭아웃 등 학습용 기능을 끄고 '평가'만 하라고 선언합니다.
    model.eval()
    
    # 평가할 때는 굳이 컴퓨터 메모리를 써가며 가중치의 기울기를 계산할 필요가 없으므로 추적을 끕니다.
    with torch.no_grad():
        # 학습 데이터에 대한 현재 손실 값을 순수한 숫자(.item())로 가져옵니다.
        train_loss = loss_fn(model(x_train), y_train).item()
        
        # 검증 데이터를 모델에 넣어 예측값을 구합니다.
        valid_logits = model(x_valid)
        # 검증 데이터에 대한 손실 값을 계산합니다.
        valid_loss = loss_fn(valid_logits, y_valid).item()
        
        # 검증 데이터의 정확도(Accuracy)를 계산합니다. 
        # (가장 높은 확률을 가진 인덱스(argmax) == 실제 정답) 비율의 평균(mean)을 냅니다.
        valid_acc = (valid_logits.argmax(1) == y_valid).float().mean().item()
        
    # 현재 에폭의 성적표(학습 오차, 검증 오차, 검증 정확도)를 화면에 출력합니다.
    print(f"epoch={epoch} train_loss={train_loss:.4f} valid_loss={valid_loss:.4f} valid_acc={valid_acc:.2f}")
    
    # 만약 이번 에폭의 검증 오차가 지금까지 중 가장 작다면(제일 똑똑하다면) 정보를 백업합니다.
    if valid_loss < best_loss:
        best_loss, best_epoch = valid_loss, epoch # 최고 기록 갱신
        # 현재 가장 우수한 모델의 가중치 상태(뼈대와 근육의 수치 정보)를 깊은 복사(copy)로 보관해둡니다.
        best_state = copy.deepcopy(model.state_dict())

# 5번의 연습이 모두 끝난 후, 보관해 두었던 '가장 성적이 좋았던 순간(best_state)'의 가중치를 모델에 최종 로드합니다.
model.load_state_dict(best_state)

# 가장 뛰어난 성능을 냈던 에폭 번호를 출력하며 마무리합니다.
print("선택한 epoch:", best_epoch)

""" 출력 결과 값
epoch=1 train_loss=0.6703 valid_loss=0.6718 valid_acc=0.50
epoch=2 train_loss=0.6389 valid_loss=0.6442 valid_acc=0.50
epoch=3 train_loss=0.5896 valid_loss=0.6020 valid_acc=1.00
epoch=4 train_loss=0.5209 valid_loss=0.5423 valid_acc=1.00
epoch=5 train_loss=0.4358 valid_loss=0.4664 valid_acc=1.00
선택한 epoch: 5 """