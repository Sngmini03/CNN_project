import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cpu")
print("사용 장치:", device)

train_dir = r"C:\Users\seung\OneDrive\사진\바탕 화면\CNN_project\POC_Dataset\Training"
test_dir = r"C:\Users\seung\OneDrive\사진\바탕 화면\CNN_project\POC_Dataset\Testing"
batch_size = 32
num_epochs = 20
learning_rate = 0.0001

### data preprocessing
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
test_dataset = datasets.ImageFolder(test_dir, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class_names = train_dataset.classes
print("class list:", class_names)

### googlenet (보조 분류기 포함)
model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)

# 메인 출력 레이어 수정
model.fc = nn.Linear(model.fc.in_features, len(class_names))

# 보조 분류기가 존재하는 경우에만 수정
if model.aux1 is not None:
    model.aux1.fc2 = nn.Linear(model.aux1.fc2.in_features, len(class_names))
if model.aux2 is not None:
    model.aux2.fc2 = nn.Linear(model.aux2.fc2.in_features, len(class_names))

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

### train loop
def train(model, loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    
    # tqdm 진행 표시줄 추가
    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]', leave=False)
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        
        # 학습 모드에서는 보조 분류기 출력 포함 (outputs, aux1, aux2)
        outputs = model(images)
        
        if isinstance(outputs, tuple):  # 학습 모드
            main_output, aux1, aux2 = outputs
            loss1 = criterion(main_output, labels)
            loss2 = criterion(aux1, labels)
            loss3 = criterion(aux2, labels)
            loss = loss1 + 0.3 * (loss2 + loss3)
        else:  # 평가 모드 (안전장치)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        # 진행 표시줄에 현재 loss 표시
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return running_loss / len(loader)

### test evaluate
def evaluate(model, loader, epoch):
    model.eval()  # 평가 모드: 보조 분류기 비활성화, 메인 출력만 반환
    correct = 0
    total = 0

    # tqdm 진행 표시줄 추가
    pbar = tqdm(loader, desc=f'Epoch {epoch} [Test]', leave=False)

    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)  # 평가 모드에서는 단일 텐서 반환
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 진행 표시줄에 현재 정확도 표시
            current_acc = 100 * correct / total
            pbar.set_postfix({'acc': f'{current_acc:.2f}%'})

    acc = 100 * correct / total
    return acc

### implement
print("\n학습 시작...\n")
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, epoch+1)
    test_acc = evaluate(model, test_loader, epoch+1)

    print(f"[Epoch {epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f} | Test Acc: {test_acc:.2f}%")

print("\n학습 완료!")    
torch.save(model.state_dict(), "googlenet_model.pth")
print("모델 저장 완료: googlenet_model.pth")