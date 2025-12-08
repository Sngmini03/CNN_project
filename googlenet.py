import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

device = torch.device("cpu")
print("사용 장치:", device)

train_dir = r"C:\Users\seung\OneDrive\사진\바탕 화면\CNN_project\POC_Dataset\Training"
test_dir = r"C:\Users\seung\OneDrive\사진\바탕 화면\CNN_project\POC_Dataset\Testing"
batch_size = 32
num_epochs = 50 
learning_rate = 0.001
patience = 7  # early stopping 

### data preprocessing
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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

# 전체 train dataset 로드
full_train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
test_dataset = datasets.ImageFolder(test_dir, transform=transform_test)

# Train:Validation = 7:3 분할
train_size = int(0.7 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

print(f"Train 데이터: {train_size}개")
print(f"Validation 데이터: {val_size}개")
print(f"Test 데이터: {len(test_dataset)}개")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class_names = full_train_dataset.classes
print("class list:", class_names)

### googlenet 
model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1, dropout=0.3)

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

# 학습률 스케줄러 (ReduceLROnPlateau)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3)

### train loop
def train(model, loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]', leave=False)
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        
        outputs = model(images)
        
        if isinstance(outputs, tuple):  
            main_output, aux1, aux2 = outputs
            loss1 = criterion(main_output, labels)
            loss2 = criterion(aux1, labels)
            loss3 = criterion(aux2, labels)
            loss = loss1 + 0.3 * (loss2 + loss3)
            
            _, predicted = torch.max(main_output, 1)
        else:  
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })

    avg_loss = running_loss / len(loader)
    avg_acc = 100 * correct / total
    return avg_loss, avg_acc

### validation/test evaluate
def evaluate(model, loader, epoch, mode='Val'):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch} [{mode}]', leave=False)

    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
            
            current_acc = 100 * correct / total
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.2f}%'
            })

    avg_loss = running_loss / len(loader)
    avg_acc = 100 * correct / total
    return avg_loss, avg_acc

### Early Stopping 클래스
class EarlyStopping:
    def __init__(self, patience=7, verbose=True, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}). Saving model...')
        torch.save(model.state_dict(), 'best_googlenet_model.pth')
        self.val_loss_min = val_loss

### implement
print("\n학습 시작...\n")
early_stopping = EarlyStopping(patience=patience, verbose=True)
best_val_acc = 0.0

for epoch in range(num_epochs):
    # Train
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch+1)
    
    # Validation
    val_loss, val_acc = evaluate(model, val_loader, epoch+1, mode='Val')
    
    # 학습률 스케줄러 업데이트
    scheduler.step(val_loss)
    
    # 현재 학습률 출력
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"[Epoch {epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
          f"LR: {current_lr:.6f}")
    
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
    
    
    early_stopping(val_loss, model)
    
    if early_stopping.early_stop:
        print(f"\nEarly stopping triggered at epoch {epoch+1}")
        break

print("\n학습 완료!")


print("\n최고 성능 모델로 테스트 진행...")
model.load_state_dict(torch.load('best_googlenet_model.pth'))


test_loss, test_acc = evaluate(model, test_loader, epoch+1, mode='Test')
print(f"\n최종 Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
print(f"최고 Validation Acc: {best_val_acc:.2f}%")

torch.save(model.state_dict(), "final_googlenet_model.pth")
print("\n모델 저장 완료:")
print("- best_googlenet_model.pth (최고 성능 모델)")
print("- final_googlenet_model.pth (최종 모델)")