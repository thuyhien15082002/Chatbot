import json
from model import NeuralNetwork
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore

# tải dữ liệu intents.json
with open('../intents.json', encoding='utf-8') as f:
    intents = json.load(f)

#Xử lý dữ liệu
all_intents = []  # Lưu tất cả các từ đã tokenized
tags = []         # Lưu danh sách các tags (unique)
xy = []           # Lưu các cặp (patterns, tag)

# Duyệt qua tất cả các intent trong intents.json
for intent in intents['intents']:
    tag = intent['tag']
    if tag not in tags:
        tags.append(tag)  # Thêm tag vào danh sách nếu chưa có
    for pattern in intent['patterns']:
        w = tokenize(pattern)  # Tokenize câu patterns Tách câu patterns thành từ
        all_intents.extend(w)  # Lưu tất cả các từ trong patterns
        xy.append((w, tag))    # Lưu cặp (tokens, tag)

# Xóa các ký tự không cần thiết
ignore_words = ['?', '!', ',', '.']
all_intents = [stem(w) for w in all_intents if w not in ignore_words] #Stemming các từ để chuẩn hóa, ví dụ: "playing" → "play".

# Loại bỏ trùng lặp và sắp xếp danh sách các từ, tags
all_intents = sorted(set(all_intents))
tags = sorted(set(tags))  # Sửa tags đúng theo các `tag` từ intents

# Tạo dữ liệu huấn luyện
X_train = []  # Mỗi câu patterns được chuyển thành vector bag-of-words(đầu vào)
Y_train = []  # Danh sách các nhãn (chỉ số của tag)

for (pattern_sentence, tag) in xy:
    # Tạo bag of words cho câu patterns
    bag = bag_of_words(pattern_sentence, all_intents)
    X_train.append(bag)

    # Tìm chỉ số của tag và thêm vào tags
    label = tags.index(tag)
    Y_train.append(label)
    
#Chuyển đổi danh sách thành numpy array
X_train = np.array(X_train, dtype=np.float32)  # Chuyển sang float32
Y_train = np.array(Y_train)

#Tạo lớp Dataset để quản lý dữ liệu huấn luyện.
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = torch.tensor(X_train, dtype=torch.float32)  # Chuyển sang tensor float32
        self.y_data = torch.tensor(Y_train, dtype=torch.long)    # Chuyển sang tensor long (int64)

#Lấy một mẫu dữ liệu (bag-of-words và tag) theo chỉ số     
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
#Trả về số lượng mẫu dữ liệu.
    def __len__(self):
        return self.n_samples

#Cấu hình và huấn luyện mô hình
if __name__ == '__main__':
    batch_size = 8  #Kích thước batch.
    hidden_size = 8 #Số nơ-ron trong tầng ẩn.
    output_size = len(tags) #số nhãn (tags).
    input_size = len(X_train[0])    #Kích thước vector đầu vào (bag-of-words).
    learning_rate = 0.001 #Tốc độ học.
    num_epochs = 1000   #Số vòng lặp huấn luyện.

#Tạo DataLoader để tải dữ liệu theo batch.
    dataset = ChatDataset()
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
#Kiểm tra và sử dụng GPU nếu có, nếu không sẽ sử dụng CPU.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#Khởi tạo mô hình NeuralNetwork.
    model = NeuralNetwork(input_size, hidden_size, output_size).to(device)
#Hàm mất mát CrossEntropyLoss (dùng cho bài toán phân loại).
    criterion = nn.CrossEntropyLoss()
#Tối ưu hóa Adam.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#Vòng lặp huấn luyện: Lặp qua tất cả các batch trong train_loader.
# Dự đoán đầu ra (outputs) và tính toán hàm mất mát (loss).
# Tính gradient và cập nhật trọng số mô hình (optimizer.step()).
    for epoch in range(num_epochs):
        for words, labels in train_loader:
            words = words.to(device)  # Đầu vào đã là float32
            labels = labels.to(device)  # Nhãn phải là long (int64)
            
            # Forward pass
            outputs = model(words)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print(f'Final loss: {loss.item():.4f}')
 #Lưu mô hình   
    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_intents": all_intents,
        "tags": tags
    }
    
    FILE = "data.pth"   #chứa trọng số và thông tin cấu hình.
    torch.save(data, FILE)
    
    print(f'Training completr. Model saved to {FILE}')
