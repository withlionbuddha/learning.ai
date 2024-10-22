
import torchvision
import torch
from torchvision import transforms


# MNIST 데이터셋 로드 함수
def load_mnist_data():
  # Define a transform to convert the images to tensors and normalize them
    transform = transforms.Compose([
        transforms.ToTensor()         # Convert images to tensor
    ])

    # Load the MNIST dataset (train set for example)
    mnist_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Create a DataLoader to fetch images
    dataloader = torch.utils.data.DataLoader(mnist_data, batch_size=2, shuffle=True)
    # Get a batch of images
    data_iter = iter(dataloader)
    x_train, labels = next(data_iter)  # Corrected this line
    
    # x_train을 (batch_size, 28*28) 형식으로 변환
    if not isinstance(x_train, torch.Tensor):
      raise TypeError("x_train은 Tensor 타입이어야 합니다.")
    
    #size? 텐서의 각 차원의 크기를 반환
    #unpacking? size객체에서 값을 각각의변수에 할당.
    batch_size, channels, height, width = x_train.size()

    print("x_tran.size() ", x_train.size())
    #dim? tensor의 차원수 반환 
    #flatten(펼침)? 다차원 데이터를 1차원 벡터로 변환. 
    #Fully Connected Layer, Dense Layer은 1차원 벡터 형태의 데이터를 요구.
    #view? 데이터 구성은 그대로, tensor의 size 차원를 변환 
    if x_train.dim() >= 2:
      x_train = x_train.view(batch_size, -1) #-1? 첫번째 차원은 유지, 나머지 차원은 하나의 차원으로 flatten.
    
    inputs = x_train.numpy() # 이미지나 다른 데이터는 신경망에 입력하기 전에 배열(numpy)로 변환하여 처리
    labels = labels.numpy()  # 레이블을 numpy 배열로 변환    

    return inputs, labels