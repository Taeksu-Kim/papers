# 참고 자료
https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLO   
https://github.com/shkim960520/YOLO-v1-for-studying   
https://github.com/motokimura/yolo_v1_pytorch   


# basic data information

dataset = https://www.kaggle.com/dataset/734b7bcb7ef13a045cbdd007a3c19874c2586ed0b02b4afc86126e89d00af8d2?select=train.csv
x는 이미지,
y는 class, Bounding Box들의 중심점 x좌표, y좌표, BB의 w, h 로 구성

input_image_size = (3 * 448 * 448)   

train_loader_shape = (iteration, batch_size, 3, 448, 488)   
x_shape = (batch_size, 3, 448, 448)   
   
output_size = (batch_size, gird_cell, gird_cell, class_num + Bounding_Box_num * 5)   
-> 5는 (confidence_socre, BB의 중심점 x, BB의 중심점 y, BB의 w, BB의 h)   
grid_cell = 7 * 7, class_num = 20, BB_num = 2   
-> (batch_size, 7, 7, 30)   

전체적인 흐름은 이미지를 (3, 448, 448)의 Tensor로 변환하여 넣고, (7, 7, 30)의 y 예측값을 낸다.   
하나의 이미지는 7 * 7의 gridcell에 2개의 BB가 포함되어 총 98개의 박스를 예측할 수 있고 그 중 IOU, NMS 등 우리가 정의한 기준에 의해   
걸러진 박스들과 실제 y의 BB 값들과 비교하여 모델의 loss와 mAP를 평가한다. 

# Model Architecture

모델의 구조에 대해서는 논문 내에 상당히 자세히 나와 있음   

![image](https://user-images.githubusercontent.com/63130907/124525241-7a392680-de39-11eb-9d69-f9eb7084d1de.png)

convolution 레이어에 대한 옵션 중 padding 값은 따로 입력해 줘야함.   
전체 크기 유지를 원할 시 -> (kernel_size - 1) // 2   
kernel size가 2 이상인데도 padding 값을 주지 않으면 크기는 kernel size - 1 만큼 줄어듬. ex) 10 * 10 이미지에 3 * 3 kernel을 사용할 경우 전체 크기는 8 * 8이 됨   

convolution layer에서는 이미지의 크기를 인자 값으로 주지는 않음.   

각 covolution layer에는 batch_norm과 LeakyReLU가 적용되므로 편의를 위해 Convolution layer 블럭으로 class를 별도 정의.   
nn.Module를 상속 받은 후에 forward에서 leakyrelu(batch_norm(convolution(x))) 형태로 전개   

convolution2d 블럭(in_channels, out_channels, kernel_size, stride, padding)   
maxpooling(kerneal_size, stride)   
   
convolution2d 블럭(3, 64, 7 * 7, 2 * 2, 3)   
maxpooling(2 * 2, 2 * 2)
   
convolution2d 블럭(64, 192, 3 * 3, 1 * 1, 1) # stride값이 없으므로 1 * 1 적용   
maxpooling(2 * 2, 2 * 2)   
   
convolution2d 블럭(192, 128, 1 * 1, 1 * 1, 0)   
convolution2d 블럭(128, 256, 3 * 3, 1 * 1, 1)   
convolution2d 블럭(256, 256, 1 * 1, 1 * 1, 0)   
convolution2d 블럭(256, 512, 3 * 3, 1 * 1, 1)   
maxpooling(2 * 2, 2 * 2)   
   
convolution2d 블럭(512, 256, 1 * 1, 1 * 1, 0)   
convolution2d 블럭(256, 512, 3 * 3, 1 * 1, 1)   
위의 두 레이어 * 4   
convolution2d 블럭(512, 512, 1 * 1, 1 * 1, 0)   
convolution2d 블럭(512, 1024, 3 * 3, 1 * 1, 1)   
maxpooling(2 * 2, 2 * 2)   
   
convolution2d 블럭(1024, 512, 1 * 1, 1 * 1, 0)   
convolution2d 블럭(512, 1024, 3 * 3, 1 * 1, 1)   
위의 두 레이어 * 2   
convolution2d 블럭(1024, 1024, 3 * 3, 1 * 1, 1)   
convolution2d 블럭(1024, 1024, 3 * 3, 2 * 2, 1)   
   
convolution2d 블럭(1024, 1024, 3 * 3, 1 * 1, 1)   
convolution2d 블럭(1024, 1024, 3 * 3, 1 * 1, 1)   
   
Flatten   
Linear(1024 * 7 * 7, 4096) # 7 * 7은 gird_cell 값   
droupout(0.5)   
LeakyRerlu(0.1)   
Linear(4096, 7 * 7 * (20 + 2 * 5)) # (7,7,30) class_num(20) + BB_num(2)*5 

-> output은 (batch_size, 1470)으로 나오므로 이를 나중에 (batch_size, 7 * 7 * 30)으로 다시 바꾼 후 사용

# Hyper parameters

weight_decay = 0.0005   
batch_size = 64   
epochs = 135   

momentum = 0.9   
learning rate = 첫번째 epoch에서는 각 iteration e-3에서 e-2로 증가, 그 다음 75 epochs는 e-2, 그 다음 30 epochs는 e-3,마지막 30 epochs는 e-4로 진행   

참고한 논문이 2016년에 수정된 논문이기 때문에 optimizer와 learning rate가 이렇게 설정되어 있지만 구현에서는 optimizer를 Adam을 사용하고 Learning rate도 고정 값을 사용할 예정.  
Learning rate schedule 구현 코드가 궁금한 사람은 https://github.com/motokimura/yolo_v1_pytorch/blob/master/train_yolo.py 을 참조.
    
batch_size에 관해서도 사용하는 GPU의 메모리에 한계가 있어 16정도로 줄여서 사용할 예정.

# IOU
참고 자료 https://ballentain.tistory.com/12   
   
![image](https://user-images.githubusercontent.com/63130907/124529728-23861980-de46-11eb-88e4-04410a5a222b.png)

![image](https://user-images.githubusercontent.com/63130907/124529737-27b23700-de46-11eb-9ff5-8abc81333413.png)

IOU는 y의 실제 BB와 예측된 y가 얼마나 겹치는지, 또한 이미지에 존재하는 98개의 BB들 중 한 박스가 다른 박스와 얼마나 겹치는지 등에 사용된다.   
   
박스의 크기가 같을 경우, 2/3가 겹칠 경우 값이 0.5가 나오기 때문에 Ground truth(실제 바운딩 박스)와의 IOU threshold를 0.5로 설정하는 경우가 많다.
   
구현을 위해서는 상자 A,B의 x1, x2, y1, y2 좌표들이 필요하다. 

![image](https://user-images.githubusercontent.com/63130907/124530731-0ce0c200-de48-11eb-98bf-bd96aefe10c2.png)
   
   
위의 자료와 같이 교집합이 되는 박스의 좌표는 결국 A, B상자들의 x1,x2,y1,y2 좌표에 의해 정해진다.   

교집합의 x1좌표는 A,B x1 좌표의 최대값, x2 좌표는 A,B의 x2 좌표의 최소값,   
y1 좌표는 A,B y1 좌표의 최대값, y2 좌표는 A,B의 y2 좌표의 최소값이다.   
   
각 상자에 (x2-x1) * (y2-y1)를 이용하여 넓이를 구할 수 있고, IOU의 공식대로 교집합의 넓이 / A의 넓이 + B의 넓이 - 교집합의 넓이를 이용하여 IOU값을 구할 수 있다. 
   
아래와 같이 A,B의 상대위치가 바뀌더라도 위의 교집합 x1, x2, y1 , y2의 계산방식은 달라지지 않는다.
![image](https://user-images.githubusercontent.com/63130907/124530774-1ec26500-de48-11eb-8159-b870c8705a74.png)

# NMS

Yolo v1의 경우 한 이미지를 7 * 7의 grid cell로 나누고 각 cell 별로 2개의 바운딩 박스를 예측하게 되기 때문에 7 * 7 * 2로 총 98개의 바운딩 박스를 예측하게 된다. 하지만 아래와 같이 바운딩 박스가 전부 표시되면 보기가 어렵고 우리가 실제로 원하는 것은 중복이 제거되고 객체를 제대로 잡은 바운딩 박스이기 때문에 NMS를 통해 98개의 바운딩 박스 중 유의미한 것들만 남기는 필터링이 필요하다.

![image](https://user-images.githubusercontent.com/63130907/126566889-dd245839-05ab-4bb8-8b33-796b201f0946.png)

NMS는 간단하게 생각하면 객체가 있을 확률값이 기준 조건보다 높은 바운딩 박스들 중에서 중복박스를 제거하는 것이라고 생각하면 된다. 때문에 먼저 객체가 있을 확률이 정의한 threshold값을 필터링하고, 그 후에 중복 박스를 제거. 중복 박스의 기준은 (더 확률 값이 높은 바운딩 박스와 해당 바운딩 박스의 클래스가 같다) and (다른 바운딩 박스와의 IOU값이 threshold 이상이다)이고, 반대로 이미 존재하는 바운딩 박스 list에서 위의 and 조건을 반대로 적용하여 (클래스가 같지 않다) or (다른 박스와의 IOU값이 threshold보다 작다)인 것들만 남겨두는 식으로 구현할 수 있다.   
   
98개의 바운딩 박스 list에서 객체 존재 확률이 threshold 이상은 박스들만 남겨두고, 객체 존재 확률로 내림차순 sort를 하고, 확률이 가장 높은 박스부터 pop으로 뽑아내며 해당 박스와 클래스가 같지 않거나, 해당 박스와의 IOU값이 threshold보다 작은 박스만 다시 후보군으로 남겨둔다. 앞에서 POP으로 추출한 확률이 가장 높은 바운딩 박스를 NMS list에 추가해준다. 이러한 작업을 반복해서 작업하면 결국 NMS list에는 각각의 객체들에 대해 중복되지 않고 확률값이 제일 높은 바운딩 박스만 남게 된다.

# mAP

