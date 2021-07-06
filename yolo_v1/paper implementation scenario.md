# 참고 자료
https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLO   
https://github.com/shkim960520/YOLO-v1-for-studying   
https://github.com/motokimura/yolo_v1_pytorch   


# basic data information

dataset = https://www.kaggle.com/dataset/734b7bcb7ef13a045cbdd007a3c19874c2586ed0b02b4afc86126e89d00af8d2?select=train.csv


input_image_size = (3 * 448 * 448)   

train_loader_shape = (iteration, batch_size, 3, 448, 488)   
x_shape = (batch_size, 3, 448, 448)   
   
output_size = (batch_size, gird_cell, gird_cell, class_num + Bounding_Box_num * 5)   
-> 5는 (confidence_socre, BB의 중심점 x, BB의 중심점 y, BB의 w, BB의 h)   
grid_cell = 7 * 7, class_num = 20, BB_num = 2   
-> (batch_size, 7, 7, 30)   

# Model Architecture

모델의 구조에 대해서는 논문 내에 상당히 자세히 나와 있음   

![image](https://user-images.githubusercontent.com/63130907/124525241-7a392680-de39-11eb-9d69-f9eb7084d1de.png)

convolution 레이어에 대한 옵션 중 padding 값은 따로 입력해 줘야함.   
전체 크기 유지를 원할 시 -> (kernel_size - 1) // 2   
kernel size가 2 이상인데도 padding 값을 주지 않으면 크기는 kernel size - 1 만큼 줄어듬. ex) 10 * 10 이미지에 3 * 3 kernel을 사용할 경우 전체 크기는 8 * 8이 됨   

convolution layer에서는 이미지의 크기를 인자 값으로 주지는 않음.   

각 covolution layer에는 batch_norm과 LeakyReLU가 적용되므로 편의를 위해 Convolution layer 블럭으로 class를 별도 정의.   
nn.Module를 상속 받은 후에 forward에서 leaykrelu(batch_norm(convolution(x))) 형태로 전개   

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



# NMS


