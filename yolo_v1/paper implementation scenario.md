# Model Architecture

모델의 구조에 대해서는 논문 내에 상당히 자세히 나와 있음

컨볼루션 레이어에 대한 옵션 중 padding 값은 따로 입력해 줘야함.
전체 크기 유지를 원할 시 -> (kernel_size - 1) // 2
kernel size가 2 이상인데도 padding 값을 주지 않으면 크기는 kernel size - 1 만큼 줄어듬. ex) 10 * 10 이미지에 3 * 3 kernel을 사용할 경우 전체 크기는 8 * 8이 됨

활성함수 = LeakyReLU(0.1)
FC레이어의 dropout = 0.5


하이퍼 파라미터 메모
weight_decay = 0.0005
batch_size = 64
epochs = 135
learning rate = schedule로 처리 -> 첫 30 epochs는 e-3, 그 다음 75 epochs는 e-2, 마지막 30 epochs는 e-4  
