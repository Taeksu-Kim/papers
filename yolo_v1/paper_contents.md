# You Only Look Once:Unified, Real-Time Object Detection
### 논문 출처 : https://arxiv.org/pdf/1506.02640.pdf

#### ※ 파이썬과 구글 번역기를 기반으로 한 번역입니다.

# Abstract

object detection에 대한 새로운 접근 방식인 YOLO를 소개합니다.   
   
object detection에 대한 선행 연구는 classifier를 사용하여 감지를 수행합니다. 대신 object detection를 공간적으로 분리된 bounding box 및 관련 클래스 확률에 대한 회귀 문제로 구성합니다. 단일 신경망은 한 번의 평가로 전체 이미지에서 직접 bounding box와 클래스 확률을 예측합니다. 전체 탐지 파이프라인이 단일 네트워크이므로 탐지 성능에 대해 직접 종단 간 최적화할 수 있습니다.   
   
통합 아키텍처는 매우 빠릅니다. 우리의 기본 YOLO 모델은 초당 45 프레임으로 실시간으로 이미지를 처리합니다. 더 작은 네트워크 버전인 Fast YOLO는 놀라운 초당 155 프레임을 처리하는 동시에 다른 실시간 탐지기의 mAP를 두 배로 늘립니다. 최첨단 탐지 시스템에 비해 YOLO는 더 많은 localization error를 발생시키지만 백그라운드에서 이 탐을 예측할 가능성이 적습니다. 마지막으로 YOLO는 object의 매우 일반적인 표현을 배웁니다. 자연 이미지에서 아트 워크와 같은 다른 도메인으로 일반화할 때 DPM 및 R-CNN을 포함한 다른 감지 방법보다 성능이 뛰어납니다.

# 1. Introduction

인간은 이미지를 한눈에 보고 이미지에 있는 개체, 개체 위치 및 상호 작용 방식을 즉시 알 수 있습니다. 인간의 시각 시스템은 빠르고 정확하기 때문에 의식적인 사고 없이 운전과 같은 복잡한 작업을 수행할 수 있습니다. object detection를 위한 빠르고 정확한 알고리즘을 통해 컴퓨터는 특수 센서 없이 자동차를 운전할 수 있고, 보조 장치가 실시간 장면 정보를 인간 사용자에게 전달할 수 있으며, 범용 반응 형 로봇 시스템의 잠재력을 잠금 해제할 수 있습니다.   
   
현재 감지 시스템은 classifier를 재사용하여 감지를 수행합니다. 물체를 감지하기 위해 이러한 시스템은 해당 물체에 대한 classifier를 가져와 테스트 이미지의 다양한 위치와 배율에서 평가합니다. 변형 가능한 부품 모델 (DPM)과 같은 시스템은 classifier가 전체 이미지에 걸쳐 균일 한 간격의 위치에서 실행되는 슬라이딩 윈도 접근 방식을 사용합니다 [10].

![image](https://user-images.githubusercontent.com/63130907/124403019-df741580-dd6e-11eb-9709-ec98b5395d29.png)

R-CNN과 같은 최근의 접근 방식은 영역 제안 방법을 사용하여 먼저 이미지에서 잠재적 bounding box를 생성한 다음 이러한 제안된 상자에서 classifier를 실행합니다. 분류 후에는 bounding box를 다듬고, 중복 탐지를 제거하고, 장면의 다른 개체를 기반으로 상자를 다시 채점하기 위해 사후 처리가 사용됩니다 [13]. 이러한 복잡한 파이프라인은 각 개별 구성 요소를 개별적으로 학습해야 하기 때문에 느리고 최적화하기 어렵습니다.   
   
object detection를 이미지 픽셀에서 bounding box 좌표 및 클래스 확률로 곧바로 단일 회귀 문제로 재구성합니다. 우리 시스템을 사용하면 이미지를 한 번만 (YOLO) 보고 어떤 물체가 있고 어디에 있는지 예측할 수 있습니다.   
   
YOLO는 매우 간단합니다. 그림 1을 참조하십시오. 단일 convolution 네트워크는 동시에 여러 bounding box와 해당 상자에 대한 클래스 확률을 예측합니다.   
   
YOLO는 전체 이미지를 학습하고 탐지 성능을 직접 최적화합니다. 이 통합 모델은 기존의 object detection 방법에 비해 몇 가지 이점이 있습니다.   
   
첫째, YOLO는 매우 빠릅니다. 회귀 문제로 감지를 프레임 화하므로 복잡한 파이프라인이 필요하지 않습니다.   
   
감지를 예측하기 위해 테스트 시간에 새 이미지에 신경망을 실행하기 만하면 됩니다. 당사의 기본 네트워크는 Titan X GPU에서 일괄 처리 없이 초당 45 프레임으로 실행되며 빠른 버전은 150fps 이상으로 실행됩니다. 즉, 지연 시간이 25 밀리 초 미만인 스트리밍 비디오를 실시간으로 처리할 수 있습니다. 또한 YOLO는 다른 실시간 시스템의 평균 평균 정밀도의 두 배 이상을 달성합니다. 웹캠에서 실시간으로 실행되는 시스템의 데모는 프로젝트 웹 페이지 http://pjreddie.com/yolo/ 를 참조하십시오.

둘째, YOLO는 예측할 때 이미지에 대해 전 세계적으로 추론합니다. 슬라이딩 윈도 및 영역 제안 기반 기술과 달리 YOLO는 학습 및 테스트 시간 동안 전체 이미지를 확인하므로 클래스와 클래스 모양에 대한 컨텍스트 정보를 암시 적으로 인코딩합니다.   
   
빠른 R-CNN, 상위 감지 방법 [14]은 더 큰 컨텍스트를 볼 수 없기 때문에 이미지의 배경 패치를 개체로 착각합니다. YOLO는 Fast R-CNN에 비해 백그라운드 오류 수를 절반 이하로 만듭니다.   
   
셋째, YOLO는 object의 일반화 가능한 표현을 학습합니다. 자연 이미지에 대한 교육을 받고 아트 워크에서 테스트할 때 YOLO는 DPM 및 R-CNN과 같은 최고의 탐지 방법보다 큰 차이가 있습니다. YOLO는 매우 일반화 가능하기 때문에 새로운 도메인이나 예상치 못한 입력에 적용될 때 분해될 가능성이 적습니다.   
   
YOLO는 정확도면에서 여전히 최첨단 탐지 시스템에 뒤처져 있습니다. 이미지에서 물체를 빠르게 식별할 수 있지만 일부 물체, 특히 작은 물체를 정확하게 위치시키는 데 어려움이 있습니다. 우리는 실험에서 이러한 장단점을 자세히 조사합니다.   
   
모든 교육 및 테스트 코드는 오픈 소스입니다. 사전 훈련된 다양한 모델도 다운로드할 수 있습니다.

# 2. Unified Detection

object detection의 개별 구성 요소를 단일 신경망으로 통합합니다. 우리의 네트워크는 전체 이미지의 특징을 사용하여 각 bounding box를 예측합니다. 또한 이미지에 대한 모든 클래스의 모든 bounding box를 동시에 예측합니다. 이것은 우리의 네트워크가 전체 이미지와 이미지의 모든 개체에 대해 전 세계적으로 이유를 설명한다는 것을 의미합니다.   
   
YOLO 디자인은 높은 평균 정밀도를 유지하면서 종단 간 교육 및 실시간 속도를 가능하게 합니다.   
   
우리 시스템은 입력 이미지를 S × S 그리드로 나눕니다.   
   
object의 중심이 그리드 셀에 들어가면 해당 그리드 셀 이 해당 object를 감지합니다.   
   
각 그리드 셀은 B bounding box와 해당 상자에 대한 신뢰 점수를 예측합니다. 이러한 신뢰도 점수는 모델이 상자에 개체가 포함되어 있다는 확신과 상자가 예측하는 것이 얼마나 정확하다고 생각하는지를 반영합니다. 공식적으로 우리는 신뢰를 Pr (Object) ∗ IOUtruth pred로 정의합니다. 해당 셀에 개체가 없으면 신뢰도 점수는 0이어야 합니다. 그렇지 않으면 신뢰도 점수가 예측 상자와 실측값 사이의 IOU (교차 교차점)와 같기를 원합니다.   
   
각 bounding box는 x, y, w, h 및 신뢰도의 5 가지 예측으로 구성됩니다. (x, y) 좌표는 그리드 셀의 경계를 기준으로 상자의 중심을 나타냅니다. 너비와 높이는 전체 이미지를 기준으로 예측됩니다. 마지막으로 신뢰도 예측은 예측된 상자와 모든 지상 진실 상자 사이의 IOU를 나타냅니다.   
   
각 그리드 셀은 또한 C 조건부 클래스 확률 Pr (Classi | Object)을 예측합니다. 이러한 확률은 object를 포함하는 그리드 셀에서 조건이 지정됩니다. 상자 B의 수에 관계없이 그리드 설당 하나의 클래스 확률 세트 만 예측합니다.

테스트 시간에 조건부 클래스 확률과 개별 상자 신뢰도 예측을 곱하여 각 상자에 대한 클래스 별 신뢰도 점수를 제공합니다. 이러한 점수는 해당 클래스가 상자에 나타날 확률과 예측 상자가 개체에 얼마나 잘 맞는지 모두를 인코딩합니다.

![image](https://user-images.githubusercontent.com/63130907/124411343-4dc4d200-dd87-11eb-9eb4-e2db66996aae.png)

![image](https://user-images.githubusercontent.com/63130907/124411396-66cd8300-dd87-11eb-944e-c84c81608ce1.png)

PASCAL VOC에서 YOLO를 평가하기 위해 S = 7, B = 2를 사용합니다. PASCAL VOC에는 20 개의 레이블이 지정된 클래스가 있으므로 C = 20입니다.   
   
우리의 최종 예측은 7 × 7 × 30 tensor입니다.

# 2.1. Network Design

우리는 이 모델을 convolution 신경망으로 구현하고 PASCAL VOC 검출 데이터 세트에서 평가합니다 [9]. 네트워크의 초기 컨 블루션 계층은 이미지에서 특징을 추출하는 반면 완전 연결 계층은 출력 확률과 좌표를 예측합니다.   
   
우리의 네트워크 아키텍처는 이미지 분류를 위한 GoogLeNet 모델에서 영감을 받았습니다 [34]. 우리의 네트워크는 24 개의 컨 블루션 레이어와 2 개의 완전히 연결된 레이어로 구성됩니다.   
   
GoogLeNet에서 사용하는 시작 모듈 대신 Lin et al [22]과 유사하게 1 × 1 축소 레이어와 3 × 3 convolution 레이어를 사용합니다. 전체 네트워크는 그림 3에 나와 있습니다.   
   
또한 빠른 object detection의 경계를 넓히기 위해 설계된 YOLO의 빠른 버전을 교육합니다. Fast YOLO는 더 적은 수의 convolution 레이어 (24 개 대신 9 개)와 해당 레이어의 필터 수가 적은 신경망을 사용합니다. 네트워크의 크기를 제외하고 모든 훈련 및 테스트 매개 변수는 YOLO와 Fast YOLO 간에 동일합니다.   
   
네트워크의 최종 출력은 7 × 7 × 30 tensor 예측입니다.

![image](https://user-images.githubusercontent.com/63130907/124411674-f3784100-dd87-11eb-8aa5-b254219289f3.png)

# 2.2. Training

ImageNet 1000 급 경쟁 데이터 세트 [30]에서 convolution 레이어를 사전 훈련합니다. 사전 훈련을 위해 그림 3의 처음 20 개의 컨 블루션 레이어를 사용한 다음 평균 풀링 레이어와 완전 연결 레이어를 사용합니다. 우리는 이 네트워크를 약 1 주일 동안 훈련하고 ImageNet 2012 검증 세트에서 88 %의 단일 자르기 상위 5 정확도를 달성했습니다. 이는 Caffe의 Model Zoo [24]의 GoogLeNet 모델과 비슷합니다. 우리는 모든 훈련과 추론에 Darknet 프레임 워크를 사용합니다 [26].   
   
그런 다음 모델을 변환하여 감지를 수행합니다. Ren et al. 컨 블루션 계층과 연결 계층을 모두 사전 훈련된 네트워크에 추가하면 성능이 향상될 수 있음을 보여줍니다 [29].   
   
예제에 따라 무작위로 초기화된 가중치를 사용하여 4 개의 컨 블루션 레이어와 2 개의 완전 연결 레이어를 추가합니다. 탐지에는 종종 세밀한 시각 정보가 필요하므로 네트워크의 입력 해상도를 224 × 224에서 448 × 448로 높입니다.   
   
마지막 레이어는 클래스 확률과 bounding box 좌표를 모두 예측합니다. bounding box 너비와 높이를 이미지 너비와 높이로 정규화하여 0과 1 사이가 되도록 합니다. bounding box x 및 y 좌표를 특정 그리드 셀 위치의 오프셋으로 매개 변수화하여 0과 1 사이의 경계가 되도록 합니다. .   
   
우리는 최종 레이어에 선형 활성화 함수를 사용하고 다른 모든 레이어는 다음과 같은 leaky rectified linear activation(역자 - Leaky ReLU)를 사용합니다.

![image](https://user-images.githubusercontent.com/63130907/124411935-713c4c80-dd88-11eb-82d4-3303b3782e34.png)

loss function의 출력에서 sum-squared error를 최적화합니다.

![image](https://user-images.githubusercontent.com/63130907/124412156-d98b2e00-dd88-11eb-927b-533ea927dee1.png)

여기서 1 obj i는 object가 셀 i에 나타나는지를 나타내고 1 obj ij는 셀 i의 j 번째 bounding box 예측자가 해당 예측에 대해 "책임이 있음"을 나타냅니다.   
   
loss function는 object가 해당 그리드 셀에 있는 경우에만 분류 오류에 페널티를 줍니다 (따라서 앞에서 설명한 조건부 클래스 확률). 또한 해당 예측자가 Ground Truth Box에 대해 "책임이 있는"경우 (즉, 해당 그리드 셀에서 예측 자의 IOU가 가장 높은 경우) bounding box 좌표 오류에만 페널티를 줍니다.   
   
PASCAL VOC 2007 및 2012의 교육 및 검증 데이터 세트에 대해 약 135 epoch 동안 네트워크를 교육합니다. 2012 년에 테스트할 때 교육용 VOC 2007 테스트 데이터도 포함합니다. 훈련 내내 우리는 배치 크기 64, 모멘텀 0.9 및 감쇠 0.0005를 사용합니다.   
   
학습률 일정은 다음과 같습니다. 첫 번째 시대에 대해 학습률을 10−3에서 10−2로 천천히 올립니다.   
   
높은 학습률에서 시작하면 불안정한 기울기로 인해 모델이 발산하는 경우가 많습니다. 우리는 75 epoch에 대해 10-2, 30 epoch에 대해 10-3, 마지막으로 30 epoch에 대해 10-4로 훈련을 계속합니다.   
   
과적 합을 방지하기 위해 드롭아웃과 광범위한 데이터 보강을 사용합니다. 첫 번째 연결 레이어 이후에 비율 = .5 인 드롭아웃 레이어는 레이어 간의 공동 적응을 방지합니다 [18].   
   
데이터 증가를 위해 원본 이미지 크기의 최대 20 %까지 무작위 크기 조정 및 변환을 도입합니다. 또한 HSV 색상 공간에서 최대 1.5 배까지 이미지의 노출과 채도를 무작위로 조정합니다.

# 2.3. Inference

훈련에서와 마찬가지로 테스트 이미지에 대한 탐지를 예측하려면 네트워크 평가가 하나만 필요합니다. PASCAL VOC에서 네트워크는 이미지 당 98 개의 bounding box와 각 상자에 대한 클래스 확률을 예측합니다. YOLO는 classifier 기반 방법과 달리 단일 네트워크 평가 만 필요하기 때문에 테스트 시간에 매우 빠릅니다.   
   
그리드 디자인은 bounding box 예측에서 공간 다양성을 적용합니다. 종종 object가 어느 그리드 셀에 속하는지 명확하고 네트워크는 각 object에 대해 하나의 상자 만 예측합니다. 그러나 일부 큰 개체 또는 여러 셀의 경계 근처에 있는 개체는 여러 셀에 의해 잘 지역화될 수 있습니다. 비 최대 억제를 사용하여 이러한 다중 탐지를 수정할 수 있습니다. R-CNN 또는 DPM의 경우처럼 성능에 중요하지는 않지만 최대가 아닌 억제는 mAP에서 2-3%를 추가합니다.
  
# 2.4. Limitations of YOLO

YOLO는 각 그리드 셀 이 두 개의 상자 만 예측하고 하나의 클래스 만 가질 수 있으므로 bounding box 예측에 강력한 공간 제약을 부과합니다. 이 공간적 제약은 모델이 예측할 수 있는 주변 물체의 수를 제한합니다. 우리 모델은 새무리와 같이 그룹으로 나타나는 작은 물체로 어려움을 겪습니다.   
   
우리 모델은 데이터에서 bounding box를 예측하는 방법을 학습하므로 새롭거나 비정상적인 종횡비 또는 구성의 개체로 일반화하는 데 어려움을 겪습니다. 또한 아키텍처에는 입력 이미지에서 여러 다운 샘플링 레이어가 있으므로 bounding box를 예측하기 위해 상대적으로 거친 기능을 사용합니다.   
   
마지막으로, 탐지 성능에 근접한 loss function에 대해 학습하는 동안 loss function는 작은 bounding box와 큰 bounding box의 오류를 동일하게 처리합니다. 큰 상자의 작은 오류는 일반적으로 무해하지만 작은 상자의 작은 오류는 IOU에 훨씬 더 큰 영향을 미칩니다.   
   
우리의 주요 오류 원인은 잘못된 localization입니다.

# 3. Comparison to Other Detection Systems

object detection는 컴퓨터 비전의 핵심 문제입니다.   
   
탐지 파이프라인은 일반적으로 입력 이미지 (Haar [25], SIFT [23], HOG [4], convolutional features [6])에서 강력한 기능 집합을 추출하는 것으로 시작합니다. 그런 다음 classifier [36, 21, 13, 10] 또는 지역화 [1, 32]를 사용하여 특징 공간에서 object를 식별합니다. 이러한 classifier 또는 로컬 라이저는 전체 이미지에 대해 슬라이딩 윈도 방식으로 실행되거나 이미지의 일부 영역에서 실행됩니다 [35, 15, 39].   
   
우리는 YOLO 탐지 시스템을 몇 가지 주요 탐지 프레임 워크와 비교하여 주요 유사점과 차이점을 강조합니다.   
   
변형 가능한 부품 모델. 변형 가능한 부품 모델 (DPM)은 슬라이딩 윈도 접근 방식을 사용하여 물체를 감지합니다 [10]. DPM은 분리된 파이프라인을 사용하여 정적 기능을 추출하고, 영역을 분류하고, 고득점 영역에 대한 bounding box를 예측하는 등의 작업을 수행합니다. 우리 시스템은 이러한 모든 분리된 부분을 단일 convolution 신경망으로 대체합니다. 네트워크는 특성 추출, bounding box 예측, 비 최대 억제 및 상황 추론을 모두 동시에 수행합니다. 네트워크는 정적 기능 대신 기능을 인라인으로 훈련하고 탐지 작업을 위해 최적화합니다. 통합 아키텍처는 DPM보다 더 빠르고 정확한 모델로 이어집니다.   
   
R-CNN. R-CNN과 그 변형은 이미지에서 object를 찾기 위해 슬라이딩 윈도 대신 영역 제안을 사용합니다. 선택적 검색 [35]은 잠재적인 bounding box를 생성하고, convolution 네트워크는 기능을 추출하고, SVM은 상자에 점수를 매기고, 선형 모델은 bounding box를 조정하고, 비 최대 억제는 중복 탐지를 제거합니다. 이 복잡한 파이프라인의 각 단계는 독립적으로 정밀하게 조정되어야 하며 결과 시스템은 매우 느려 테스트 시간에 이미지 당 40 초 이상 걸립니다 [14].   
   
YOLO는 R-CNN과 몇 가지 유사점을 공유합니다. 각 그리드 셀은 잠재적인 bounding box를 제안하고 컨 블루션 기능을 사용하여 해당 상자에 점수를 매 깁니다. 그러나 우리 시스템은 그리드 셀 제안에 공간 제약을 두어 동일한 물체에 대한 다중 탐지를 완화합니다. 또한 우리 시스템은 선택 검색의 약 2000 개에 비해 이미지 당 98 개에 불과한 훨씬 적은 수의 bounding box를 제안합니다.   
   
마지막으로 우리 시스템은 이러한 개별 구성 요소를 공동으로 최적화된 단일 모델로 결합합니다.   
   
기타 Fast Detector Fast and Faster R-CNN은 계산을 공유하고 신경망을 사용하여 선택적 검색 대신 지역을 제안함으로써 R-CNN 프레임 워크의 속도를 높이는 데 중점을 둡니다 [14] [28]. R-CNN보다 속도와 정확도가 향상되었지만 둘 다 여전히 실시간 성능에 미치지 못합니다.   
   
많은 연구 노력은 DPM 파이프라인의 속도를 높이는 데 중점을 둡니다 [31] [38] [5]. HOG 계산 속도를 높이고 캐스케이드를 사용하며 계산을 GPU로 푸시 합니다. 그러나 실제로는 30Hz DPM [31]만이 실시간으로 실행됩니다.   
   
대규모 탐지 파이프라인의 개별 구성 요소를 최적화하는 대신 YOLO는 파이프라인을 완전히 버리고 설계 상 빠릅니다.   
   
얼굴이나 사람과 같은 단일 클래스에 대한 감지기는 훨씬 적은 편차를 처리해야 하기 때문에 고도로 최적화될 수 있습니다 [37]. YOLO는 다양한 물체를 동시에 감지하는 방법을 배우는 범용 감지기입니다.   
   
딥 멀티 박스. R-CNN과 달리 Szegedy et al. 선택적 검색을 사용하는 대신 관심 영역 [8]을 예측하도록 컨 블루션 신경망을 훈련시킵니다. MultiBox는 신뢰도 예측을 단일 클래스 예측으로 대체하여 단일 object detection를 수행할 수도 있습니다. 그러나 MultiBox는 일반적인 object detection를 수행할 수 없으며 여전히 더 큰 감지 파이프라인의 일부일 뿐이므로 추가 이미지 패치 분류가 필요합니다. YOLO와 MultiBox는 모두 convolution 네트워크를 사용하여 이미지의 bounding box를 예측하지만 YOLO는 완전한 감지 시스템입니다.   
   
OverFeat. Sermanet et al. 컨 블루션 신경망을 훈련시켜 지역화를 수행하고 탐지를 수행하도록 지역화를 적용합니다 [32]. OverFeat는 슬라이딩 윈도 감지를 효율적으로 수행하지만 여전히 분리된 시스템입니다. OverFeat는 탐지 성능이 아닌 localization를 위해 최적화합니다.   
   
DPM과 마찬가지로 로컬 라이저는 예측 시 로컬 정보 만 볼 수 있습니다. OverFeat는 글로벌 컨텍스트를 추론할 수 없으므로 일관된 감지를 생성하기 위해 상당한 후 처리가 필요합니다.   
   
MultiGrasp. 우리의 작업은 Redmon 등 [27]의 파악 감지 작업과 유사합니다. bounding box 예측에 대한 우리의 그리드 접근 방식은 파악에 대한 회귀를 위한 MultiGrasp 시스템을 기반으로 합니다. 그러나 파악 감지는 object detection보다 훨씬 간단한 작업입니다. MultiGrasp은 하나의 개체를 포함하는 이미지에 대해 파악 가능한 단일 영역 만 예측하면 됩니다. 물체의 크기, 위치 또는 경계를 추정하거나 등급을 예측할 필요가 없으며 잡기에 적합한 영역 만 찾습니다. YOLO는 이미지에서 여러 클래스의 여러 개체에 대한 bounding box와 클래스 확률을 모두 예측합니다.

# 4. Experiments

먼저 YOLO를 PASCAL VOC 2007의 다른 실시간 탐지 시스템과 비교합니다. YOLO와 R-CNN 변형의 차이점을 이해하기 위해 YOLO와 Fast R-CNN에서 만든 VOC 2007의 오류를 살펴봅니다. R-CNN [14]. 다양한 오류 프로필을 기반으로 YOLO를 사용하여 Fast R-CNN 탐지를 다시 채점하고 백그라운드 오 탐지로 인한 오류를 줄여 성능을 크게 향상시킬 수 있음을 보여줍니다. 또한 VOC 2012 결과를 제시하고 mAP를 현재의 최신 방법과 비교합니다. 마지막으로, 우리는 YOLO가 두 아트 워크 데이터 세트에서 다른 탐지기보다 새로운 도메인으로 일반화된다는 것을 보여줍니다.

# 4.1. Comparison to Other Real-Time Systems

object detection에 대한 많은 연구 노력은 표준 감지 파이프라인을 빠르게 만드는 데 중점을 둡니다. [5] [38] [31] [14] [17] [28] 그러나 Sadeghi et al. 실제로 실시간으로 실행되는 감지 시스템을 생성합니다 (초당 30 프레임 이상) [31]. YOLO를 30Hz 또는 100Hz에서 실행되는 DPM의 GPU 구현과 비교합니다.   
   
다른 노력은 실시간 이정표에 도달하지 못하지만 상대적인 mAP와 속도를 비교하여 object detection 시스템에서 사용할 수 있는 정확도-성능 균형을 조사합니다.   
   
Fast YOLO는 PASCAL에서 가장 빠른 object detection 방법입니다. 우리가 아는 한, 현존하는 가장 빠른 물체 탐지기입니다. 52.7%의 mAP로 실시간 탐지에 대한 이전 작업보다 두 배 이상 정확합니다. YOLO는 실시간 성능을 유지하면서 mAP를 63.4%로 높였습니다.   
   
우리는 또한 VGG-16을 사용하여 YOLO를 훈련합니다. 이 모델은 YOLO보다 더 정확하지만 상당히 느립니다. VGG-16에 의존하는 다른 감지 시스템과 비교하는 데 유용하지만 실시간보다 느리기 때문에 나머지 문서에서는 더 빠른 모델에 중점을 둡니다.   
   
가장 빠른 DPM은 많은 mAP를 희생하지 않고 효과적으로 DPM 속도를 높이지만 여전히 실시간 성능을 2 배 정도 떨어뜨립니다 [38]. 또한 신경망 접근 방식에 비해 DPM의 탐지 정확도가 상대적으로 낮기 때문에 제한됩니다.   
   
R-CNN 마이너스 R는 선택적 검색을 정적 bounding box 제안으로 대체합니다 [20]. R-CNN보다 훨씬 빠르지 만 여전히 실시간에 미치지 못하며 좋은 제안이 없기 때문에 상당한 정확도가 떨어집니다.   

![image](https://user-images.githubusercontent.com/63130907/124413048-d2651f80-dd8a-11eb-853c-569c1b19a7af.png)

   
Fast R-CNN은 R-CNN의 분류 단계를 가속화하지만 bounding box 제안을 생성하는 데 이미지 당 약 2 초가 걸릴 수 있는 선택적 검색에 여전히 의존합니다.   
   
따라서 mAP가 높지만 0.5fps에서는 여전히 실시간과는 거리가 멀습니다.   
   
최근 Faster R-CNN은 Szegedy et al. 와 유사한 bounding box를 제안하기 위해 선택적 검색을 신경망으로 대체합니다. [8] 테스트에서 가장 정확한 모델은 7fps를 달성하고 더 작고 덜 정확한 모델은 18fps에서 실행됩니다. Faster R-CNN의 VGG-16 버전은 10mAP 높지만 YOLO보다 6 배 느립니다. ZeilerFergus Faster R-CNN은 YOLO보다 2.5 배 느리지 만 정확도도 떨어집니다.   

# 4.2. VOC 2007 Error Analysis

YOLO와 최첨단 검출기의 차이점을 자세히 조사하기 위해 VOC 2007에 대한 자세한 결과 분석을 살펴봅니다. Fast R-CNN은 PASCAL에서 가장 성능이 뛰어난 검출기 중 하나이기 때문에 YOLO와 Fast RCNN을 비교합니다. 탐지는 공개적으로 사용 가능합니다.   
   
우리는 Hoiem 외의 방법론과 도구를 사용합니다. 테스트 시간에 각 범주에 대해 해당 범주에 대한 상위 N 개의 예측을 살펴봅니다. 각 예측은 정확하거나 오류 유형에 따라 분류됩니다.   

![image](https://user-images.githubusercontent.com/63130907/124413446-a0a08880-dd8b-11eb-8c58-5723925416c8.png)

그림 4는 20 개 클래스 전체에서 평균화된 각 오류 유형의 분석을 보여줍니다.   
   
YOLO는 개체를 올바르게 localization 하는 데 어려움을 겪고 있습니다. localization error는 다른 모든 소스를 합친 것보다 YOLO의 오류를 더 많이 설명합니다. Fast R-CNN은 localization error는 훨씬 적지 만 백그라운드 오류는 훨씬 더 많습니다. 상위 탐지 항목 중 13.6%는 개체가 포함되지 않은 오 탐지입니다. Fast R-CNN은 YOLO보다 백그라운드 탐지를 예측할 가능성이 거의 3 배 더 높습니다.

# 4.3. Combining Fast R-CNN and YOLO

OLO는 Fast R-CNN보다 배경 실수가 훨씬 적습니다. YOLO를 사용하여 Fast R-CNN에서 백그라운드 감지를 제거하면 성능이 크게 향상됩니다. R-CNN이 예측하는 모든 bounding box에 대해 YOLO가 유사한 상자를 예측하는지 확인합니다. 만약 그렇다면, 우리는 YOLO가 예측 한 확률과 두 상자 사이의 겹침을 기반으로 예측을 향상시킵니다.

![image](https://user-images.githubusercontent.com/63130907/124413646-0725a680-dd8c-11eb-9df8-3cc85aba2493.png)

![image](https://user-images.githubusercontent.com/63130907/124413702-202e5780-dd8c-11eb-9e36-be7bdce69481.png)

최고의 Fast R-CNN 모델은 VOC 2007 테스트 세트에서 71.8%의 mAP를 달성합니다. YOLO와 결합하면 mAP가 3.2% 증가한 75.0%입니다. 또한 상위 Fast R-CNN 모델을 다른 여러 버전의 Fast R-CNN과 결합해 보았습니다. 이러한 앙상블은. 3에서. 6% 사이의 mAP를 약간 증가시켰습니다. 자세한 내용은 표 2를 참조하십시오.   
   
YOLO의 향상은 다른 버전의 Fast R-CNN을 결합하여 얻을 수 있는 이점이 거의 없기 때문에 단순히 모델 통합의 부산물이 아닙니다. 오히려 YOLO가 테스트 시간에 여러 종류의 실수를 저지르기 때문에 Fast R-CNN의 성능을 높이는 데 매우 효과적입니다.   
   
안타깝게도 이 조합은 각 모델을 개별적으로 실행한 다음 결과를 결합하기 때문에 YOLO의 속도의 이점을 얻지 못합니다. 그러나 YOLO는 매우 빠르기 때문에 Fast R-CNN에 비해 상당한 계산 시간을 추가하지 않습니다.   

# 4.4. VOC 2012 Results

VOC 2012 테스트 세트에서 YOLO는 57.9% mAP를 기록했습니다.   
   
이것은 VGG-16을 사용하는 원래 R-CNN에 더 가까운 현재의 최신 상태보다 낮습니다. 표 3을 참조하십시오. 우리 시스템은 가장 가까운 경쟁자에 비해 작은 물체로 어려움을 겪습니다. 병, 양, TV / 모니터와 같은 카테고리에서 YOLO는 R-CNN 또는 Feature Edit보다 8-10% 낮은 점수를 받았습니다. 그러나 고양이 및 기차와 같은 다른 범주에서는 YOLO가 더 높은 성능을 달성합니다.   
   
결합된 Fast R-CNN + YOLO 모델은 가장 성능이 좋은 탐지 방법 중 하나입니다. Fast R-CNN은 YOLO와의 조합에서 2.3% 향상되어 공개 순위표에서 5 단계 상승했습니다.

# 4.5. Generalizability: Person Detection in Artwork

object detection를 위한 학술 데이터 세트는 동일한 분포에서 훈련 및 테스트 데이터를 가져옵니다. 실제 애플리케이션에서는 가능한 모든 사용 사례를 예측하기가 어렵고 테스트 데이터는 시스템이 이전에 본 것과 다를 수 있습니다 [3]. 우리는 YOLO를 Picasso Dataset [12]과 People-Art Dataset [3]의 다른 감지 시스템과 비교합니다.   
   
그림 5는 YOLO와 다른 탐지 방법 간의 비교 성능을 보여줍니다. 참고로 모든 모델이 VOC 2007 데이터로만 훈련된 사람에게 VOC 2007 감지 AP를 제공합니다. Picasso에서 모델은 VOC 2012에서 교육을 받고 People-Art에서는 VOC 2010에서 교육을 받습니다.   
   
R-CNN은 VOC 2007에서 높은 AP를 가지고 있습니다. 그러나 R-CNN은 아트웍에 적용될 때 상당히 떨어집니다. R-CNN은 자연 이미지에 맞게 조정된 bounding box 제안을 위해 선택적 검색을 사용합니다. R-CNN의 classifier 단계는 작은 지역 만보고 좋은 제안이 필요합니다.   
   
DPM은 아트웍에 적용될 때 AP를 잘 유지합니다.   
   
이전 작업에서는 DPM이 개체의 모양과 레이아웃에 대한 강력한 공간 모델을 가지고 있기 때문에 성능이 좋다는 이론을 세웠습니다.   
   
DPM은 R-CNN 만큼 저하되지 않지만 더 낮은 AP에서 시작됩니다.   
   
YOLO는 VOC 2007에서 좋은 성능을 가지고 있으며 아트웍에 적용될 때 AP가 다른 방법보다 덜 저하됩니다.   
   
DPM과 마찬가지로 YOLO는 개체의 크기와 모양은 물론 개체 간의 관계와 개체가 일반적으로 나타나는 위치를 모델링 합니다. 아트 워크와 자연 이미지는 픽셀 수준에서 매우 다르지만 물체의 크기와 모양이 비슷하기 때문에 YOLO는 여전히 좋은 bounding box와 탐지를 예측할 수 있습니다. 

# 5. Real-Time Detection In The Wild

YOLO는 빠르고 정확한 object detection 기로 컴퓨터 비전 애플리케이션에 이상적입니다. YOLO를 웹캠에 연결하고 카메라에서 이미지를 가져오고 탐지를 표시하는 시간을 포함하여 실시간 성능을 유지하는지 확인합니다.   
   
결과 시스템은 상호 작용하고 매력적입니다. YOLO는 이미지를 개별적으로 처리하지만 웹캠에 연결하면 추적 시스템처럼 작동하여 물체가 움직이고 모양이 변할 때이를 감지합니다. 시스템 데모와 소스 코드는 프로젝트 웹 사이트 http://pjreddie.com/yolo/ 에서 찾을 수 있습니다.

![image](https://user-images.githubusercontent.com/63130907/124415247-6b963500-dd8f-11eb-83e4-1759e72e90d9.png)

![image](https://user-images.githubusercontent.com/63130907/124415287-84064f80-dd8f-11eb-9cd2-f34c52745137.png)

# 6. Conclusion

object detection를 위한 통합 모델 YOLO를 소개합니다. 우리의 모델은 구성이 간단하고 전체 이미지에서 직접 학습할 수 있습니다. classifier 기반 접근 방식과 달리 YOLO는 탐지 성능에 직접적으로 해당하는 loss function에 대해 훈련되고 전체 모델은 공동으로 훈련됩니다.   
   
Fast YOLO는 문헌에서 가장 빠른 범용 물체 탐지기이며 YOLO는 실시간 물체 탐지에서 최첨단 기술을 추진합니다. YOLO는 또한 새로운 도메인으로 잘 일반화되어 빠르고 강력한 object detection에 의존하는 애플리케이션에 이상적입니다.   
   
감사의 말 : 이 작업은 ONR N00014-13-1-0720, NSF IIS-1338054 및 The Allen Distinguished Investigator Award에 의해 부분적으로 지원됩니다. 










