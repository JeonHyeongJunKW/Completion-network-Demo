# Completion-network-Demo
이미지 채우기 네트워크 구현입니다.

## 목표 : Globally and Locally Consistent Image Completion(2017)

데이터셋은 256*256의 크기의 데이터셋을 사용하였고, 랜덤리사이즈 크로핑을 적용하였습니다. 
데이터셋의 사이즈는 10000장 정도로 줄여서 사용하였습니다. Dataset.py 내부의 코드를 수정해서 읽어오는 경로를 다르게 해야합니다.

---
## 최종 모델의 출력 이미지 샘플 
---
## 복잡한 구조의 이미지의 경우
- 학습 시간 및 데이터셋 부족으로 복잡한 물체를 표현하지는 못한다.
- 색상의 일관성이 없고, 엣지정보가 많이 사라진다.

(원본, 마스킹된 이미지, 합성된 이미지)

![image](https://user-images.githubusercontent.com/63538314/147930216-83f47732-d568-4b5c-81a1-b611186d78ba.jpg)
![mask image3429](https://user-images.githubusercontent.com/63538314/147930451-17d1499e-84a7-4252-b46d-93b8b3531126.jpg)
![fake image3429](https://user-images.githubusercontent.com/63538314/147930485-52f7a1e4-4c79-4091-b8d0-a55f1aa0ff40.jpg)

## 단순한 구조의 이미지의 경우
- 단순한 경우에는 제법 원본과 비슷한 이미지를 형성한다.

(원본, 마스킹된 이미지, 합성된 이미지)

![real image804](https://user-images.githubusercontent.com/63538314/147931211-9cfd41c8-662e-4bd1-a513-14ed655efe94.jpg)
![mask image804](https://user-images.githubusercontent.com/63538314/147931219-f19310dc-c24e-47bd-8e12-a53eefd0ee1c.jpg)
![fake image804](https://user-images.githubusercontent.com/63538314/147931224-30999b08-6a37-4e1c-8755-57529d51d89f.jpg)
--- 

## *파일구조설명

폴더를 추가했습니다. 

- model_weight : Generator와 Discriminator의 가중치를 저장합니다.
- runs : tensorboardx에 모델 손실정보등을 저장합니다.
- train_image : 모델 학습간에 train 데이터셋에 대한 결과이미지를 저장합니다.
- test_image : 모델 학습간에 test 데이터셋에 대한 결과이미지를 저장합니다.

파일 설명
- Dataset.py : 데이터 로드 및 전처리를 수행합니다.
- Learning_manage.py : 학습의 메인 코드입니다. 이파일을 실행하면 학습이 시작됩니다.
- Network.py : 모델구조를 정의합니다.
- Mask_Maker.py : Discriminator에 사용할 이미지를 크롭 및 합성 연산을 제공합니다.
- Draw2Writer.py : train과 test결과를 저장하는 코드입니다.
- TestModel.py : 실험결과를 확인해볼 모델로 결과를 확인합니다.

## *본인 환경에서 코딩시 주의할점

- Dataset.py 파일의 dataset 변수를 수정해야합니다.
- TestModel.py 파일에서 읽어올 모델 파일이름을 수정해야합니다.

## 진행사항 
- Dataset 및 Datset 완성 (21/12/24)
  - 각 이미지에 대한 데이터셋입니다. 랜덤사이즈의 직사각형 구멍을 형성하는 출력하게 하였습니다. 

- 학습 모델 및 이미지 저정 완성 (21/12/27)
  - 생성기는 마스크와 마스킹된 이미지를 입력으로 받아서 생성한다.
  - 판별기는 생성된 이미지를 global과 local 단위로 입력받아서 괜찮은 이미지인지 확인한다. 

- 생성기의 네트워크 출력 활성화함수를 hyperbolic tangent로 수정함(21/12/29)
  - 현재 생성기 모델의 학습이 수렴중 

- 마스킹 및 loss함수를 조정하였고, 최종결과 확인(22/1/3)
  - 최종적인 학습은 Disciriminator가 가짜데이터를 잘 구분하지 못하는 문제 발생. (fake데이터에 대한 loss가 커짐 )
  - 추가적인 수정이 요구됨.