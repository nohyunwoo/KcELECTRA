# KoELECTRA 한국어 감정 분석 모델

KoELECTRA 기반 한국어 텍스트 감정 분류 모델입니다. 7가지 감정(공포, 놀람, 분노, 슬픔, 중립, 행복, 혐오)을 분류할 수 있습니다.

## 주요 특징

- **사전 학습 모델**: beomi/KcELECTRA-base-v2022
- **감정 분류**: 7가지 감정 카테고리 지원
- **데이터 증강**: 유사 감정 쌍 기반 자동 증강
- **클래스 가중치**: 불균형 데이터 처리를 위한 가중치 적용
- **실시간 추론**: 대화형 감정 예측 인터페이스

## 감정 레이블

| 레이블 | 설명 |
|--------|------|
| 공포 | 두려움, 무서움 |
| 놀람 | 깜짝 놀람, 의외 |
| 분노 | 화남, 짜증 |
| 슬픔 | 우울함, 슬픔 |
| 중립 | 중립적, 객관적 |
| 행복 | 기쁨, 즐거움 |
| 혐오 | 싫음, 역겨움 |

## 설치 방법

### 필수 라이브러리

```bash
pip install torch transformers pandas scikit-learn openpyxl matplotlib seaborn
```

### 요구사항

- Python 3.7+
- PyTorch 1.9+
- Transformers 4.0+

## 데이터 준비

1. 데이터셋 형식: Excel 파일 (.xlsx)
2. 필수 컬럼:
   - `Sentence`: 텍스트 문장
   - `Emotion`: 감정 레이블 (위 7가지 중 하나)

3. 데이터 경로 설정:
```python
file_path = r'경로/한국어_단발성_대화_데이터셋.xlsx'
```

## 사용 방법

### 1. 모델 학습

```python
# 처음부터 학습
finetune_and_save()
```

**학습 설정**:
- 에폭: 5
- 배치 크기: 8 (실질적 배치: 32, gradient accumulation)
- 학습률: 2e-5
- 데이터 분할: 학습 60%, 검증 20%, 테스트 20%
- Early Stopping: 3회 patience

### 2. 실시간 감정 예측

```python
# 대화형 인터페이스 실행
interactive_inference()
```

```
문장 입력 (종료하려면 'exit' 입력): 오늘 정말 기분이 좋아요!

예측 감정: 행복 (신뢰도: 0.9234)
모든 감정 확률:
  - 행복: 0.9234
  - 중립: 0.0543
  - 놀람: 0.0123
  ...
```

### 3. 모델 평가

```python
# 테스트 데이터로 평가
evaluate_on_test_data()

# 또는 검증 데이터로 평가
evaluate_model()
```

### 4. 단일 문장 예측

```python
model, tokenizer = load_koelectra_model(use_saved=True)

result = predict_emotion("오늘은 날씨가 좋아요!", model, tokenizer)
print(f"감정: {result['emotion']}")
print(f"신뢰도: {result['confidence']}")
print(f"전체 확률: {result['all_probabilities']}")
```

## 코드 구조

```
├── load_koelectra_model()      # 모델 로드
├── predict_emotion()            # 감정 예측
├── finetune_and_save()         # 모델 학습 및 저장
├── evaluate_on_test_data()     # 테스트 평가
├── evaluate_model()            # 검증 평가
├── interactive_inference()     # 실시간 추론
├── EmotionDataset              # 데이터셋 클래스
└── WeightedLossTrainer         # 가중치 손실 Trainer
```

## 주요 기능

### 데이터 증강
유사 감정 쌍 간 혼동을 줄이기 위한 데이터 증강:
- 분노 ↔ 혐오
- 공포 ↔ 놀람
- 슬픔 ↔ 공포
- 중립 ↔ 기타 감정

### 클래스 가중치
불균형 데이터 처리를 위한 감정별 가중치:
- 공포: 1.75
- 놀람: 1.32
- 분노: 2.51
- 슬픔: 1.95
- 중립: 1.44
- 행복: 1.82
- 혐오: 2.24

### 평가 지표
- Accuracy (정확도)
- Macro F1 Score
- Precision/Recall (각 클래스별)
- Confusion Matrix (혼동 행렬)

## 출력 파일

### 학습 중
- `./results/`: 체크포인트 저장
- `./logs/`: 학습 로그
- `./test_data.csv`: 테스트 데이터

### 평가 후
- `./finetuned_kcelectra/`: 최종 모델
- `validation_data.pkl`: 검증 데이터
- `confusion_matrix.png`: 혼동 행렬 이미지

## 성능 향상 팁

1. **데이터 증강**: similar_emotion_pairs 수정
2. **가중치 조정**: manual_weights와 confusion_penalty 조정
3. **하이퍼파라미터**: learning_rate, batch_size, epochs 조정
4. **드롭아웃**: hidden_dropout_prob 조정

## 예제 결과

```
[문장 1] 오늘은 날씨가 너무 좋아서 기분이 좋아요!
예측 감정: 행복 (신뢰도: 0.9456)

[문장 2] 시험에서 떨어져서 너무 슬퍼요.
예측 감정: 슬픔 (신뢰도: 0.8923)

[문장 3] 이 상황이 너무 화가 나고 짜증이 납니다.
예측 감정: 분노 (신뢰도: 0.9012)
```

## 문제 해결

### 모델 로드 실패
```python
# 새로운 모델 다운로드
model, tokenizer = load_koelectra_model(use_saved=False)
```

### 메모리 부족
```python
# 배치 크기 줄이기
per_device_train_batch_size=4
```

### 정확도 낮음
- 더 많은 에폭 학습
- 데이터 증강 비율 조정
- 클래스 가중치 재조정

## 라이선스

이 프로젝트는 학습 및 연구 목적으로 사용됩니다.

## 참고

- KcELECTRA: https://github.com/Beomi/KcELECTRA
- Hugging Face Transformers: https://huggingface.co/transformers/

## 업데이트 노트

- v1.0: 초기 버전
- 7가지 감정 분류
- 실시간 추론 지원
- 클래스 가중치 최적화