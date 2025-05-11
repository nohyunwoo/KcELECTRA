import torch
from transformers import ElectraForSequenceClassification, ElectraTokenizer, Trainer, TrainingArguments, TrainerCallback
import numpy as np
from torch.nn.functional import softmax
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import os
import pickle

# 감정 레이블 정의 (7가지 감정)
emotion_labels = ["공포", "놀람", "분노", "슬픔", "중립", "행복", "혐오"]
label2id = {label: i for i, label in enumerate(emotion_labels)}

def load_koelectra_model(num_labels=7, use_saved=True, model_path="./finetuned_koelectra"):
    from transformers import ElectraForSequenceClassification, ElectraConfig, ElectraTokenizer
    
    # 저장된 모델을 사용하는 경우
    if use_saved and os.path.exists(model_path):
        model = ElectraForSequenceClassification.from_pretrained(model_path)
        tokenizer = ElectraTokenizer.from_pretrained(model_path)
        print(f"모델을 {model_path}에서 불러왔습니다.")
    else:
        # 사전 학습된 KoELECTRA 모델 로드
        model_name = "monologg/koelectra-base-v3-discriminator"
        tokenizer = ElectraTokenizer.from_pretrained(model_name)
        
        # 분류 작업을 위한 모델 구성
        config = ElectraConfig.from_pretrained(model_name, num_labels=num_labels)
        model = ElectraForSequenceClassification.from_pretrained(model_name, config=config)
        print(f"사전 학습된 KoELECTRA 모델을 불러왔습니다.")
    
    return model, tokenizer

def predict_emotion(text, model, tokenizer):
    """입력된 텍스트의 감정을 예측합니다"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    probs = softmax(logits, dim=1).squeeze().numpy()
    predicted_class = np.argmax(probs)
    predicted_emotion = emotion_labels[predicted_class]
    confidence = probs[predicted_class]
    emotion_probs = {emotion: float(prob) for emotion, prob in zip(emotion_labels, probs)}
    return {
        "emotion": predicted_emotion,
        "confidence": float(confidence),
        "all_probabilities": emotion_probs
    }

class EmotionDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len=128):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        text = self.sentences[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label)
        return item

class PrintProgressCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        print(f"\n==== Epoch {int(state.epoch)+1} 시작 ====")
    def on_step_end(self, args, state, control, **kwargs):
        print(f"Step {state.global_step} / {state.max_steps} (Epoch {state.epoch:.2f})")
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"==== Epoch {int(state.epoch)+1} 종료 ====")

def save_validation_data(val_df, save_path='validation_data.pkl'):
    """검증 데이터셋을 저장합니다"""
    with open(save_path, 'wb') as f:
        pickle.dump(val_df, f)
    print(f"검증 데이터셋이 {save_path}에 저장되었습니다.")

def load_validation_data(load_path='validation_data.pkl'):
    """저장된 검증 데이터셋을 불러옵니다"""
    try:
        with open(load_path, 'rb') as f:
            val_df = pickle.load(f)
        print(f"검증 데이터셋을 {load_path}에서 불러왔습니다.")
        return val_df
    except FileNotFoundError:
        print(f"저장된 검증 데이터셋이 없습니다. 새로운 검증 데이터셋을 생성합니다.")
        return None

def finetune_and_save():
    # 1. 데이터 로드
    file_path = r'c:/Users/nohyunwoo/Desktop/AI/data set/한국어 감정 정보가 포함된 단발성 대화 데이터셋/한국어_단발성_대화_데이터셋.xlsx'
    df = pd.read_excel(file_path)
    
    # 전체 데이터셋의 레이블 분포 확인
    print("\n=== 전체 데이터셋 레이블 분포 ===")
    print(df['Emotion'].value_counts())
    print("\n레이블 순서:", emotion_labels)
    
    # 유효한 레이블만 필터링
    df = df[df['Emotion'].isin(label2id)]
    print("\n=== 필터링 후 레이블 분포 ===")
    print(df['Emotion'].value_counts())
    
    # 2. 데이터를 학습(60%), 검증(20%), 테스트(20%)로 분할
    # 먼저 테스트 세트 분리
    train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Emotion'])
    # 나머지 데이터에서 학습과 검증 세트 분리
    train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42, stratify=train_val_df['Emotion'])
    # (0.8 * 0.25 = 0.2이므로, 전체의 20%가 검증 세트)
    
    # 각 데이터셋의 크기와 레이블 분포 출력
    print("\n=== 데이터셋 분할 결과 ===")
    print(f"학습 데이터 크기: {len(train_df)} 샘플")
    print("\n학습 데이터 레이블 분포:")
    print(train_df['Emotion'].value_counts())
    
    print(f"\n검증 데이터 크기: {len(val_df)} 샘플")
    print("\n검증 데이터 레이블 분포:")
    print(val_df['Emotion'].value_counts())
    
    print(f"\n테스트 데이터 크기: {len(test_df)} 샘플")
    print("\n테스트 데이터 레이블 분포:")
    print(test_df['Emotion'].value_counts())
    
    # 3. 학습/검증용 데이터셋 준비
    train_texts = train_df['Sentence'].tolist()
    train_labels = train_df['Emotion'].map(label2id).tolist()
    val_texts = val_df['Sentence'].tolist()
    val_labels = val_df['Emotion'].map(label2id).tolist()
    
    # 테스트 데이터는 별도로 저장 (나중에 평가용)
    test_df.to_csv('./test_data.csv', index=False)
    print("\n테스트 데이터가 'test_data.csv'에 저장되었습니다.")
    
    # 4. 토크나이저 및 데이터셋 생성
    tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
    val_dataset = EmotionDataset(val_texts, val_labels, tokenizer)
    
    # 5. 모델 준비
    model = ElectraForSequenceClassification.from_pretrained(
        "monologg/koelectra-base-v3-discriminator", num_labels=7
    )
    
    # 6. Trainer로 파인튜닝
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=4,               # 에폭 수
        per_device_train_batch_size=8,     # 배치 크기 감소
        per_device_eval_batch_size=8,      # 평가 배치 크기도 감소
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,                  # 10 스텝마다 로깅
        eval_strategy="epoch",             # 에폭 단위로 평가
        save_strategy="epoch",             # 에폭 단위로 저장
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=1,
        report_to="none",
        learning_rate=2e-5,               # 학습률 명시적 설정
        warmup_steps=100,                 # 워밍업 스텝 추가
    )
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        # 각 클래스별 예측 수 계산
        pred_counts = np.bincount(predictions, minlength=len(emotion_labels))
        true_counts = np.bincount(labels, minlength=len(emotion_labels))
        
        # 정확도 계산
        accuracy = np.mean(predictions == labels)
        
        # Label Smoothing이 적용된 손실값 계산
        smoothing = 0.1  # Label Smoothing 계수
        num_classes = len(emotion_labels)
        
        # 원-핫 인코딩으로 변환
        labels_one_hot = torch.zeros(len(labels), num_classes)
        labels_one_hot.scatter_(1, torch.tensor(labels).unsqueeze(1), 1)
        
        # Label Smoothing 적용
        labels_one_hot = labels_one_hot * (1 - smoothing) + smoothing / num_classes
        
        # CrossEntropyLoss 계산
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(torch.tensor(logits), labels_one_hot)
        
        # 각 클래스별 정확도 계산
        class_accuracies = {}
        for i in range(len(emotion_labels)):
            mask = labels == i
            if np.sum(mask) > 0:  # 해당 클래스의 샘플이 있는 경우
                class_acc = np.mean(predictions[mask] == labels[mask])
                class_accuracies[f"{emotion_labels[i]}_accuracy"] = class_acc
        
        return {
            "accuracy": accuracy,
            "loss": loss.item(),
            "class_distribution": {emotion: count for emotion, count in zip(emotion_labels, pred_counts)},
            "true_distribution": {emotion: count for emotion, count in zip(emotion_labels, true_counts)},
            **class_accuracies
        }
    
    # Early Stopping 콜백 추가
    from transformers import EarlyStoppingCallback
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3,        # 3번의 평가에서 개선이 없으면 중단
        early_stopping_threshold=0.001    # 최소 개선 기준
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,  # 평가 지표 추가
        callbacks=[PrintProgressCallback(), early_stopping],  # Early Stopping 콜백 추가
    )
    
    # 학습 진행
    trainer.train()
    
    # 7. 모델 저장
    model.save_pretrained('./finetuned_koelectra')
    tokenizer.save_pretrained('./finetuned_koelectra')
    print("모델과 토크나이저가 './finetuned_koelectra'에 저장되었습니다.")

def evaluate_on_test_data():
    """저장된 테스트 데이터로 모델 성능을 평가합니다"""
    print("저장된 테스트 데이터에서 모델 성능을 평가합니다...")
    
    # 모델 및 토크나이저 로드
    model, tokenizer = load_koelectra_model(use_saved=True)
    
    # 테스트 데이터 로드
    test_df = pd.read_csv('./test_data.csv')
    
    # 테스트 데이터의 감정 분포 출력
    print("\n테스트 데이터의 감정 분포:")
    print(test_df['Emotion'].value_counts())
    
    # 예측 및 평가
    true_labels = test_df['Emotion'].map(label2id).tolist()
    pred_labels = []
    all_probs = []
    
    print("\n각 문장별 예측 결과:")
    for idx, sentence in enumerate(test_df['Sentence'].tolist()):
        result = predict_emotion(sentence, model, tokenizer)
        pred_emotion = result['emotion']
        pred_labels.append(label2id[pred_emotion])
        all_probs.append(result['all_probabilities'])
        
        # 처음 5개 문장만 상세 출력
        if idx < 5:
            print(f"\n문장 {idx+1}: {sentence}")
            print(f"실제 감정: {test_df['Emotion'].iloc[idx]}")
            print(f"예측 감정: {pred_emotion} (신뢰도: {result['confidence']:.4f})")
            print("모든 감정 확률:")
            for emotion, prob in sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
                print(f"  - {emotion}: {prob:.4f}")
    
    # 평가 지표 계산
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    
    accuracy = accuracy_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, target_names=emotion_labels)
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    
    print(f"\n=== 전체 평가 결과 ===")
    print(f"테스트 세트 정확도: {accuracy:.4f}")
    print("\n분류 리포트:")
    print(report)
    
    print("\n혼동 행렬:")
    print(conf_matrix)
    
    # 각 감정별 평균 신뢰도 계산
    print("\n=== 각 감정별 평균 신뢰도 ===")
    for emotion in emotion_labels:
        emotion_idx = label2id[emotion]
        emotion_probs = [probs[emotion] for probs in all_probs]
        avg_confidence = sum(emotion_probs) / len(emotion_probs)
        print(f"{emotion}: {avg_confidence:.4f}")

def main():
    print("KoELECTRA 감정 분석 테스트를 시작합니다...")

    # 모델 및 토크나이저 로드
    try:
        model, tokenizer = load_koelectra_model(use_saved=True)
        print("모델 로드 완료!")
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        return

    # 테스트할 문장들
    test_sentences = [
        "오늘은 날씨가 너무 좋아서 기분이 좋아요!",
        "시험에서 떨어져서 너무 슬퍼요.",
        "이 상황이 너무 화가 나고 짜증이 납니다.",
        "갑자기 무서운 소리가 들려서 심장이 쿵쾅거려요.",
        "이 음식은 너무 이상한 냄새가 나서 먹을 수 없을 것 같아요.",
        "갑자기 친구가 뒤에서 놀래켜서 깜짝 놀랐어요.",
        "오늘 날씨는 맑고 기온은 23도입니다."
    ]

    # 각 문장에 대한 감정 예측
    print("\n=== 감정 분석 결과 ===")
    for i, sentence in enumerate(test_sentences):
        result = predict_emotion(sentence, model, tokenizer)
        print(f"\n[문장 {i+1}] {sentence}")
        print(f"예측 감정: {result['emotion']} (신뢰도: {result['confidence']:.4f})")
        print("모든 감정 확률:")
        sorted_probs = sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)
        for emotion, prob in sorted_probs:
            print(f"  - {emotion}: {prob:.4f}")

def evaluate_model():
    """모델의 성능을 평가합니다"""
    print("KoELECTRA 감정 분석 모델 평가를 시작합니다...")
    
    # 모델 및 토크나이저 로드
    model, tokenizer = load_koelectra_model(use_saved=True)
    
    # 저장된 검증 데이터셋 불러오기
    test_df = load_validation_data()
    
    # 저장된 검증 데이터셋이 없는 경우 새로운 데이터셋 생성
    if test_df is None:
        file_path = r'c:/Users/nohyunwoo/Desktop/AI/data set/한국어 감정 정보가 포함된 단발성 대화 데이터셋/한국어_단발성_대화_데이터셋.xlsx'
        df = pd.read_excel(file_path)
        df = df[df['Emotion'].isin(label2id)]
        _, test_df = train_test_split(df, test_size=0.2, random_state=42)
        save_validation_data(test_df)
    
    # 실제 레이블과 예측 결과 저장할 리스트
    true_labels = []
    pred_labels = []
    
    # 각 문장에 대한 감정 예측
    print("\n검증 데이터셋 평가 중...")
    for idx, row in test_df.iterrows():
        sentence = row['Sentence']
        true_emotion = row['Emotion']
        true_label = label2id[true_emotion]
        
        result = predict_emotion(sentence, model, tokenizer)
        pred_emotion = result['emotion']
        pred_label = label2id[pred_emotion]
        
        true_labels.append(true_label)
        pred_labels.append(pred_label)
        
    # 평가 지표 계산
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    
    accuracy = accuracy_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, target_names=emotion_labels)
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    
    # 결과 출력
    print(f"\n=== 모델 평가 결과 ===")
    print(f"정확도 (Accuracy): {accuracy:.4f}")
    print("\n분류 리포트:")
    print(report)
    
    print("\n혼동 행렬 (Confusion Matrix):")
    print(conf_matrix)
    
    # 결과를 시각화
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=emotion_labels, yticklabels=emotion_labels)
        plt.xlabel('예측된 감정')
        plt.ylabel('실제 감정')
        plt.title('감정 분류 혼동 행렬')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        print("\n혼동 행렬 이미지가 'confusion_matrix.png' 파일로 저장되었습니다.")
    except ImportError:
        print("\n시각화 라이브러리가 설치되지 않아 그래프를 생성할 수 없습니다.")
    
    # 오분류된 예시 확인
    print("\n=== 오분류된 예시 ===")
    misclassified = []
    for idx, row in test_df.iterrows():
        sentence = row['Sentence']
        true_emotion = row['Emotion']
        
        result = predict_emotion(sentence, model, tokenizer)
        pred_emotion = result['emotion']
        
        if true_emotion != pred_emotion:
            misclassified.append({
                'sentence': sentence,
                'true_emotion': true_emotion,
                'predicted_emotion': pred_emotion,
                'confidence': result['confidence']
            })
    
    # 오분류된 예시 중 일부 출력
    sample_size = min(10, len(misclassified))
    for i, example in enumerate(misclassified[:sample_size]):
        print(f"\n[오분류 예시 {i+1}]")
        print(f"문장: {example['sentence']}")
        print(f"실제 감정: {example['true_emotion']}")
        print(f"예측 감정: {example['predicted_emotion']} (신뢰도: {example['confidence']:.4f})")

def check_model_output():
    model, tokenizer = load_koelectra_model(use_saved=True)
    test_text = "오늘은 정말 화가 납니다."
    inputs = tokenizer(test_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = softmax(logits, dim=1).squeeze().numpy()
    print("모델 출력 확률:", probs)
    print("예측된 클래스:", np.argmax(probs))
    print("예측된 감정:", emotion_labels[np.argmax(probs)])

if __name__ == "__main__":
    # 모델 출력 확인
    # check_model_output()
    
    # 파인튜닝 및 저장을 원할 때 아래 함수 실행
    finetune_and_save()
    
    # 감정 예측 테스트를 원할 때 main() 실행
    # main()
    
    # 테스트 데이터로 모델 평가를 원할 때 evaluate_on_test_data() 실행
    # evaluate_on_test_data()