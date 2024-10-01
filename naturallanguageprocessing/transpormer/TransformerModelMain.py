import torch
from Transformer import Transformer

# 간단한 단어 사전
vocab = {
    "hello": 0,
    "world": 1,
    "this": 2,
    "is": 3,
    "a": 4,
    "test": 5,
    "transformer": 6,
    "model": 7,
    "example": 8
}

# 문장을 단어 인덱스 시퀀스로 변환
def sentence_to_indices(sentence, vocab):
    return [vocab.get(word, vocab['hello']) for word in sentence.split()]

# Mask 생성
def create_mask(seq):
    seq_mask = (seq != 0).unsqueeze(1).unsqueeze(2)
    return seq_mask

def main():
    src_vocab_size = len(vocab)
    tgt_vocab_size = len(vocab)
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_len = 100
    dropout = 0.1

    # Transformer 모델 생성
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout)

    # 입력 문장
    src_sentence = "hello world this is a test"
    tgt_sentence = "this is a transformer model example"

    # 시퀀스를 인덱스로 변환
    src_seq = torch.tensor([sentence_to_indices(src_sentence, vocab)])
    tgt_seq = torch.tensor([sentence_to_indices(tgt_sentence, vocab)])

    # 마스크 생성
    src_mask = create_mask(src_seq)
    tgt_mask = create_mask(tgt_seq)

    # Transformer 모델 출력
    output = model(src_seq, tgt_seq, src_mask, tgt_mask)
    print("Transformer Output Shape is ", output.shape)

if __name__ == "__main__":
    main()
