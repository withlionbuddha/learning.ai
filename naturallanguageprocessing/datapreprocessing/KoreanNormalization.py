from Normalization import Normalization
import torch
from torch.nn.utils.rnn import pad_sequence

class KoreanNormalization(Normalization):
    def __init__(self, encoded_data):
        super().__init__(encoded_data)

    def normalize(self, max_length):
        # Step 1 시퀀스를 텐서로 변환
        tensor_sequences = [torch.tensor(seq, dtype=torch.long) for seq in self.encoded_data]
        print(f" tensor_sequences is {tensor_sequences}")
        
        for i, tensor in enumerate(tensor_sequences):
            print(f" shape of tensor {i} is {tensor.shape}")

        # step 2 스칼라를 1D 텐서로 변환
        tensor_sequences = [t.unsqueeze(0) for t in tensor_sequences]
        
        # Step 3 pad_sequence를 사용하여 시퀀스의 길이를 동일하게 맞추고 패딩을 추가
        padded_data = pad_sequence(tensor_sequences, batch_first=True, padding_value=0)
        print(f" padded_data is {padded_data}")
        
        return padded_data[:, :max_length]  # max_length에 맞춰 길이 자르기
