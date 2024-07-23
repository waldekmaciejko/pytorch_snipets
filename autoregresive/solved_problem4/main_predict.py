import torch
import torch.nn as nn

import learning_helpers
import RNNmodel
import series_definition
import data_generators

input_future = 1
hidden_size = 50
output_size = 1

model = RNNmodel.LSTMmodel(input_future, hidden_size, output_size)
loss_fn = nn.MSELoss()
optim = torch.optim.SGD(params=model.parameters(), lr=0.01)

checkpoint = learning_helpers.load_last_checkpoint(dir_to_checkpoints="../tSeriesResume/exp/model_8_cpu.pt",
                                                    model=model,
                                                    optim=optim)

model = checkpoint['model']
optim = checkpoint['optim']

# ========== prepare synthetic data
term_to_predict = 100
seq_len = 60000
window_len = 40
inc = 1

sinus_generator = series_definition.sinus_generator_v2
data = data_generators.synthetic_data_generator_v4(sinus_generator, seq_len=seq_len, window_len=window_len, inc=inc)
X_data = data[0]

learning_helpers.predict(model, data, term_to_predict, device='cpu')


stop = 0


