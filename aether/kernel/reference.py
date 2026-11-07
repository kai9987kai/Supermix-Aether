"""Independent NumPy scalar dynamics; never calls the tensor step."""
import numpy as np


def scalar_step(model, observation, state):
    """Return (logits, next_state) using a snapshot of model scalar parameters."""
    weights = {name: tensor.detach().cpu().numpy().copy()
               for name, tensor in model.state_dict().items()}
    observation, state = np.asarray(observation), np.asarray(state)
    c = model.config
    if observation.ndim != 2 or observation.shape[1] != c.input_dimension:
        raise ValueError('invalid observation shape')
    if state.shape != (observation.shape[0], c.neuron_count, c.state_dimension):
        raise ValueError('invalid state shape')
    aggregate = np.zeros_like(state)
    for edge, (source, target) in enumerate(zip(weights['scalar_sources'], weights['scalar_targets'])):
        aggregate[:, target] += (state[:, source] *
                                (weights['edge_weights'][edge] / weights['scalar_degree'][target]))
    sensory = observation @ weights['input_encoder.weight'].T + weights['input_encoder.bias']
    drive = (state * weights['self_gain'] + aggregate @ weights['local_transform']
             + sensory[:, None] * weights['input_profile'] + weights['neuron_bias'])
    next_state = np.tanh(drive)
    readout = next_state[:, weights['readout_neurons']].mean(axis=1)
    logits = readout @ weights['output_decoder.weight'].T + weights['output_decoder.bias']
    return logits, next_state
