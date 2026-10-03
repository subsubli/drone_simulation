"""Export deterministic X500 policy with embedded observation normalization."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from src.policy import GaussianPolicy


class DeploymentPolicy(torch.nn.Module):
    def __init__(self, policy, mean, std):
        super().__init__()
        self.net = policy.net
        self.register_buffer('obs_mean', torch.from_numpy(mean))
        self.register_buffer('obs_std', torch.from_numpy(std))

    def forward(self, observations):
        return torch.tanh(self.net((observations-self.obs_mean)/self.obs_std))



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--csv-file', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = json.loads((args.run_dir/'config.json').read_text())
    if config['dataset_format'] != 'x500_ctbr' or config['max_action'] != 1.:
        raise ValueError('Expected X500 CTBR policy with action bound 1')
    policy = GaussianPolicy(config['obs_dim'], 4, hidden_dim=config['hidden_dim'], n_hidden=config['n_hidden'], max_action=1.)
    state = torch.load(args.run_dir/'final.pt', map_location='cpu', weights_only=True)
    policy.load_state_dict({k.removeprefix('policy.'): v for k, v in state.items() if k.startswith('policy.')})
    norm = np.load(args.run_dir/'obs_normalization.npz')
    model = DeploymentPolicy(policy, norm['mean'], norm['std']).eval()
    output = args.run_dir/'policy.onnx'
    torch.onnx.export(model, torch.zeros(1, 25), str(output), opset_version=17,
                      input_names=['observations'], output_names=['actions'],
                      dynamic_axes={'observations': {0: 'batch'}, 'actions': {0: 'batch'}},
                      dynamo=False)
    graph = onnx.load(str(output))
    onnx.helper.set_model_props(graph, {
        'observations': 'float32 [batch,25], raw X500 observations; normalization embedded',
        'actions': 'float32 [batch,4], deterministic normalized CTBR in [-1,1]',
        'coordinates': 'world NWU, body FLU',
        'control_hz': '100',
        'action_conversion': 'thrust_N=(a0+1)/2*34.19; rates_deg_s=a[1:]*[220,220,200]',
    })
    onnx.checker.check_model(graph)
    onnx.save(graph, str(output))
    with gzip.open(args.csv_file, 'rt') as f:
        data = np.genfromtxt(f, delimiter=',', names=True, dtype=np.float32)
    split = json.loads((args.run_dir/'dataset_split.json').read_text())
    heldout = np.isin(data['flight'], split['validation_flights'])
    obs = np.column_stack([data[f'obs_{i}'][heldout] for i in range(25)])
    session = ort.InferenceSession(str(output), providers=['CPUExecutionProvider'])
    max_error = 0.
    for batch in [obs[:1], *np.array_split(obs, 23)]:
        actual = session.run(['actions'], {'observations': batch})[0]
        with torch.no_grad():
            expected = policy((torch.from_numpy(batch)-model.obs_mean)/model.obs_std).mean.numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
        assert np.isfinite(actual).all() and np.abs(actual).max() <= 1.
        max_error = max(max_error, float(np.abs(actual-expected).max()))
    report = dict(heldout_rows=len(obs), max_absolute_error=max_error,
                  input='observations: float32 [batch,25], raw (not normalized)',
                  output='actions: float32 [batch,4], normalized CTBR',
                  normalization_embedded=True, opset=17,
                  onnx_version=onnx.__version__, onnxruntime_version=ort.__version__)
    (args.run_dir/'onnx_validation.json').write_text(json.dumps(report, indent=2))
    print(output)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
