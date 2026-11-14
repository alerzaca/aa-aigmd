# AA-AIGMD: Adversarial Attacks on AI-Generated Music Detectors

A framework for running Adversarial Attacks against AI-Generated Music Detectors (`--detector` x `--attack`).

### Registered detectors: 

| Detector | Source | Paper |
|---|---|---|
| SONICS SpecTTTra Gamma, 5s (`awsaf49/sonics-spectttra-gamma-5s`) | [Hugging Face](https://huggingface.co/awsaf49/sonics-spectttra-gamma-5s); [GitHub](https://github.com/awsaf49/sonics) | [SONICS: Synthetic Or Not - Identifying Counterfeit Songs](https://arxiv.org/abs/2408.14080) Rahman et al., ICLR 2025 |
| SONICS SpecTTTra Alpha, 5s (`awsaf49/sonics-spectttra-alpha-5s`) | [Hugging Face](https://huggingface.co/awsaf49/sonics-spectttra-alpha-5s); [GitHub](https://github.com/awsaf49/sonics) | [SONICS: Synthetic Or Not - Identifying Counterfeit Songs](https://arxiv.org/abs/2408.14080) Rahman et al., ICLR 2025 |
| SONICS SpecTTTra Gamma, 120s (`awsaf49/sonics-spectttra-gamma-120s`) | [Hugging Face](https://huggingface.co/awsaf49/sonics-spectttra-gamma-120s); [GitHub](https://github.com/awsaf49/sonics) | [SONICS: Synthetic Or Not - Identifying Counterfeit Songs](https://arxiv.org/abs/2408.14080) Rahman et al., ICLR 2025 |
| SONICS SpecTTTra Alpha, 120s (`awsaf49/sonics-spectttra-alpha-120s`) | [Hugging Face](https://huggingface.co/awsaf49/sonics-spectttra-alpha-120s); [GitHub](https://github.com/awsaf49/sonics) | [SONICS: Synthetic Or Not - Identifying Counterfeit Songs](https://arxiv.org/abs/2408.14080) Rahman et al., ICLR 2025 |
| LOFCZ AI Music Detector (`lofcz/ai-music-detector`) | [Hugging Face](https://huggingface.co/lofcz/ai-music-detector); [GitHub](https://github.com/lofcz/ai-music-detector) | [A Fourier Explanation of AI-music Artifacts](https://arxiv.org/abs/2506.19108) Afchar et al., ISMIR 2025 |
| Deezer `specnn_amplitude` research checkpoint | [Research repository](https://github.com/deezer/deepfake-detector) | [AI-Generated Music Detection and its Challenges](https://arxiv.org/abs/2501.10111) Afchar et al., ICASSP 2025 |

### Registered attacks: 

| Attack | Source | Paper |
|---|---|---|
| FGSM (Fast Gradient Sign Method) | [PyTorch Tutorial](https://docs.pytorch.org/tutorials/beginner/fgsm_tutorial.html) | [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) Goodfellow et al., 2015 |
| PGD (Projected Gradient Descent) | [GitHub](https://github.com/BorealisAI/advertorch) | [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083) Madry et al., 2018 | 
| C&W (Carlini–Wagner) | [GitHub](https://github.com/carlini/audio_adversarial_examples) | [Audio Adversarial Examples: Targeted Attacks on Speech-to-Text](https://arxiv.org/abs/1801.01944) Carlini & Wagner, 2018 |
| Gaussian-noise baseline | [NumPy Gaussian random sampling](https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html) | No single canonical paper; standard experimental baseline |


## Setup

The confirmed supported Python version for the project is 3.12 (3.12.15). Create a new environment or activate an existing one with this version, for example:

```powershell
conda create -n mgr-env python=3.12
conda activate mgr-env
```

Install platform-appropriate PyTorch build (`torch` and `torchaudio`) for the local CPU/CUDA setup using the instructions at [pytorch.org](https://pytorch.org/get-started/locally/), then install other dependencies:

```powershell
python -m pip install -r requirements.txt
```

### Install relevant detectors:

**SONICS:**
```powershell
python -m pip install git+https://github.com/awsaf49/sonics.git
```

**LOFCZ and Deezer:** The dependencies installed from
`requirements.txt` are sufficient.

### Athena environment

On Athena, create the Python 3.12 environment and place Conda, pip, and
Hugging Face caches on the writable `$SCRATCH` filesystem:

```bash
bash scripts/create-athena-env.sh
```

The script does not install PyTorch or project dependencies. Install
`torch`/`torchaudio` using a CUDA build compatible with Athena's GPU driver,
then install the project requirements. TensorFlow is only needed for Deezer;
it can be omitted for SONICS-only runs. The Slurm batch script loads this
environment automatically.


## Data preparation

- Supported formats: WAV and MP3.
- Put verified real audio files in `data/real/`.
- Put verified fake audio files in `data/fake/`.
- Optionally pass `--source <subfolder>` to restrict an attack to one data subfolder.

## Workflow

The predefined batch experiments can be run with `scripts/run.ps1` on Windows,
`scripts/run.sh` on Linux, or `scripts/run-slurm.sh` on a Slurm cluster. The Slurm runner
submits one independent job per detector; each job runs its baselines before
its attacks.

The commands can also be run individually, when you want to choose the parameters yourself, or when you need only one specific test instead of bulk run.

### Individual experiment

Run these commands from the workspace root. First check the environment and the detector of your choosing:

```powershell
python -m src.main check-env
python -m src.main check-gradient --detector <name>
```

For example, run a single-file Deezer prediction with:

```powershell
python -m src.main infer --detector deezer --sample-file ..\deepfake-detector\audio_examples\encodec3\000002.mp3
```

Then create a **baseline**. A baseline runs the detector on the unmodified
audio and is required before running an attack. Use a new results directory for
each experiment:

```powershell
python -m src.main baseline --detector <name> --representation <waveform/mel> --results-dir <path>
```

Finally run one or more **attacks** using the same detector, representation, and
results directory:

```powershell
python -m src.main attack --detector <name> --attack <name> --epsilon <e> --alpha <a> --steps <n> --results-dir <path>
```

The commands above attack waveforms. SONICS also supports `--representation
mel`; LOFCZ supports waveform attacks only; Deezer does not support attacks at this stage. The attack command processes fake
audio, while the baseline includes both real and fake audio.

After the experiments finish, create the report:

```powershell
python analysis\report.py results\sonics\my_run
```

The sequence is therefore:

```text
check-env -> infer/check-gradient (optional) -> baseline -> attack -> report
```

Experiment commands resume from successful rows in their existing CSV files:
completed baseline files and attack file/budget pairs are skipped, while missing
or failed work is retried. Re-run a batch script with the same run name and
options to continue after an interruption. Baseline and attack configuration
files are checked before resuming; changed settings require a new results
directory. When resuming older results without a configuration file, the
current settings are recorded with a warning, so verify they match the original
run. The report only reads raw results and writes tables and figures to
`<run>/report/`, so it can be re-run without repeating the experiment. Inputs
default to `data/` regardless of the working directory.

### Batch experiments

From the project root, edit `RUN_NAME` or `DATA_DIR` in
`scripts/run-slurm.sh` if needed, then submit both alpha detectors:

```bash
bash scripts/run-slurm.sh
```

The runner expects the datasets under `data/fake/sonics-part1` and
`data/fake/sonics-part2`. Adjust CPU, memory, time, and the grant's GPU request
in `scripts/run_detector.sbatch`. On Windows, run `scripts/run.ps1` with the
project environment active.

```powershell
.\scripts\run.ps1
```
