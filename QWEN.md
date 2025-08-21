# OpenEvolve Project Context for Qwen Code

## Project Overview

OpenEvolve is an open-source evolutionary coding agent that began as a faithful implementation of AlphaEvolve and has evolved far beyond it. It uses Large Language Models (LLMs) to automatically optimize and discover algorithms through iterative improvement. It serves as both a research platform for evolutionary AI and a practical tool for automated code optimization.

### Key Technologies
- Python 3.9+
- LLM APIs (OpenAI-compatible)
- PyYAML for configuration
- NumPy for numerical computations
- Flask for visualization web server

### Core Architecture
1. **Evolutionary Coding Agent**: LLM-guided evolution of entire code files
2. **Distributed Controller Loop**: Asynchronous pipeline coordinating LLMs, evaluators, and databases
3. **Program Database**: Storage and sampling of evolved programs with evaluation metrics
4. **Prompt Sampling**: Context-rich prompts with past programs, scores, and problem descriptions
5. **LLM Ensemble**: Multiple language models working together for code generation
6. **Multi-objective Optimization**: Simultaneous optimization of multiple evaluation metrics
7. **Checkpoint System**: Automatic saving and resuming of evolution state

### Advanced Features
- **Scientific Reproducibility**: Full deterministic reproduction with hash-based component isolation
- **Advanced LLM Integration**: Ensemble sophistication, Test-Time Compute (optillm integration), Universal API Support
- **Evolution Algorithm Innovations**: MAP-Elites Implementation, Island-Based Evolution, Multi-Strategy Selection
- **Evaluation & Feedback Systems**: Artifacts Side-Channel, Cascade Evaluation, LLM-Based Feedback
- **Multi-Language & Platform Support**: Language Agnostic, Platform Optimization
- **Developer Experience & Tooling**: Real-Time Visualization, Advanced CLI, Comprehensive Examples
- **Performance & Scalability**: Process-Based Parallelism, Resource Management

## Building and Running

### Installation
```bash
git clone https://github.com/codelion/openevolve.git
cd openevolve
pip install -e .
```

Or using Docker:
```bash
docker build -t openevolve .
```

### Quick Start
1. Set up LLM access by exporting the `OPENAI_API_KEY` environment variable:
   ```bash
   export OPENAI_API_KEY=your-api-key-here
   ```
2. Run an example:
   ```bash
   python openevolve-run.py examples/function_minimization/initial_program.py examples/function_minimization/evaluator.py --config examples/function_minimization/config.yaml --iterations 1000
   ```

### Command-Line Usage
```bash
python openevolve-run.py path/to/initial_program.py path/to/evaluator.py --config path/to/config.yaml --iterations 1000
```

### Resuming from Checkpoints
```bash
python openevolve-run.py path/to/initial_program.py path/to/evaluator.py \
  --config path/to/config.yaml \
  --checkpoint path/to/checkpoint_directory \
  --iterations 50
```

### Visualization
1. Install requirements:
   ```bash
   pip install -r scripts/requirements.txt
   ```
2. Start the visualization web server:
   ```bash
   python scripts/visualizer.py
   ```

## Development Conventions

### Code Structure
- `/openevolve`: Main source code
- `/examples`: Example problems and configurations
- `/configs`: Configuration templates
- `/scripts`: Utility scripts (including visualizer)
- `/tests`: Test suite (not explored in detail)

### Configuration
OpenEvolve is highly configurable using YAML files. Key sections include:
- `llm`: LLM model configurations
- `database`: MAP-Elites and island-based evolution settings
- `evaluator`: Evaluation pipeline settings
- `prompt`: Prompt engineering configurations

### Key Components
1. **Controller** (`openevolve/controller.py`): Main orchestration logic
2. **CLI** (`openevolve/cli.py`): Command-line interface
3. **Config** (`openevolve/config.py`): Configuration handling
4. **Database** (`openevolve/database.py`): Program storage and sampling
5. **Evaluator** (`examples/*/evaluator.py`): Problem-specific evaluation logic
6. **Prompt Sampler** (`openevolve/prompt/sampler.py`): Prompt generation logic

### Example Problem Structure
Each example consists of:
1. `initial_program.py`: The starting code to evolve (with `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END` markers)
2. `evaluator.py`: Evaluation function that returns metrics
3. `config.yaml`: Configuration for the evolution run

## Project-Specific Instructions for Qwen Code

1. When modifying code, ensure it's within the `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END` markers if you want it to be evolved.
2. When writing tests or evaluation functions, make sure to return a dictionary of metrics.
3. Pay attention to the configuration options in `configs/` when setting up new experiments.
4. Use the checkpoint system for long-running experiments to ensure results are saved.
5. For visualization, run the `scripts/visualizer.py` script after an evolution run.
6. When implementing new features, follow the existing code style and patterns.









要求：
1、分析尽可能说中文；
2、我的总目标是利用openevolve来写triton kernel，并且会利用kernelbench来跑测试，主要是level1的；
3、能修改的代码只能在/Users/qiaolina/Code/openevolve/examples/gen_triton_kernel这个目录里；
4、kernelbench目录在这里：/Users/qiaolina/Code/openevolve/examples/gen_triton_kernel/KernelBench

## KernelBench Information

KernelBench is a benchmark for evaluating LLMs' ability to generate efficient GPU kernels. The user's goal is to use OpenEvolve to write Triton kernels and test them using KernelBench, focusing on Level 1 problems.

### KernelBench Task Description

KernelBench structures the problem for LLMs to transpile operators described in PyTorch to CUDA/Triton kernels. It has 4 levels of categories:

- **Level 1 🧱**: Single-kernel operators (100 Problems)
  The foundational building blocks of neural nets (Convolutions, Matrix multiplies, Layer normalization)
- **Level 2 🔗**: Simple fusion patterns (100 Problems)
  A fused kernel would be faster than separated kernels (Conv + Bias + ReLU, Matmul + Scale + Sigmoid)
- **Level 3 ⚛️**: Full model architectures (50 Problems)
  Optimize entire model architectures end-to-end (MobileNet, VGG, MiniGPT, Mamba)
- **Level 4 🤗**: Level Hugging Face
  Optimize whole model architectures from HuggingFace

### KernelBench Evaluation Methodology

- **Correctness ✅**: Check against reference torch operators `n_correctness` times on randomized inputs.
- **Performance ⏱️**: Compare against reference torch operators `n_trial` times to measure speedup between runtimes.

The key metric is `fast_p`: fraction of tasks that are both correct and have a speedup greater than threshold `p`.
- `fast_1`: Fraction of tasks where kernels are both correct and faster than PyTorch baseline.
- `fast_2`: Fraction of tasks where kernels are both correct and at least 2x faster than PyTorch baseline.
- `fast_0`: Fraction of tasks where kernels are correct (same as correctness rate).

### User's Project Structure

The user will be working within the `/Users/qiaolina/Code/openevolve/examples/gen_triton_kernel` directory. The KernelBench repository is located at `/Users/qiaolina/Code/openevolve/examples/gen_triton_kernel/KernelBench`.

### Usage for User's Project

The user's workflow will involve:
1.  Creating an `initial_program.py` with a starting Triton kernel implementation.
2.  Developing an `evaluator.py` that uses KernelBench's logic (likely by calling scripts like `scripts/run_and_check.py` or directly using functions from `src/eval.py`) to test the generated kernels against Level 1 problems.
3.  Configuring `config.yaml` for the OpenEvolve run.
4.  Running OpenEvolve to evolve the Triton kernels.
5.  Analyzing the results using KernelBench's analysis tools or custom scripts to determine the `fast_p` metric.

## Using Local vLLM Service

To use a local vLLM service instead of an external API, you need to modify the `config.yaml` file in your project:

1. Set `llm.api_base` to the address of your local vLLM service (e.g., `http://localhost:8000/v1`).
2. Set `llm.api_key` to a placeholder value (e.g., `"none"`) as vLLM typically doesn't require an API key.
3. Ensure the model names in `llm.models` match the models loaded in your vLLM service.