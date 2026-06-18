# ExecAgent

本仓库包含两个执行推理方法的最终代码目录：

1. `function_method/`：函数级 CFG 引导执行推理方法，面向单函数执行任务。
2. `repository_inference/`：仓库级执行推理方法，面向 `PyRepoExec`，包含 route agent、state agent、递归函数展开、状态维护和结果评估。

## 1. 环境依赖

建议使用 Python 3.10 或兼容的 Python 3 版本。

安装必要的 Python 包：

```bash
pip install openai python-dotenv json-repair
```

代码中还会导入 `cfg_builder.build_cfg_text`。请确保原实验使用的 `cfg_builder` 包或模块已经安装，或已加入 `PYTHONPATH`。

### API Key 配置

两个方法都使用 OpenAI-compatible Chat API。

`repository_inference` 默认配置为：

```text
api_key_env = DEEPSEEK_API_KEY
base_url = https://api.deepseek.com
model = deepseek-v4-flash
```

可以在命令行中设置 API key：

```bash
set DEEPSEEK_API_KEY=your_key_here
```

也可以在仓库根目录放置 `.env` 文件：

```text
DEEPSEEK_API_KEY=your_key_here
```

`function_method` 需要在 `function_method/config.py` 中根据所使用的 OpenAI-compatible 服务修改：

```text
MODEL_CONFIG["api_key_env"]
MODEL_CONFIG["base_url"]
MethodConfig.model
```

## 2. 目录结构

```text
function_method/
  config.py          模型、路径和运行配置
  dataset.py         jsonl 数据集读取
  cfg_utils.py       代码清理和 CFG 构造输入
  prompt.py          函数级 CFG 引导 prompt
  llm.py             异步 OpenAI-compatible 模型调用
  pipeline.py        单样本和数据集执行流程
  result_parser.py   模型响应解析
  evaluate.py        保存结果的评估逻辑
  run_method.py      命令行入口

repository_inference/
  config.py           模型、路径和运行配置
  dataset.py          PyRepoExec 样本读取
  preprocess.py       仓库切片和静态分析预处理
  static_tools.py     静态源码查找辅助函数
  prompts.py          route/state/simulation prompt
  engine.py           仓库级推理主引擎
  run_inference.py    推理运行入口
  evaluate_results.py 结果评估入口
```

## 3. 函数级方法

函数级方法面向 `Dataset/` 目录下的 jsonl 格式函数执行数据集。

已配置的数据集名称包括：

```text
CRUXEval
HumanEval
LCB-Exec
CodeSense-O
```

### 小规模测试

```bash
python -m function_method.run_method --limit 5
```

### 运行一个数据集

```bash
python -m function_method.run_method --dataset-name CodeSense-O
```

其他示例：

```bash
python -m function_method.run_method --dataset-name CRUXEval
python -m function_method.run_method --dataset-name HumanEval
python -m function_method.run_method --dataset-name LCB-Exec
```

### 运行一个样本

```bash
python -m function_method.run_method --target sample_0
```

### 评估已有输出

```bash
python -m function_method.run_method --eval-only
```

默认输出位置：

```text
clean_prompt/function_method
clean_result/function_method
```

## 4. 仓库级方法

仓库级方法面向 `PyRepoExec`。

期望的数据集结构如下：

```text
PyRepoExec/
  project_name/
    task1/
      task.json
      files.txt
      slicing/
        package@@module.py
```

每个任务包含入口脚本和期望打印输出。方法会为每个样本保存详细 prompt、模型原始响应、每轮状态文件，以及最终的 `summary.json`。

### 方法流程

1. 读取一个 `PyRepoExec` 样本。
2. 预处理切片后的仓库文件和入口脚本。
3. 解析函数、方法、类、import、调用点和共享绑定。
4. 为当前执行单元构造 CFG 文本。
5. 使用 route agent 判断调用应该 inline 还是 expand。
6. 使用 simulation agent 模拟当前 frame 的执行。
7. 当执行暂停在被调函数处时，使用 state agent 恢复子 frame 的运行时状态。
8. 当函数执行结束时，使用 state agent 将持久化状态变更写回共享状态。
9. 回到 caller 继续执行，直到入口脚本结束。
10. 保存 `summary.json` 和所有中间推理痕迹。

### 运行完整数据集

```bash
python -m repository_inference.run_inference \
  --dataset-root PyRepoExec \
  --output-dir repo_inference_result \
  --model deepseek-v4-flash
```

### 运行一个项目

```bash
python -m repository_inference.run_inference \
  --dataset-root PyRepoExec \
  --output-dir repo_inference_result \
  --model deepseek-v4-flash \
  --project django
```

### 运行一个任务

```bash
python -m repository_inference.run_inference \
  --dataset-root PyRepoExec \
  --output-dir repo_inference_result \
  --model deepseek-v4-flash \
  --task django/task20
```

也可以使用项目名和任务名组合指定：

```bash
python -m repository_inference.run_inference \
  --dataset-root PyRepoExec \
  --output-dir repo_inference_result \
  --model deepseek-v4-flash \
  --project django \
  --task task20
```

### 并发运行

仓库级方法支持样本级并发：

```bash
python -m repository_inference.run_inference \
  --dataset-root PyRepoExec \
  --output-dir repo_inference_result \
  --model deepseek-v4-flash \
  --max-workers 20
```

如果需要调试，可以改为顺序执行：

```bash
python -m repository_inference.run_inference \
  --dataset-root PyRepoExec \
  --output-dir repo_inference_result \
  --model deepseek-v4-flash \
  --sequential
```

## 5. 仓库级结果评估

对保存的 `summary.json` 结果进行评估：

```bash
python -m repository_inference.evaluate_results \
  --dataset-root PyRepoExec \
  --output-dir repo_inference_result
```

输出错误或缺失的样本：

```bash
python -m repository_inference.evaluate_results \
  --dataset-root PyRepoExec \
  --output-dir repo_inference_result \
  --show-wrong
```

评估一个项目：

```bash
python -m repository_inference.evaluate_results \
  --dataset-root PyRepoExec \
  --output-dir repo_inference_result \
  --project django
```

评估一个任务：

```bash
python -m repository_inference.evaluate_results \
  --dataset-root PyRepoExec \
  --output-dir repo_inference_result \
  --task django/task20 \
  --show-wrong
```
