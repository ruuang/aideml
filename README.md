# AIDE 运行说明

## 数据集及任务描述文档存放目录结构

```
<data-dir>/
  └── <competition-id>/
      └── prepared/
          └── public/
              ├── train.csv
              ├── test.csv
              └── ...

<desc-dir>/
  └── <competition-id>/
      └── full_instructions.txt
```

**注意：** 对于 MLE-bench 的任务，借用 [ML-Master](https://github.com/sjtu-sai-agents/ML-Master) 的工作，将各任务说明文档存放在 [dataset/full_instructions](dataset/full_instructions) 文件夹中。

## AIDE 本身运行方式
一次只能运行一个task。AIDE原文README见 [aide/README.md](aide/README.md).
### Setup

设置 OpenAI (or Anthropic) API key:

```bash
export OPENAI_API_KEY=<your API key>
# or
export ANTHROPIC_API_KEY=<your API key>
```

### To run AIDE:

```bash
aide data_dir="<path to your data directory>" goal="<describe the agent's goal for your task>" eval="<(optional) describe the evaluation metric the agent should use>"
```

使用示例:

```bash
aide data_dir="example_tasks/house_prices" goal="Predict the sales price for each house" eval="Use the RMSE metric between the logarithm of the predicted and observed values."
```

或者提供任务描述文档（推荐）.

```bash
aide data_dir="my_data_dir" desc_file="my_task_description.txt"
```

### 命令参数说明
完整参数配置文件在[aide/utils/config.yaml](aide/utils/config.yaml)，参数可通过命令行设置。

示例：
- `data_dir` (required): 设置数据集地址。
- `code.model`: 设置coding模型。

### 输出结果
输出结果存放在log_dir参数控制的文件路径下。

- `best_solution`和`best_submission`: 最优解的代码和该代码生成的提交文件。
- `aide.log`和`aide.verbose.log`: 执行日志，`aide.verbose.log`相比`aide.log`多了每次调用llm时的完整prompt。
- `journal.json`: 每次迭代的代码+执行反馈等信息。
- `tree_plot.html`: 解决方案树。

## AIDE 批量运行
自动执行多个任务。

### 竞赛集文件
每次运行通过竞赛集文件给定要执行的tasks名称。

竞赛集文件是一个文本文件，每行包含一个竞赛ID。支持以下格式：

```
# 注释行会被自动忽略
competition-id-1
competition-id-2
competition-id-3
```

### 使用示例

```bash
python run_competitions.py \
    --competition-set competition_sets/full.txt \
    --max-workers 4 \
    --timeout 7200 \
    --aide-args "agent.steps=20" "agent.code.model=gpt-4o-2024-08-06" \
    --skip-existing \
    --log-dir logs/gpt4o_20steps \
    --n-seed 3 \
    --continue-on-error
```

### 命令参数说明
- `--competition-set <path>` (required)：竞赛集文件路径（每行一个竞赛ID）

- `--data-dir <path>`：数据集根目录（默认：`/root/datasets2/mle-bench-lite/raw`）
  
- `--desc-dir <path>`：Instruction 文件根目录（默认：`dataset/full_instructions`）
  - 每个竞赛的 instruction 文件位于 `<desc-dir>/<competition-id>/full_instructions.txt`

- `--max-workers <num>`：并行运行的最大线程数（默认：1）
  
- `--timeout <seconds>`：每个竞赛的最大运行时间（秒）
  - 默认无限制
  - 超过时间限制的任务会被终止
  - 不能只在[aide/utils/config.yaml](aide/utils/config.yaml)中配置，因为在原来的aide脚本里面这会用到prompt构建中，并没有起到实质的限制作用。
  
- `--n-seed <num>`：每个竞赛的重复运行次数（默认：1）


- `--continue-on-error`：即使某个竞赛失败也继续运行剩余竞赛（默认：启用）

- `--skip-existing`：跳过已存在完整日志的竞赛（默认：启用）

- `--aide-args <args...>`：传递给 aide 命令的额外参数
  - 可以传递多个参数，例如：`--aide-args "agent.steps=10" "agent.code.model=gpt-4o"`

## 后续评估：若使用MLE-bench

###  评估前准备
将所有 competition 的 submission.csv 文件路径整理到一个 JSON 文件中。JSON 文件格式如下：

**JSON 文件结构：**
```json
[
  {
    "competition_id": "<string>",
    "submission_path": "<string>",
    "logs_path": "<string>",
    "code_path": "<string>"
  },
  ...
]
```

**字段说明：**
- `competition_id`: 竞赛名称
- `submission_path` : 最优解决方案提交文件的路径
- `logs_path` : 日志文件的路径
- `code_path` : 最优解决方案代码文件的路径

###  评估
使用 `mlebench grade`命令:

```bash
mlebench grade --submission <path to your submission file> --output-dir <path to your output directory>