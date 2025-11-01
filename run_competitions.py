#!/usr/bin/env python3
"""
AIDE 批量运行脚本
自动为多个竞赛执行 aide 命令
支持并行运行多个竞赛以提高效率
"""
import argparse
import logging
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional
import threading

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# 线程锁用于线程安全的日志记录
log_lock = threading.Lock()




def load_competition_ids(competition_set_file: Path) -> List[str]:
    """从文件加载竞赛ID列表"""
    if not competition_set_file.exists():
        raise FileNotFoundError(f"竞赛集文件不存在: {competition_set_file}")
    
    competition_ids = []
    with open(competition_set_file, 'r') as f:
        for line in f:
            line = line.strip()
            # 跳过空行和注释行（以#开头）
            if line and not line.startswith('#'):
                competition_ids.append(line)
    
    logger.info(f"从 {competition_set_file} 加载了 {len(competition_ids)} 个竞赛")
    return competition_ids


def safe_log(message: str, level: str = "info"):
    """线程安全的日志记录"""
    with log_lock:
        if level == "info":
            logger.info(message)
        elif level == "error":
            logger.error(message)
        elif level == "warning":
            logger.warning(message)


def check_existing_logs(competition_id: str, log_base_dir: Path) -> bool:
    """检查是否已存在完整的实验日志"""
    log_dir = log_base_dir / competition_id
    
    # 检查日志目录是否存在
    if not log_dir.exists():
        return False
    
    # 检查关键文件是否存在
    required_files = [
        "aide.log",
        "journal.json", 
        "config.yaml",
        "tree_plot.html"
    ]
    
    for file_name in required_files:
        if not (log_dir / file_name).exists():
            return False
    
    # 检查journal.json是否包含有效的实验数据
    try:
        import json
        with open(log_dir / "journal.json", 'r') as f:
            journal_data = json.load(f)
        
        # 检查是否有节点数据
        if not journal_data.get("nodes") or len(journal_data["nodes"]) == 0:
            return False
            
        # 检查是否有最佳节点
        nodes = journal_data["nodes"]
        has_good_node = any(not node.get("is_buggy", True) for node in nodes)
        
        return has_good_node
        
    except (json.JSONDecodeError, KeyError, FileNotFoundError):
        return False


def run_competition(
    competition_id: str,
    data_base_dir: Path,
    desc_base_dir: Path,
    extra_aide_args: List[str],
    timeout: Optional[int] = None,
    thread_id: Optional[int] = None,
    log_base_dir: Optional[Path] = None,
    skip_existing: bool = False,
    seed: Optional[int] = None,
) -> dict:
    """运行单个竞赛"""
    thread_prefix = f"[线程{thread_id}] " if thread_id is not None else ""
    safe_log(f"{thread_prefix}{'='*60}")
    safe_log(f"{thread_prefix}开始运行竞赛: {competition_id}")
    safe_log(f"{thread_prefix}{'='*60}")
    
    # 生成实验名称（competition name + seed）
    if seed is not None:
        exp_name = f"{competition_id}_seed_{seed}"
    else:
        exp_name = competition_id
    
    # 检查是否跳过已存在的日志
    if skip_existing and log_base_dir:
        if check_existing_logs(exp_name, log_base_dir):
            safe_log(f"{thread_prefix}✓ 竞赛 {exp_name} 已存在完整日志，跳过运行")
            return {"success": True, "competition_id": competition_id, "exp_name": exp_name, "skipped": True, "elapsed_time": 0}
    
    # 构建路径
    data_dir = data_base_dir / competition_id / "prepared" / "public"
    desc_file = desc_base_dir / competition_id / "full_instructions.txt"
    
    # 检查文件是否存在
    if not data_dir.exists():
        safe_log(f"{thread_prefix}数据目录不存在: {data_dir}", "error")
        return {"success": False, "error": f"数据目录不存在: {data_dir}"}
    
    if not desc_file.exists():
        safe_log(f"{thread_prefix}Instruction文件不存在: {desc_file}", "error")
        return {"success": False, "error": f"Instruction文件不存在: {desc_file}"}
    
    
    # 构建aide命令
    aide_cmd = [
        "aide",
        f"data_dir={data_dir}",
        f"desc_file={desc_file}",
        f"exp_name={exp_name}",
    ]
    
    # 添加额外参数
    if extra_aide_args:
        aide_cmd.extend(extra_aide_args)
    
    # 如果有timeout，使用timeout命令包装
    if timeout:
        cmd = ["timeout", str(timeout)] + aide_cmd
    else:
        cmd = aide_cmd
    
    safe_log(f"{thread_prefix}执行命令: {' '.join(cmd)}")
    safe_log(f"{thread_prefix}数据目录: {data_dir}")
    safe_log(f"{thread_prefix}Instruction文件: {desc_file}")
    safe_log(f"{thread_prefix}实验名称: {exp_name}")
    
    # 运行aide
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            text=True,
        )
        
        elapsed_time = time.time() - start_time
        success = (result.returncode == 0)
        
        if success:
            safe_log(f"{thread_prefix}✓ 竞赛 {exp_name} 运行成功 (耗时: {elapsed_time:.2f}秒)")
            return {"success": True, "competition_id": competition_id, "exp_name": exp_name, "elapsed_time": elapsed_time}
        else:
            # 检查是否是timeout导致的退出
            if timeout and result.returncode == 124:
                safe_log(f"{thread_prefix}✗ 竞赛 {exp_name} 运行超时 (超过 {timeout}秒)", "error")
                return {"success": True, "error": f"运行超时 (超过 {timeout}秒)", "competition_id": competition_id, "exp_name": exp_name, "elapsed_time": elapsed_time}
            else:
                safe_log(f"{thread_prefix}✗ 竞赛 {exp_name} 运行失败 (返回码: {result.returncode}, 耗时: {elapsed_time:.2f}秒)", "error")
                return {"success": False, "error": f"命令返回非零退出码: {result.returncode}", "competition_id": competition_id, "exp_name": exp_name, "elapsed_time": elapsed_time}
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        safe_log(f"{thread_prefix}✗ 运行竞赛 {exp_name} 时发生错误: {e}", "error")
        return {"success": False, "error": str(e), "competition_id": competition_id, "exp_name": exp_name, "elapsed_time": elapsed_time}


def main():
    parser = argparse.ArgumentParser(
        description="AIDE 批量运行脚本，自动为多个竞赛执行 aide 命令",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认路径运行（串行）
  python run_competitions.py --competition-set competition_sets/example.txt
  
  
  # 指定自定义并行度
  python run_competitions.py \\
      --competition-set competition_sets/full.txt \\
      --max-workers 8
  
  # 指定自定义路径
  python run_competitions.py \\
      --competition-set competition_sets/full.txt \\
      --data-dir /root/datasets2/mle-bench-lite \\
      --desc-dir dataset/full_instructions
  
  # 添加额外的 aide 参数
  python run_competitions.py \\
      --competition-set competition_sets/example.txt \\
      --aide-args "agent.steps=10" "agent.code.model=gpt-4o"
  
  # 并行运行并设置超时
  python run_competitions.py \\
      --competition-set competition_sets/example.txt \\
      --parallel \\
      --timeout 3600
  
  # 跳过已存在完整日志的竞赛
  python run_competitions.py \\
      --competition-set competition_sets/full.txt \\
      --skip-existing \\
      --log-dir logs/20steps
  
  # 每个竞赛运行3次（使用不同随机种子）
  python run_competitions.py \\
      --competition-set competition_sets/example.txt \\
      --n-seed 3 \\
      --log-dir logs/20steps
        """
    )
    
    parser.add_argument(
        "--competition-set",
        type=str,
        required=True,
        help="竞赛列表文件路径（每行一个竞赛ID）",
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/root/datasets2/mle-bench-lite/raw",
        help="数据集根目录（默认: /root/datasets2/mle-bench-lite），每个竞赛数据在 <data-dir>/<competition-id>/ 下",
    )
    
    parser.add_argument(
        "--desc-dir",
        type=str,
        default="dataset/full_instructions",
        help="Instruction文件根目录（默认: dataset/full_instructions），每个竞赛的instruction在 <desc-dir>/<competition-id>/full_instructions.txt",
    )
    
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="每个竞赛的最大运行时间（秒），默认无限制",
    )
    
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="即使某个竞赛失败也继续运行剩余竞赛",
    )
    
    parser.add_argument(
        "--aide-args",
        nargs="*",
        default=[],
        help="传递给aide命令的额外参数（例如: agent.steps=10 agent.code.model=gpt-4o）",
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="并行运行的最大线程数（默认: 1，即串行运行）",
    )
    
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="跳过已存在完整日志的竞赛（默认: ",
    )
    
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/20steps",
        help="日志根目录（默认: logs/20steps），用于检查已存在的实验",
    )
    
    parser.add_argument(
        "--n-seed",
        type=int,
        default=1,
        help="每个竞赛的重复运行次数（默认: 1），每次运行使用不同的随机种子",
    )
    
    
    args = parser.parse_args()
    
    # 处理并行参数
    max_workers = args.max_workers
    
    # 解析路径
    script_dir = Path(__file__).parent
    data_base_dir = Path(args.data_dir).expanduser().resolve()
    desc_base_dir = (script_dir / args.desc_dir).resolve()
    log_base_dir = (script_dir / args.log_dir).resolve()
    
    logger.info(f"数据根目录: {data_base_dir}")
    logger.info(f"Instruction根目录: {desc_base_dir}")
    logger.info(f"日志根目录: {log_base_dir}")
    logger.info(f"并行模式: {'是' if max_workers > 1 else '否'} (最大线程数: {max_workers})")
    logger.info(f"跳过已存在: {'是' if args.skip_existing else '否'}")
    
    # 加载竞赛列表
    competition_set_file = Path(args.competition_set)
    competition_ids = load_competition_ids(competition_set_file)
    
    if not competition_ids:
        logger.error("没有找到有效的竞赛ID")
        sys.exit(1)
    
    # 生成所有任务（包括多次运行）
    all_tasks = []
    for competition_id in competition_ids:
        for seed in range(args.n_seed):
            all_tasks.append({
                "competition_id": competition_id,
                "seed": seed,
                "exp_name": f"{competition_id}_seed_{seed}" if args.n_seed > 1 else competition_id
            })
    
    logger.info(f"将运行 {len(competition_ids)} 个竞赛，每个竞赛运行 {args.n_seed} 次，总共 {len(all_tasks)} 个任务")
    
    # 运行所有任务
    results = []
    start_time = time.time()
    
    if max_workers == 1:
        # 串行执行
        for i, task in enumerate(all_tasks, 1):
            logger.info(f"进度: {i}/{len(all_tasks)} - {task['exp_name']}")
            
            result = run_competition(
                competition_id=task["competition_id"],
                data_base_dir=data_base_dir,
                desc_base_dir=desc_base_dir,
                extra_aide_args=args.aide_args,
                timeout=args.timeout,
                seed=task["seed"],
                log_base_dir=log_base_dir,
                skip_existing=args.skip_existing,
            )
            
            results.append(result)
            
            # 如果失败且不继续运行，则停止
            if not result["success"] and not args.continue_on_error:
                logger.error(f"\n任务 {task['exp_name']} 失败，停止运行")
                logger.error("如需继续运行剩余任务，请使用 --continue-on-error 参数")
                break
    else:
        # 并行执行
        logger.info(f"开始并行执行，使用 {max_workers} 个线程")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_task = {}
            for i, task in enumerate(all_tasks):
                future = executor.submit(
                    run_competition,
                    competition_id=task["competition_id"],
                    data_base_dir=data_base_dir,
                    desc_base_dir=desc_base_dir,
                    extra_aide_args=args.aide_args,
                    timeout=args.timeout,
                    thread_id=i+1,
                    log_base_dir=log_base_dir,
                    skip_existing=args.skip_existing,
                    seed=task["seed"],
                )
                future_to_task[future] = task
            
            # 处理完成的任务
            completed_count = 0
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                completed_count += 1
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    # 显示进度
                    safe_log(f"进度: {completed_count}/{len(all_tasks)} - 任务 {task['exp_name']} 完成")
                    
                    # 如果失败且不继续运行，则取消剩余任务
                    if not result["success"] and not args.continue_on_error:
                        safe_log(f"任务 {task['exp_name']} 失败，取消剩余任务", "error")
                        # 取消所有未完成的任务
                        for f in future_to_task:
                            if not f.done():
                                f.cancel()
                        break
                        
                except Exception as e:
                    safe_log(f"处理任务 {task['exp_name']} 时发生异常: {e}", "error")
                    results.append({
                        "success": False, 
                        "error": str(e), 
                        "competition_id": task["competition_id"],
                        "exp_name": task["exp_name"],
                        "elapsed_time": 0
                    })
        
    
    total_time = time.time() - start_time
    
    # 打印总结
    logger.info(f"\n{'='*80}")
    logger.info("运行总结")
    
    success_count = sum(1 for r in results if r["success"])
    skipped_count = sum(1 for r in results if r.get("skipped", False))
    completed_count = len(results)
    total_tasks = len(all_tasks)
    unique_competitions = len(competition_ids)
    
    logger.info(f"总计竞赛: {unique_competitions} 个")
    logger.info(f"每个竞赛运行次数: {args.n_seed} 次")
    logger.info(f"总任务数: {total_tasks} 个")
    logger.info(f"已完成: {completed_count - skipped_count} 个")
    logger.info(f"成功: {success_count} 个")
    logger.info(f"跳过: {skipped_count} 个")
    logger.info(f"失败: {completed_count - success_count - skipped_count} 个")
    logger.info(f"总耗时: {total_time:.2f}秒 ({total_time/3600:.2f}小时)")
    
    if success_count < completed_count:
        logger.info(f"\n失败的任务:")
        for r in results:
            if not r["success"]:
                exp_name = r.get('exp_name', 'Unknown')
                error = r.get('error', 'Unknown error')
                logger.info(f"  - {exp_name}: {error}")
    
    logger.info(f"{'='*80}\n")
    
    # 如果有失败的任务，返回非零退出码
    if success_count < completed_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
