#!/usr/bin/env python3
"""
run_all_evaluations.py - 批量评估所有消融实验结果

用法：
    python run_all_evaluations.py --group 0
    python run_all_evaluations.py --group all
    python run_all_evaluations.py --exp exp0_1_full_finetune
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
import argparse


class EvaluationRunner:
    def __init__(self, base_output_dir="./eval_output/ablation"):
        self.base_output_dir = Path(base_output_dir)
        self.ablation_dir = Path("./ablation_output")
        
        # 数据集配置
        self.images_dir = "data/IRSTD-1k/images/test"
        self.coco_gt = "data/IRSTD-1k/annotations/test.json"
        
        # 模型配置
        self.model_config = "config/cfg_odvg.py"
        self.text_prompt = "Infrared small target"
        self.pretrain_model = "/media/sisu/X/hc/projects/Open-GroundingDino/weights/groundingdino_swint_ogc.pth"
        
        # 评估阈值
        self.box_threshold = 0.001
        self.text_threshold = 0.85
        self.nms_threshold = 0.5
        
        # 定义所有实验
        self.experiments = self._define_experiments()
    
    def _define_experiments(self):
        """定义所有消融实验的配置"""
        return [
            # Group 0: Baseline
            {
                "group": "0",
                "name": "Exp-0.1 Full Fine-tuning",
                "id": "exp0_1_full_finetune",
                "use_lora": False,
                "lora_checkpoint": None,
                "hf_lora_modules": [],
            },
            {
                "group": "0",
                "name": "Exp-0.2 Standard LoRA",
                "id": "exp0_2_standard_lora",
                "use_lora": True,
                "lora_checkpoint": "group0/exp0_2_standard_lora/lora_checkpoint_best",
                "hf_lora_modules": [],
            },
            
            # Group 1: HF-LoRA 消融
            {
                "group": "1",
                "name": "Exp-1.1 HF-LoRA qkv",
                "id": "exp1_1_hf_lora_qkv",
                "use_lora": True,
                "lora_checkpoint": "group1/exp1_1_hf_lora_qkv/lora_checkpoint_best",
                "hf_lora_modules": ["qkv"],
            },
            {
                "group": "1",
                "name": "Exp-1.2 HF-LoRA fc1",
                "id": "exp1_2_hf_lora_fc1",
                "use_lora": True,
                "lora_checkpoint": "group1/exp1_2_hf_lora_fc1/lora_checkpoint_best",
                "hf_lora_modules": ["fc1"],
            },
            {
                "group": "1",
                "name": "Exp-1.3 HF-LoRA fc2",
                "id": "exp1_3_hf_lora_fc2",
                "use_lora": True,
                "lora_checkpoint": "group1/exp1_3_hf_lora_fc2/lora_checkpoint_best",
                "hf_lora_modules": ["fc2"],
            },
            {
                "group": "1",
                "name": "Exp-1.4 HF-LoRA qkv+fc1",
                "id": "exp1_4_hf_lora_qkv_fc1",
                "use_lora": True,
                "lora_checkpoint": "group1/exp1_4_hf_lora_qkv_fc1/lora_checkpoint_best",
                "hf_lora_modules": ["qkv", "fc1"],
            },
            {
                "group": "1",
                "name": "Exp-1.5 HF-LoRA Full",
                "id": "exp1_5_hf_lora_full",
                "use_lora": True,
                "lora_checkpoint": "group1/exp1_5_hf_lora_full/lora_checkpoint_best",
                "hf_lora_modules": ["qkv", "fc1", "fc2"],
            },
            
            # Group 2: 提示词库消融
            {
                "group": "2",
                "name": "Exp-2.1 Prompt Generic",
                "id": "exp2_1_prompt_generic",
                "use_lora": True,
                "lora_checkpoint": "group2/exp2_1_prompt_generic/lora_checkpoint_best",
                "hf_lora_modules": [],
            },
            {
                "group": "2",
                "name": "Exp-2.2 Prompt Gen+App",
                "id": "exp2_2_prompt_gen_app",
                "use_lora": True,
                "lora_checkpoint": "group2/exp2_2_prompt_gen_app/lora_checkpoint_best",
                "hf_lora_modules": [],
            },
            {
                "group": "2",
                "name": "Exp-2.3 Prompt Gen+Phy",
                "id": "exp2_3_prompt_gen_phy",
                "use_lora": True,
                "lora_checkpoint": "group2/exp2_3_prompt_gen_phy/lora_checkpoint_best",
                "hf_lora_modules": [],
            },
            {
                "group": "2",
                "name": "Exp-2.4 Prompt 3cat",
                "id": "exp2_4_prompt_3cat",
                "use_lora": True,
                "lora_checkpoint": "group2/exp2_4_prompt_3cat/lora_checkpoint_best",
                "hf_lora_modules": [],
            },
            {
                "group": "2",
                "name": "Exp-2.5 Prompt 5cat",
                "id": "exp2_5_prompt_5cat",
                "use_lora": True,
                "lora_checkpoint": "group2/exp2_5_prompt_5cat/lora_checkpoint_best",
                "hf_lora_modules": [],
            },
            
            # Group 3: 组合消融
            {
                "group": "3",
                "name": "Exp-3.1 HF+Prompt 3cat",
                "id": "exp3_1_hf_prompt_3cat",
                "use_lora": True,
                "lora_checkpoint": "group3/exp3_1_hf_prompt_3cat/lora_checkpoint_best",
                "hf_lora_modules": ["qkv", "fc1", "fc2"],
            },
            {
                "group": "3",
                "name": "Exp-3.2 HF+Prompt 5cat",
                "id": "exp3_2_hf_prompt_5cat",
                "use_lora": True,
                "lora_checkpoint": "group3/exp3_2_hf_prompt_5cat/lora_checkpoint_best",
                "hf_lora_modules": ["qkv", "fc1", "fc2"],
            },
            
            # Group 4: 超参数消融
            {
                "group": "4",
                "name": "Exp-4.1 r=8",
                "id": "exp4_1_r8",
                "use_lora": True,
                "lora_checkpoint": "group4/exp4_1_r8/lora_checkpoint_best",
                "hf_lora_modules": ["qkv", "fc1", "fc2"],
            },
            {
                "group": "4",
                "name": "Exp-4.2 r=32",
                "id": "exp4_2_r32",
                "use_lora": True,
                "lora_checkpoint": "group4/exp4_2_r32/lora_checkpoint_best",
                "hf_lora_modules": ["qkv", "fc1", "fc2"],
            },
            {
                "group": "4",
                "name": "Exp-4.3 alpha=16",
                "id": "exp4_3_alpha16",
                "use_lora": True,
                "lora_checkpoint": "group4/exp4_3_alpha16/lora_checkpoint_best",
                "hf_lora_modules": ["qkv", "fc1", "fc2"],
            },
            {
                "group": "4",
                "name": "Exp-4.4 alpha=64",
                "id": "exp4_4_alpha64",
                "use_lora": True,
                "lora_checkpoint": "group4/exp4_4_alpha64/lora_checkpoint_best",
                "hf_lora_modules": ["qkv", "fc1", "fc2"],
            },
        ]
    
    def run_evaluation(self, exp):
        """运行单个实验的评估"""
        print("\n" + "=" * 70)
        print(f"Evaluating: {exp['name']}")
        print("=" * 70)
        
        # 检查 LoRA checkpoint 是否存在
        if exp["use_lora"]:
            lora_path = self.ablation_dir / exp["lora_checkpoint"]
            if not lora_path.exists():
                print(f"⚠️  Warning: LoRA checkpoint not found: {lora_path}")
                print(f"   Skipping this experiment...")
                return None
        
        # 创建评估配置
        eval_config = {
            "images_dir": self.images_dir,
            "coco_gt": self.coco_gt,
            "config": self.model_config,
            "text_prompt": self.text_prompt,
            "weights": self.pretrain_model,
            "use_lora": exp["use_lora"],
            "lora_checkpoint": str(self.ablation_dir / exp["lora_checkpoint"]) if exp["use_lora"] else None,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_bias": "none",
            "hf_lora_modules": exp["hf_lora_modules"],
            "box_threshold": self.box_threshold,
            "text_threshold": self.text_threshold,
            "nms_threshold": self.nms_threshold,
            "min_area": 0.0,
            "max_area": 1e18,
            "v5_metric": False,
            "force_single_class": False,
            "output_dir": str(self.base_output_dir / f"group{exp['group']}" / exp["id"]),
            "save_pred_json": True,
            "save_pr_curve_data": True,
        }
        
        # 创建临时评估脚本
        temp_script = Path(f"/tmp/eval_{exp['id']}.py")
        self._write_eval_script(temp_script, eval_config)
        
        # 运行评估
        try:
            result = subprocess.run(
                [sys.executable, str(temp_script)],
                capture_output=True,
                text=True,
                timeout=3600  # 1小时超时
            )
            
            if result.returncode != 0:
                print(f"❌ Error running evaluation for {exp['name']}")
                print(f"   Error: {result.stderr}")
                return None
            
            # 解析输出
            return self._parse_output(result.stdout, exp)
            
        except subprocess.TimeoutExpired:
            print(f"⏱️  Timeout: {exp['name']} took too long")
            return None
        except Exception as e:
            print(f"❌ Exception: {e}")
            return None
        finally:
            # 清理临时文件
            if temp_script.exists():
                temp_script.unlink()
    
    def _write_eval_script(self, script_path, config):
        """写入评估脚本"""
        project_dir = Path.cwd()
        
        # 自定义序列化函数，确保布尔值是Python格式
        def pythonize(obj):
            if isinstance(obj, bool):
                return "True" if obj else "False"
            elif isinstance(obj, str):
                return f'"{obj}"'
            elif isinstance(obj, list):
                return "[" + ", ".join([pythonize(x) for x in obj]) + "]"
            elif isinstance(obj, dict):
                items = []
                for k, v in obj.items():
                    items.append(f'"{k}": {pythonize(v)}')
                return "{" + ", ".join(items) + "}"
            elif obj is None:
                return "None"
            else:
                return str(obj)
        
        cfg_str = pythonize(config)
        
        script_content = f'''import sys
from pathlib import Path

# 添加项目目录到 Python 路径
sys.path.insert(0, "{project_dir}")
sys.path.insert(0, "{project_dir}/eval")

from gdino_eval_core import evaluate_gdino_on_coco
import json
from datetime import datetime
from pathlib import Path

CFG = {cfg_str}

if __name__ == "__main__":
    metrics = evaluate_gdino_on_coco(CFG)
    
    # 保存结果
    out_dir = Path(CFG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    payload = {{
        "time": datetime.now().isoformat(timespec="seconds"),
        "cfg": CFG,
        "metrics": metrics
    }}
    
    (out_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8"
    )
    
    # 打印结果
    print("\\n" + "=" * 60)
    print("Results: " + CFG["output_dir"])
    print("=" * 60)
    print("P: {{metrics['P']:.4f}} | R: {{metrics['R']:.4f}} | F1: {{metrics.get('F1', float('nan')):.4f}}")
    print("mAP@0.5: {{metrics.get('mAP@0.5', float('nan')):.4f}}")
'''
        script_path.write_text(script_content)
    
    def _parse_output(self, output, exp):
        """解析评估输出"""
        try:
            # 读取保存的 metrics.json
            metrics_file = self.base_output_dir / f"group{exp['group']}" / exp["id"] / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                    return data["metrics"]
        except:
            pass
        return None
    
    def run_group(self, group_num):
        """运行指定组的所有实验"""
        group_exps = [e for e in self.experiments if e["group"] == str(group_num)]
        
        print(f"\n{'='*70}")
        print(f"Running Group {group_num}: {len(group_exps)} experiments")
        print(f"{'='*70}")
        
        results = []
        for exp in group_exps:
            result = self.run_evaluation(exp)
            if result:
                results.append({
                    "name": exp["name"],
                    "id": exp["id"],
                    "metrics": result
                })
        
        return results
    
    def run_all(self):
        """运行所有实验"""
        print(f"\n{'='*70}")
        print(f"Running ALL experiments: {len(self.experiments)} total")
        print(f"{'='*70}")
        
        results = []
        for exp in self.experiments:
            result = self.run_evaluation(exp)
            if result:
                results.append({
                    "name": exp["name"],
                    "id": exp["id"],
                    "metrics": result
                })
        
        return results
    
    def run_single(self, exp_id):
        """运行单个实验"""
        for exp in self.experiments:
            if exp["id"] == exp_id:
                result = self.run_evaluation(exp)
                if result:
                    return {
                        "name": exp["name"],
                        "id": exp["id"],
                        "metrics": result
                    }
        return None


def main():
    parser = argparse.ArgumentParser(description="批量评估消融实验")
    parser.add_argument("--group", type=str, help="运行指定组 (0, 1, 2, 3, 4, all)")
    parser.add_argument("--exp", type=str, help="运行单个实验 (如: exp0_1_full_finetune)")
    parser.add_argument("--output_dir", type=str, default="./eval_output/ablation", help="评估结果输出目录")
    
    args = parser.parse_args()
    
    runner = EvaluationRunner(args.output_dir)
    
    if args.exp:
        # 运行单个实验
        print(f"\nRunning single experiment: {args.exp}")
        result = runner.run_single(args.exp)
        if result:
            print(f"\n✓ Completed: {result['name']}")
            print(f"  P: {result['metrics']['P']:.4f}")
            print(f"  R: {result['metrics']['R']:.4f}")
            print(f"  F1: {result['metrics'].get('F1', float('nan')):.4f}")
            print(f"  mAP@0.5: {result['metrics'].get('mAP@0.5', float('nan')):.4f}")
    elif args.group:
        if args.group.lower() == "all":
            # 运行所有实验
            results = runner.run_all()
        else:
            # 运行指定组
            results = runner.run_group(args.group)
        
        # 打印汇总
        print(f"\n{'='*70}")
        print("Evaluation Summary")
        print(f"{'='*70}")
        for r in results:
            print(f"\n{r['name']}:")
            print(f"  P: {r['metrics']['P']:.4f} | R: {r['metrics']['R']:.4f} | F1: {r['metrics'].get('F1', float('nan')):.4f}")
            print(f"  mAP@0.5: {r['metrics'].get('mAP@0.5', float('nan')):.4f}")
        
        print(f"\n{'='*70}")
        print(f"✓ Completed {len(results)} evaluations")
        print(f"  Results saved to: {args.output_dir}")
        print(f"{'='*70}")
    else:
        print("请指定 --group 或 --exp 参数")
        print("示例:")
        print("  python run_all_evaluations.py --group 0")
        print("  python run_all_evaluations.py --group all")
        print("  python run_all_evaluations.py --exp exp0_1_full_finetune")


if __name__ == "__main__":
    main()
