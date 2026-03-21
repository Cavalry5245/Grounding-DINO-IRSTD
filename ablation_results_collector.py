#!/usr/bin/env python3
"""
ablation_results_collector.py - 消融实验结果收集和汇总脚本

用法：
    python ablation_results_collector.py --output_dir ./ablation_output
    python ablation_results_collector.py --output_dir ./ablation_output --save_csv
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd


class AblationResultsCollector:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.results = []
        
    def collect_results(self):
        """收集所有消融实验的结果"""
        
        # 定义所有实验的配置
        experiments = [
            # Group 0: Baseline
            {"group": "0", "name": "Exp-0.1", "desc": "Full Fine-tuning", "hf_lora": False, "prompt_bank": False, "path": "ablation_output/group0/exp0_1_full_finetune"},
            {"group": "0", "name": "Exp-0.2", "desc": "Standard LoRA", "hf_lora": False, "prompt_bank": False, "path": "ablation_output/group0/exp0_2_standard_lora"},
            
            # Group 1: HF-LoRA 消融
            {"group": "1", "name": "Exp-1.1", "desc": "HF-LoRA (qkv only)", "hf_lora": True, "hf_modules": "qkv", "prompt_bank": False, "path": "ablation_output/group1/exp1_1_hf_lora_qkv"},
            {"group": "1", "name": "Exp-1.2", "desc": "HF-LoRA (fc1 only)", "hf_lora": True, "hf_modules": "fc1", "prompt_bank": False, "path": "ablation_output/group1/exp1_2_hf_lora_fc1"},
            {"group": "1", "name": "Exp-1.3", "desc": "HF-LoRA (fc2 only)", "hf_lora": True, "hf_modules": "fc2", "prompt_bank": False, "path": "ablation_output/group1/exp1_3_hf_lora_fc2"},
            {"group": "1", "name": "Exp-1.4", "desc": "HF-LoRA (qkv+fc1)", "hf_lora": True, "hf_modules": "qkv+fc1", "prompt_bank": False, "path": "ablation_output/group1/exp1_4_hf_lora_qkv_fc1"},
            {"group": "1", "name": "Exp-1.5", "desc": "HF-LoRA (Full)", "hf_lora": True, "hf_modules": "qkv+fc1+fc2", "prompt_bank": False, "path": "ablation_output/group1/exp1_5_hf_lora_full"},
            
            # Group 2: 提示词库消融
            {"group": "2", "name": "Exp-2.1", "desc": "Prompt Bank (Generic)", "hf_lora": False, "prompt_bank": True, "prompt_cats": "generic", "path": "ablation_output/group2/exp2_1_prompt_generic"},
            {"group": "2", "name": "Exp-2.2", "desc": "Prompt Bank (Gen+App)", "hf_lora": False, "prompt_bank": True, "prompt_cats": "generic+appearance", "path": "ablation_output/group2/exp2_2_prompt_gen_app"},
            {"group": "2", "name": "Exp-2.3", "desc": "Prompt Bank (Gen+Phy)", "hf_lora": False, "prompt_bank": True, "prompt_cats": "generic+physical", "path": "ablation_output/group2/exp2_3_prompt_gen_phy"},
            {"group": "2", "name": "Exp-2.4", "desc": "Prompt Bank (3 cat)", "hf_lora": False, "prompt_bank": True, "prompt_cats": "gen+app+phy", "path": "ablation_output/group2/exp2_4_prompt_3cat"},
            {"group": "2", "name": "Exp-2.5", "desc": "Prompt Bank (5 cat)", "hf_lora": False, "prompt_bank": True, "prompt_cats": "all 5", "path": "ablation_output/group2/exp2_5_prompt_5cat"},
            
            # Group 3: 组合消融
            {"group": "3", "name": "Exp-3.1", "desc": "HF-LoRA + Prompt (3 cat)", "hf_lora": True, "hf_modules": "qkv+fc1+fc2", "prompt_bank": True, "prompt_cats": "gen+app+phy", "path": "ablation_output/group3/exp3_1_hf_prompt_3cat"},
            {"group": "3", "name": "Exp-3.2", "desc": "HF-LoRA + Prompt (5 cat)", "hf_lora": True, "hf_modules": "qkv+fc1+fc2", "prompt_bank": True, "prompt_cats": "all 5", "path": "ablation_output/group3/exp3_2_hf_prompt_5cat"},
            
            # Group 4: 超参数消融
            {"group": "4", "name": "Exp-4.1", "desc": "HF-LoRA r=8, alpha=16", "hf_lora": True, "hf_modules": "qkv+fc1+fc2", "prompt_bank": True, "prompt_cats": "gen+app+phy", "lora_r": 8, "lora_alpha": 16, "path": "ablation_output/group4/exp4_1_r8"},
            {"group": "4", "name": "Exp-4.2", "desc": "HF-LoRA r=32, alpha=64", "hf_lora": True, "hf_modules": "qkv+fc1+fc2", "prompt_bank": True, "prompt_cats": "gen+app+phy", "lora_r": 32, "lora_alpha": 64, "path": "ablation_output/group4/exp4_2_r32"},
            {"group": "4", "name": "Exp-4.3", "desc": "HF-LoRA r=16, alpha=16", "hf_lora": True, "hf_modules": "qkv+fc1+fc2", "prompt_bank": True, "prompt_cats": "gen+app+phy", "lora_r": 16, "lora_alpha": 16, "path": "ablation_output/group4/exp4_3_alpha16"},
            {"group": "4", "name": "Exp-4.4", "desc": "HF-LoRA r=16, alpha=64", "hf_lora": True, "hf_modules": "qkv+fc1+fc2", "prompt_bank": True, "prompt_cats": "gen+app+phy", "lora_r": 16, "lora_alpha": 64, "path": "ablation_output/group4/exp4_4_alpha64"},
        ]
        
        for exp in experiments:
            result = self._extract_experiment_result(exp)
            if result:
                self.results.append(result)
        
        return self.results
    
    def _extract_experiment_result(self, exp: Dict[str, Any]) -> Dict[str, Any]:
        """从实验目录中提取结果"""
        exp_path = self.output_dir.parent / exp["path"]
        
        if not exp_path.exists():
            return None
        
        result = {
            "Group": exp["group"],
            "Experiment": exp["name"],
            "Description": exp["desc"],
            "HF-LoRA": "Yes" if exp.get("hf_lora") else "No",
            "HF-Modules": exp.get("hf_modules", "-"),
            "Prompt Bank": "Yes" if exp.get("prompt_bank") else "No",
            "Prompt Cats": exp.get("prompt_cats", "-"),
            "LoRA-r": exp.get("lora_r", 16),
            "LoRA-alpha": exp.get("lora_alpha", 32),
        }
        
        # 尝试从 info.txt 中提取最佳 mAP
        info_file = exp_path / "info.txt"
        if info_file.exists():
            best_map = self._extract_best_map(info_file)
            result["Best mAP"] = best_map
        
        # 尝试从评估结果中提取指标
        eval_output = exp_path / "eval_output" / "metrics.json"
        if eval_output.exists():
            with open(eval_output, 'r') as f:
                eval_metrics = json.load(f)
                result["P"] = eval_metrics.get("P", "-")
                result["R"] = eval_metrics.get("R", "-")
                result["F1"] = eval_metrics.get("F1", "-")
                result["PD"] = eval_metrics.get("PD", "-")
                result["FA"] = eval_metrics.get("FA", "-")
                result["mAP@0.5"] = eval_metrics.get("mAP@0.5", "-")
                result["mAP@0.5:0.95"] = eval_metrics.get("mAP@0.5:0.95", "-")
        
        return result
    
    def _extract_best_map(self, info_file: Path) -> str:
        """从 info.txt 中提取最佳 mAP"""
        try:
            with open(info_file, 'r') as f:
                for line in f:
                    if "Best mAP:" in line:
                        parts = line.split("Best mAP:")
                        if len(parts) > 1:
                            map_value = parts[1].strip().split()[0]
                            return map_value
        except:
            pass
        return "-"
    
    def print_summary(self):
        """打印结果汇总"""
        print("\n" + "=" * 120)
        print("消融实验结果汇总")
        print("=" * 120)
        
        df = pd.DataFrame(self.results)
        
        # 按组分组显示
        for group in sorted(df["Group"].unique()):
            group_df = df[df["Group"] == group]
            print(f"\n{'=' * 120}")
            print(f"Group {group}")
            print('=' * 120)
            
            display_cols = ["Experiment", "Description", "HF-LoRA", "HF-Modules", "Prompt Bank", "Prompt Cats", "Best mAP", "mAP@0.5", "F1"]
            print(group_df[display_cols].to_string(index=False))
    
    def save_csv(self, output_file: str = "ablation_results.csv"):
        """保存结果到 CSV"""
        df = pd.DataFrame(self.results)
        output_path = self.output_dir / output_file
        df.to_csv(output_path, index=False)
        print(f"\n结果已保存到: {output_path}")
    
    def save_markdown(self, output_file: str = "ablation_results.md"):
        """保存结果到 Markdown"""
        df = pd.DataFrame(self.results)
        output_path = self.output_dir / output_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# 消融实验结果\n\n")
            
            for group in sorted(df["Group"].unique()):
                group_df = df[df["Group"] == group]
                f.write(f"## Group {group}\n\n")
                f.write(group_df.to_markdown(index=False))
                f.write("\n\n")
        
        print(f"结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="收集和汇总消融实验结果")
    parser.add_argument("--output_dir", type=str, default="./ablation_output", help="消融实验输出目录")
    parser.add_argument("--save_csv", action="store_true", help="保存结果到 CSV")
    parser.add_argument("--save_md", action="store_true", help="保存结果到 Markdown")
    
    args = parser.parse_args()
    
    collector = AblationResultsCollector(args.output_dir)
    collector.collect_results()
    collector.print_summary()
    
    if args.save_csv:
        collector.save_csv()
    
    if args.save_md:
        collector.save_markdown()


if __name__ == "__main__":
    main()
