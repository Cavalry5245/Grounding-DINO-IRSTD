#!/usr/bin/env python3
"""
generate_ablation_table.py - 生成消融实验结果表格

用法：
    python generate_ablation_table.py --output_dir ./eval_output/ablation
    python generate_ablation_table.py --output_dir ./eval_output/ablation --save_csv
    python generate_ablation_table.py --output_dir ./eval_output/ablation --save_md
"""

import json
import argparse
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any


class AblationTableGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        
        # 定义所有实验
        self.experiments = self._define_experiments()
    
    def _define_experiments(self):
        """定义所有消融实验"""
        return [
            # Group 0: Baseline
            {"group": "0", "name": "Exp-0.1", "desc": "Full Fine-tuning", "hf_lora": False, "prompt_bank": False, "path": "group0/exp0_1_full_finetune"},
            {"group": "0", "name": "Exp-0.2", "desc": "Standard LoRA", "hf_lora": False, "prompt_bank": False, "path": "group0/exp0_2_standard_lora"},
            
            # Group 1: HF-LoRA 消融
            {"group": "1", "name": "Exp-1.1", "desc": "HF-LoRA (qkv only)", "hf_lora": True, "hf_modules": "qkv", "prompt_bank": False, "path": "group1/exp1_1_hf_lora_qkv"},
            {"group": "1", "name": "Exp-1.2", "desc": "HF-LoRA (fc1 only)", "hf_lora": True, "hf_modules": "fc1", "prompt_bank": False, "path": "group1/exp1_2_hf_lora_fc1"},
            {"group": "1", "name": "Exp-1.3", "desc": "HF-LoRA (fc2 only)", "hf_lora": True, "hf_modules": "fc2", "prompt_bank": False, "path": "group1/exp1_3_hf_lora_fc2"},
            {"group": "1", "name": "Exp-1.4", "desc": "HF-LoRA (qkv+fc1)", "hf_lora": True, "hf_modules": "qkv+fc1", "prompt_bank": False, "path": "group1/exp1_4_hf_lora_qkv_fc1"},
            {"group": "1", "name": "Exp-1.5", "desc": "HF-LoRA (Full)", "hf_lora": True, "hf_modules": "qkv+fc1+fc2", "prompt_bank": False, "path": "group1/exp1_5_hf_lora_full"},
            
            # Group 2: 提示词库消融
            {"group": "2", "name": "Exp-2.1", "desc": "Prompt Bank (Generic)", "hf_lora": False, "prompt_bank": True, "prompt_cats": "generic", "path": "group2/exp2_1_prompt_generic"},
            {"group": "2", "name": "Exp-2.2", "desc": "Prompt Bank (Gen+App)", "hf_lora": False, "prompt_bank": True, "prompt_cats": "gen+app", "path": "group2/exp2_2_prompt_gen_app"},
            {"group": "2", "name": "Exp-2.3", "desc": "Prompt Bank (Gen+Phy)", "hf_lora": False, "prompt_bank": True, "prompt_cats": "gen+phy", "path": "group2/exp2_3_prompt_gen_phy"},
            {"group": "2", "name": "Exp-2.4", "desc": "Prompt Bank (3 cat)", "hf_lora": False, "prompt_bank": True, "prompt_cats": "gen+app+phy", "path": "group2/exp2_4_prompt_3cat"},
            {"group": "2", "name": "Exp-2.5", "desc": "Prompt Bank (5 cat)", "hf_lora": False, "prompt_bank": True, "prompt_cats": "all 5", "path": "group2/exp2_5_prompt_5cat"},
            
            # Group 3: 组合消融
            {"group": "3", "name": "Exp-3.1", "desc": "HF-LoRA + Prompt (3 cat)", "hf_lora": True, "hf_modules": "qkv+fc1+fc2", "prompt_bank": True, "prompt_cats": "gen+app+phy", "path": "group3/exp3_1_hf_prompt_3cat"},
            {"group": "3", "name": "Exp-3.2", "desc": "HF-LoRA + Prompt (5 cat)", "hf_lora": True, "hf_modules": "qkv+fc1+fc2", "prompt_bank": True, "prompt_cats": "all 5", "path": "group3/exp3_2_hf_prompt_5cat"},
            
            # Group 4: 超参数消融
            {"group": "4", "name": "Exp-4.1", "desc": "HF-LoRA r=8", "hf_lora": True, "hf_modules": "qkv+fc1+fc2", "prompt_bank": True, "prompt_cats": "gen+app+phy", "lora_r": 8, "lora_alpha": 16, "path": "group4/exp4_1_r8"},
            {"group": "4", "name": "Exp-4.2", "desc": "HF-LoRA r=32", "hf_lora": True, "hf_modules": "qkv+fc1+fc2", "prompt_bank": True, "prompt_cats": "gen+app+phy", "lora_r": 32, "lora_alpha": 64, "path": "group4/exp4_2_r32"},
            {"group": "4", "name": "Exp-4.3", "desc": "HF-LoRA alpha=16", "hf_lora": True, "hf_modules": "qkv+fc1+fc2", "prompt_bank": True, "prompt_cats": "gen+app+phy", "lora_r": 16, "lora_alpha": 16, "path": "group4/exp4_3_alpha16"},
            {"group": "4", "name": "Exp-4.4", "desc": "HF-LoRA alpha=64", "hf_lora": True, "hf_modules": "qkv+fc1+fc2", "prompt_bank": True, "prompt_cats": "gen+app+phy", "lora_r": 16, "lora_alpha": 64, "path": "group4/exp4_4_alpha64"},
        ]
    
    def collect_results(self):
        """收集所有实验结果"""
        results = []
        
        for exp in self.experiments:
            result = self._extract_result(exp)
            if result:
                results.append(result)
        
        return results
    
    def _extract_result(self, exp: Dict[str, Any]) -> Dict[str, Any]:
        """从实验目录中提取结果"""
        exp_path = self.output_dir / exp["path"]
        
        if not exp_path.exists():
            return None
        
        metrics_file = exp_path / "metrics.json"
        if not metrics_file.exists():
            return None
        
        try:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                metrics = data.get("metrics", {})
                
                result = {
                    "Group": exp["group"],
                    "Experiment": exp["name"],
                    "Description": exp["desc"],
                    "HF-LoRA": "Yes" if exp.get("hf_lora") else "No",
                    "HF-Modules": exp.get("hf_modules", "-"),
                    "Prompt Bank": "Yes" if exp.get("prompt_bank") else "No",
                    "Prompt Cats": exp.get("prompt_cats", "-"),
                    "LoRA-r": exp.get("lora_r", "-"),
                    "LoRA-alpha": exp.get("lora_alpha", "-"),
                }
                
                # 添加评估指标
                result["P"] = metrics.get("P", "-")
                result["R"] = metrics.get("R", "-")
                result["F1"] = metrics.get("F1", "-")
                result["mAP@0.5"] = metrics.get("mAP@0.5", "-")
                result["mAP@0.5:0.95"] = metrics.get("mAP@0.5:0.95", "-")
                result["PD"] = metrics.get("PD", "-")
                result["FA"] = metrics.get("FA", "-")
                
                return result
        except Exception as e:
            print(f"Error reading {metrics_file}: {e}")
            return None
    
    def print_summary(self, results: List[Dict[str, Any]]):
        """打印结果汇总"""
        print("\n" + "=" * 120)
        print("消融实验结果汇总")
        print("=" * 120)
        
        df = pd.DataFrame(results)
        
        # 按组分组显示
        for group in sorted(df["Group"].unique()):
            group_df = df[df["Group"] == group]
            print(f"\n{'=' * 120}")
            print(f"Group {group}")
            print('=' * 120)
            
            # 选择显示的列
            display_cols = ["Experiment", "Description", "HF-LoRA", "HF-Modules", "Prompt Bank", "Prompt Cats", "P", "R", "F1", "mAP@0.5"]
            print(group_df[display_cols].to_string(index=False))
    
    def save_csv(self, results: List[Dict[str, Any]], output_file: str = "ablation_results.csv"):
        """保存结果到CSV"""
        df = pd.DataFrame(results)
        output_path = self.output_dir / output_file
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n结果已保存到: {output_path}")
    
    def save_markdown(self, results: List[Dict[str, Any]], output_file: str = "ablation_results.md"):
        """保存结果到Markdown"""
        df = pd.DataFrame(results)
        output_path = self.output_dir / output_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# 消融实验结果\n\n")
            
            # 按组分组
            for group in sorted(df["Group"].unique()):
                group_df = df[df["Group"] == group]
                f.write(f"## Group {group}\n\n")
                
                # 转换为Markdown表格
                display_cols = ["Experiment", "Description", "HF-LoRA", "HF-Modules", "Prompt Bank", "Prompt Cats", "P", "R", "F1", "mAP@0.5"]
                f.write(group_df[display_cols].to_markdown(index=False))
                f.write("\n\n")
        
        print(f"结果已保存到: {output_path}")
    
    def save_latex(self, results: List[Dict[str, Any]], output_file: str = "ablation_results.tex"):
        """保存结果到LaTeX表格"""
        df = pd.DataFrame(results)
        output_path = self.output_dir / output_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{消融实验结果}\n")
            f.write("\\label{tab:ablation}\n")
            f.write("\\begin{tabular}{lccccccccc}\n")
            f.write("\\hline\n")
            f.write("Exp & Description & HF-LoRA & HF-Modules & Prompt Bank & P & R & F1 & mAP@0.5 \\\\\n")
            f.write("\\hline\n")
            
            for _, row in df.iterrows():
                f.write(f"{row['Experiment']} & {row['Description']} & {row['HF-LoRA']} & {row['HF-Modules']} & {row['Prompt Bank']} & ")
                if row['P'] != '-':
                    f.write(f"{row['P']:.4f} & ")
                else:
                    f.write(f"{row['P']} & ")
                
                if row['R'] != '-':
                    f.write(f"{row['R']:.4f} & ")
                else:
                    f.write(f"{row['R']} & ")
                
                if row['F1'] != '-':
                    f.write(f"{row['F1']:.4f} & ")
                else:
                    f.write(f"{row['F1']} & ")
                
                if row['mAP@0.5'] != '-':
                    f.write(f"{row['mAP@0.5']:.4f}")
                else:
                    f.write(f"{row['mAP@0.5']}")
                
                f.write(" \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        print(f"结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="生成消融实验结果表格")
    parser.add_argument("--output_dir", type=str, default="./eval_output/ablation", help="评估结果输出目录")
    parser.add_argument("--save_csv", action="store_true", help="保存结果到CSV")
    parser.add_argument("--save_md", action="store_true", help="保存结果到Markdown")
    parser.add_argument("--save_tex", action="store_true", help="保存结果到LaTeX")
    parser.add_argument("--all_formats", action="store_true", help="保存所有格式")
    
    args = parser.parse_args()
    
    generator = AblationTableGenerator(args.output_dir)
    results = generator.collect_results()
    
    if not results:
        print("未找到任何评估结果！")
        print(f"请检查目录: {args.output_dir}")
        return
    
    # 打印汇总
    generator.print_summary(results)
    
    # 保存结果
    if args.save_csv or args.all_formats:
        generator.save_csv(results)
    
    if args.save_md or args.all_formats:
        generator.save_markdown(results)
    
    if args.save_tex or args.all_formats:
        generator.save_latex(results)
    
    print(f"\n{'=' * 120}")
    print(f"共收集 {len(results)} 个实验结果")
    print(f"{'=' * 120}")


if __name__ == "__main__":
    main()
