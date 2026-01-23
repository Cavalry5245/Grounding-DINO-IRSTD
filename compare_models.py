import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_compare_models(model_paths, save_path='comparison_pr.png'):
    """
    绘制多个模型的PR曲线对比图
    
    Args:
        model_paths: 字典，格式为 {'模型名称': '该模型results.pt的路径'}
        save_path: 图片保存路径
    """
    
    # 设置绘图风格
    plt.figure(figsize=(10, 8))
    # 使用一般色彩循环
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # 标准 Recall 轴
    x_grid = np.linspace(0, 1, 101)
    
    has_valid_data = False

    for i, (name, file_path) in enumerate(model_paths.items()):
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"警告: 文件不存在 {file_path}")
            continue
            
        print(f"正在处理: {name}...")
        try:
            # 加载数据
            results = torch.load(file_path)
            
            # 获取曲线数据
            precision_curves = results.get('precision_curves', [])
            recall_curves = results.get('recall_curves', [])
            
            # 如果没有数据则跳过
            if not precision_curves:
                print(f"  - {name} 无曲线数据")
                continue
                
            # --- 数据插值处理 (核心逻辑) ---
            interp_precisions = []
            for prec, rec in zip(precision_curves, recall_curves):
                if len(prec) == 0 or len(rec) == 0:
                    continue
                
                # 确保 recall 递增
                if rec.shape[0] > 1 and rec[0] > rec[-1]:
                    rec = rec[::-1]
                    prec = prec[::-1]
                elif rec.shape[0] > 1 and not np.all(rec[1:] >= rec[:-1]):
                    sort_idx = np.argsort(rec)
                    rec = rec[sort_idx]
                    prec = prec[sort_idx]
                
                # 插值到标准轴
                p_interp = np.interp(x_grid, rec, prec, left=1.0, right=0.0)
                interp_precisions.append(p_interp)
            
            if not interp_precisions:
                continue

            # 计算平均曲线
            interp_precisions = np.array(interp_precisions)
            mean_precision = interp_precisions.mean(axis=0)
            
            # 获取 mAP@0.5 用于图例
            map50 = results.get('ap50', 0.0)
            
            # --- 绘图 ---
            color = colors[i % len(colors)]
            plt.plot(x_grid, mean_precision, linewidth=2, color=color,
                     label=f'{name} (mAP@0.5={map50:.3f})')
            
            has_valid_data = True
            
        except Exception as e:
            print(f"处理 {name} 时出错: {e}")

    if not has_valid_data:
        print("未绘制任何曲线，请检查数据文件。")
        return

    # 设置图表装饰
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.title('Model Comparison: Precision-Recall Curve', fontsize=14)
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 保存
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n对比图已保存至: {save_path}")

# ================= 使用示例 =================
if __name__ == '__main__':
    # 1. 在这里填入你的模型名称和对应的 results.pt 路径
    # 注意：路径要是刚才第一步修改代码后，重新跑出来的 .pt 文件路径
    models_to_compare = {
        '0120exp1': 'eval_output/0121_exp1/results.pt',
        '0119exp1': 'eval_output/0119_exp1/results.pt',
    }
    
    # 2. 运行函数
    # 确保文件路径存在，否则会报错
    # 如果你还没有跑出数据，先去跑 eval，然后再运行这个脚本
    
    # 为了测试，你可以先注释掉上面的 dict，手动指向一个真实存在的路径
    # plot_compare_models({'MyModel': 'path/to/your/results.pt'}) 
    
    # print("请配置好 models_to_compare 里的路径后运行脚本。")
    plot_compare_models(models_to_compare)