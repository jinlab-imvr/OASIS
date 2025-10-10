import os
from vos_benchmark.benchmark import benchmark


def auto_benchmark(dataset_dir, results_dir, annotation_suffix="d17-val/Annotations"):
    # 获取所有实验目录路径
    experiment_paths = [
        os.path.join(results_dir, exp, annotation_suffix)
        for exp in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, exp))
    ]

    # 调用 benchmark 函数
    for exp in experiment_paths:
        print(exp)
        benchmark([dataset_dir], [exp], verbose=False)


# 示例使用
auto_benchmark("/mnt/iMVR/guanyi/dataset/DAVIS_17-val", "results/ablation")
