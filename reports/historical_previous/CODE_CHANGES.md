# 工程改造记录

所有新增/修改均在 CIR-FULL；两个源包原样复制至 archive。没有修改旧源码、依赖清单、权重或历史结果。main 是本地分支，未推送或上传。

| 文件 | 改造与依据 |
|---|---|
| scripts/organize.py | 先审计/空间预检，8MiB分块复制，SHA验证、原子替换、冲突版本保留、重复运行跳过和来源合并；来源为用户整理要求 |
| scripts/cir.py、src/cir_full/cli.py | 工程根路径解析、CLI/config覆盖、CPU/CUDA显式设备、统一阶段/流水线、完整状态与blocked退出码 |
| src/cir_full/model.py | 保留历史train_fashioniq_feature_reranker.py的RR结构，所有候选真实点积，确定性query/epoch采样；禁止正目标固定分数、source混入及重复负样本填充 |
| src/cir_full/assets.py | 读取真实PT schema、ID/归一化/分数/图库检查，旧权重strict加载；将旧pickle中的Path仅作可移植元数据读取，不使用旧路径 |
| src/cir_full/engine.py | 分块旧/新评估、固定融合、多K、train last/best+optimizer/RNG/cursor恢复、原子指标物化、TopK重建、失败/纠正/损害、全查询导出、DQU导入、实际图片可视化、冻结RR成本 |
| src/cir_full/runtime.py | stdout/stderr双写、flush/心跳、异常traceback、信号处理、代码快照/实际环境/输入指纹、阶段尝试历史和已完成输出校验 |
| configs/default.json | 超参以真实旧checkpoint args为准；固定blend由历史评估确认；新增log/checkpoint间隔是工程选择，不声称来自论文 |
| tests/test_workflow.py、test_signal_resume.py | 隔离fixture测试候选/标签/真实分数、连续vs恢复、实际SIGTERM、optimizer/参数/顺序、错误映射/坏块、完成跳过及搬迁 |
| scripts/inventory.py、package.py | 整个工程实际文件manifest、SHA校验后显式本地打包，防止把输出包递归打进项目；无上传/联网功能 |

为了避免破坏历史依赖，原DQU/CN训练与raw-image exporter连同model/datasets/scripts整体保存于各archive/code目录。主动入口只运行已核验的冻结特征RR工作流；原exporter依赖缺失，未虚称已完成原图特征再生成。未将minimal与phase assets自动合并为同一特征空间。

训练恢复以last.pt为事务提交边界：先保存完整状态，再从已提交记录物化指标/best；未提交操作可重算，已提交step不重复。使用显式num_workers=0等价的手动batch遍历，epoch permutation和query采样独立可重建，dropout使用恢复的torch RNG。没有新增AMP/scheduler或distillation，null状态显式存储。

每次运行保存当前主动Python源码和git信息。源码快照对应运行当时版本；早期探索运行也保留，正式对照以delivery-final及其code快照为准。没有安装新的A100依赖。
