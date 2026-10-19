# CIR-FULL分支工程改造

此前实体目录先完整保全并校验5705文件，再将主动工程复制到本分支根目录，冗余同名套目录从本分支移出（保全位置记录于ROOT_MIGRATION）；main指向未改。原始两个源包仍在，未删除或改写。此前有效源映射、审计、完整运行/测试产物也保留。

原有RR修正、原子恢复、日志心跳、来源校验保留。新增：

- upstream_model.py：直接移植原Base/DQU/PFP/REFCOND类及参数命名，改为实际设备tokenizer与明确本地、校验过的预训练文件；不编造CN架构。
- raw_data.py：官方train/val、显式source/target/query/gallery、source-caption+修改文本、历史风格OpenCV视觉写字、真实缓存优先/可声明替代、图片与配置指纹。公开CLIP默认只从已验证本地文件加载；fixture backend强制隔离。
- upstream.py：双LR AdamW、原衰减/BN策略、AMP/NCE/soft/conflict、确定性完整训练恢复、分块同源导出/TopK、teacher bank、完整val和短基准；best状态不额外保留GPU深拷贝，latest包含pending-best恢复标记。
- full_cli.py：新主线所有阶段、严格预检查、单卡可见性、明确方法依赖、阶段fingerprint/原子状态、配置overlay、完整统一预测/分析/CSV/SVG；benchmark与正式训练隔离。
- experiments.py：实际prefix-K效果/覆盖/预测/成本、在线真实编码成本，不拿缓存后计时冒充端到端。
- model.py：主RR仍全候选真实点积；已知旧bug只能经显式legacy-control在独立控制目录执行，主Proposed配置拒绝该score规则，比较用同一新CN输入。
- assets.py/engine.py：新producer provenance验证、自定义实际K、新资产原始修改文本输出；旧路径默认Top100及旧推理逻辑保留。
- prepare.py：官方固定JSON与可信hash、公开HF revision/LFS核验和恢复、真实图片/缓存安全导入、本地BLIP可恢复生成。没有付费API或密钥。
- setup_server.sh/requirements-full.txt：由用户未来主动执行的无sudo、匹配CPU/cu121 wheels与普通依赖，当前只安装CPU测试库。
- materialize_old.py/artifacts.py：Git原始核心资产可重建；当前被Git排除的大文件/完整记录都有路径、大小、来源、SHA与显式单独导出方式。
- tests：原RR测试保留；新增完整上游→新RR轻量流水线、真实上游SIGTERM和恢复、K边界/控制实验/在线与训练基准/腐坏/搬迁；可选真实OpenCLIP包CPU接口测试，绝不作为预训练模型性能。

所有源码/配置/环境信息按每次运行保存。旧源码中的绝对历史路径仅作证据，不被新入口调用。详细参数证据、必要工程选择、每阶段输入输出和恢复范围见docs/METHOD_AND_STAGES.md；对HF模型下载及真实A100性能未验证之处明确列出。
