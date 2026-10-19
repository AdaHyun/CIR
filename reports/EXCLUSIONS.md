# 复制/打包边界与排除记录

两个源包的全部实际文件均保留，包括51张案例PNG、源码、环境声明、.out/.err/.log/history、真实大权重和特征。没有因为体积大省略实际模型或特征。已复制内容见source_mapping；当前可访问范围无额外训练压缩备份，见source_discovery。

未将下列不可迁移或无实验内容的机器设施作为项目依赖复制：

- `/workspace/cir-venv`：本机会话原有Linux CPU虚拟环境；不是可移植Windows/Linux环境。版本记录保存在archive/onboarding/dependency-record.txt和各run/config/environment记录。
- `/workspace/.cache/uv`：包下载/解压缓存，可从正常依赖来源再生成；运行不依赖这个路径。
- `/workspace/CIR/.git`：版本库内部元数据；代码身份/git信息在运行快照中。项目在当前main工作区交付，打包不需要嵌入.git。
- Python `__pycache__`、`.pyc`：再生字节码；不纳入交付checksum或打包。没有自动删除这些文件。
- 未完成的 `.partial`：从不作为完成产物或压缩包输入，保留原位供诊断；恢复会验证并重写本次操作的临时文件。
- workspace的受保护 .aws/.codex/.agents 等机器设施与凭证：不是CIR材料，未复制、未输出其内容。

`/workspace/cir-validation` 是本任务前期生成的验证副本，已完整复制到archive/onboarding/validation（含原始PT，未因重复体积省略），前期/最终执行日志亦保留。整理恢复再次运行产生新的日志，不覆盖旧尝试。用户提供的原始粘贴需求保存于archive/requests。

Windows的 D:\Study\cjf 没有挂载，不能声称已扫描或写入。后续可在该机器运行organize.py显式补入真实来源。本次未联网找回HPC文件，没有上传，也没有安装A100环境。
