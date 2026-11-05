# 仓库工作约定

先读 docs/README.md 和 docs/reading-guide.md；目标尚未映射到专项入口时，使用 skills/hcu-plugin-assistant/SKILL.md 检索并补齐必要输入；已明确任务时直接选一个主要专项 skills/*/SKILL.md。安装看 docs/installation.md（hcu-plugin-install），工程初始化看 docs/first-plugin.md（hcu-plugin-scaffold），能力开发看 docs/plugins.md（hcu-plugin-dev），故障诊断看 docs/troubleshooting.md（hcu-plugin-triage）。各功能文档含本任务 Skill 调用示例。
本仓按 plugin-v3-1028 要求已包含 hcu_sglang 运行时与 pyproject.toml；方法、Skill、工具和代码分目录维护。旧 output-code/sglang-plugin 是迁入来源，不能继续解释为本仓禁止放运行时代码。新建其他独立插件或生成 embedded 产物仍使用指定的新目录。每次运行时代码修改更新 docs/hcu-sglang-development.md，记录步骤、提交范围、实际验证和未验证项；原始迁入清单不伪装成未来代码完整性清单。
adapter 与 plugin 独立开发、运行时可组合；实现归属按 docs/governance.md 拆分，不能把两仓部署解释成二选一。跨仓目标、加载顺序、配置及有效基线回退必须单独验证。
文档按 docs/README.md 的术语与编写规范统一标题、用途、范围、执行/检查和当前/规划状态，新增入口同步导航与 Skill；功能文档需说明对应 Skill 的输入、调用及交付边界。
共享文件使用相对路径或显式路径变量；本机位置放 ignored .env，配置方式见 docs/configuration.md。历史来源位置使用脱敏占位符，保留原始版本、文件哈希与验证边界；不改写 ignored 原始日志。
修改工具后运行 python3 ci/l0_gate.py 与 python3 -m unittest discover -s tests -v（Python 3.10+）。文档/skill 修改执行结构、链接/锚点和元信息检查；共享内容修改后用 python3 tools/build_guide.py 重建导览，再用 --check 检查快照；图文导览还需检查实际浏览器渲染、交互与窄屏布局；检查受阻时记录原因和未验证项，不以 DOM 检查代替截图验收。
审计可记录脏输入，回溯必须拒绝脏输入；不清理来源仓库来迎合工具。历史运行证据不能记作本次验证，V1 来源 JSON 保留快照语义。
协作、提交与证据要求见 docs/contributing.md；当前门禁和后续规划见 docs/ci.md。

生成、抽取或修改外部 plugin/embedded 代码时，不新增说明性注释、生成标记、来源说明 docstring 或 lint 抑制注释。保留来源已有注释、docstring、版权和许可声明；来源、差异与验证说明写入独立文档或 manifest。具体规则见 docs/contributing.md 的“生成代码约定”。

## adapter/plugin 适配与全量整理约定

- 先读本仓 `docs/governance.md`，按固定社区版本逐项判断归属：参照平台已有行为的 HCU 兼容归 adapter；上游没有的新机制、融合和调度归 plugin。VP、LayerSplit 的新增逻辑必须覆盖配置、注册、生产者、消费者和生命周期，不能只迁移核心类。社区源码保持原样；缺安全接缝时明确记录 integration gap。
- 开始修改前固定社区、来源镜像、adapter、plugin 的提交及实际安装路径、文件哈希、依赖 ABI。指定镜像内的实现优先；不能用远端最新源码冒充镜像实现。保留原始版权与许可，来源和有意差异记录在独立文档。
- 先取得原生完整 prefill/decode profile，适用时覆盖 MTP 和 graph，核查实际处理请求的各并行组和 rank。逐项建立框架入口 → adapter/plugin 分发 → 库/API/设备内核的真实调用对照；仅导入成功、名称相同、启动成功或局部 kernel 命中均不代表对齐。
- 同时核对 shape/stride、dtype、权重/缓存布局、激活量化、scale 计算与应用、累加精度、残差、舍入和索引顺序。只对已明确差异做最小修改，以同一实际输入核对输出及必要中间量；graph 必须验证更换输入后的重放。禁止把 W4A8 降级成 W4A16，禁止擅自更换数学策略。
- 全量整理覆盖本次适配全部差异，不止用户点名的功能。逐项记录归属、入口、所有消费者、当前有效 original 回退、旧别名、加载顺序、环境变量、重复实现和 fork-only 依赖；不靠整模块 globals 复制隐藏依赖，不从 fork 动态加载业务实现。其他模型的遗留分支须有单独证据，不能为清理而直接删除。
- 算子对照清单使用“已对齐 / 存在差异 / 未验证”，绑定版本、profile、输入输出证据；静态源码一致另记，不能替代运行对齐。未关闭的算子项先补齐，不进入整模型反复试错或性能调参。
- 按项目 CI 的命令、配置和工具版本对本次修改运行 lint；检查自动修复 diff，保留动态注册等运行时引用。另做 Python 语法检查和 `git diff --check`。用户未要求时不新增单测，不用完整测试套件替代 lint；远端 CI 在 GitHub 执行，失败再修复。
- 保留并行工作区的未提交修改，隔离整理与正在执行的测试。完整代码整理、台账和 lint 收尾后，才交付固定提交及验证清单；由测试方先做组件对拍和 graph，再做整模型验证。不得提前通知切换，或将未完成状态写成“可验收”。测试通过后通过 PR 集成，不直接推送受保护/共享基线。
- 分支不用 `codex/`。优先遵循仓库规范，否则用 `feat/`、`fix/`、`refactor/`、`docs/`、`chore/` 或 `tools/` 加简短主题。
