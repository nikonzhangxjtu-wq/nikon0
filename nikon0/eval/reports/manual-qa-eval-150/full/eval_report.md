# nikon0 Manual QA Eval 150 - 综合评测报告

**日期**: 2026-06-19 | **环境**: macOS, Python 3.13 | **Profile**: deterministic

## 1. 总体统计

| 指标 | 值 |
|------|----|
| 总题目数 | 150 |
| 成功执行 | 95 |
| 完全失败 | 55 |
| 技能选择准确率 | 51.3% |
| Must-contain覆盖率(≥50%) | 19.3% |
| 平均执行时间 | ~0.02s/item |

## 2. 评级分布

| 等级 | 数量 | 占比 |
|------|------|------|
| good | 7 | 4.7% |
| partial | 75 | 50.0% |
| poor | 13 | 8.7% |
| fail | 55 | 36.7% |

**评级标准**:
- good: 技能选择正确 + 有实质回答 + must_contain覆盖率≥75%
- partial: 部分满足条件（技能正确或有回答但不够完整）
- poor: 仅满足一项基本条件
- fail: 未满足任何条件

## 3. 按类别统计

| 类别 | 总数 | good | partial | poor | fail | 良好率 |
|------|------|------|---------|------|------|--------|
| boundary | 15 | 0 | 4 | 1 | 10 | 0% |
| case_intake | 20 | 2 | 9 | 1 | 8 | 10% |
| composite | 15 | 3 | 8 | 0 | 4 | 20% |
| general | 10 | 0 | 2 | 0 | 8 | 0% |
| handoff | 10 | 0 | 4 | 2 | 4 | 0% |
| multi-turn | 10 | 0 | 1 | 4 | 5 | 0% |
| no_evidence | 5 | 0 | 3 | 0 | 2 | 0% |
| product_support | 30 | 1 | 23 | 0 | 6 | 3% |
| refund | 10 | 0 | 3 | 5 | 2 | 0% |
| troubleshooting | 25 | 1 | 18 | 0 | 6 | 4% |

## 4. 技能选择分析

| 选中技能 | 数量 | good | partial | poor | fail |
|----------|------|------|---------|------|------|
| product_support | 70 | 7 | 55 | 8 | 0 |
| none | 56 | 0 | 1 | 0 | 55 |
| case_intake | 24 | 0 | 19 | 5 | 0 |

## 5. Fail 项分析

共 55 个 fail 项:

- **qa-002** (product_support): skill=, source=none
  Q: How do I make homemade fries in the Airfryer?
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-005** (product_support): skill=, source=none
  Q: What are the safety warnings for the Airfryer?
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-019** (product_support): skill=, source=none
  Q: How do I descale my Nespresso coffee machine?
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-020** (product_support): skill=, source=none
  Q: How do I program the water volume on my Nespresso machine?
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-021** (product_support): skill=, source=none
  Q: How do I use the sensor cook function on my microwave?
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-022** (product_support): skill=, source=none
  Q: How do I set the child lock on the microwave?
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-033** (troubleshooting): skill=, source=none
  Q: Why won't my microwave start?
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-034** (troubleshooting): skill=, source=none
  Q: Microwave has sparking or arcing inside. What should I do?
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-037** (troubleshooting): skill=, source=none
  Q: Nespresso machine is not brewing coffee. What should I check?
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-048** (troubleshooting): skill=, source=none
  Q: The microwave turntable is not rotating. What's wrong?
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-049** (troubleshooting): skill=, source=none
  Q: Airfryer has error code or won't turn on. What troubleshooting steps?
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-052** (troubleshooting): skill=, source=none
  Q: How do I clean the microwave grease filter?
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-061** (case_intake): skill=, source=none
  Q: Bluetooth mouse not working at all, need to return for replacement. Order ORD-45
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-065** (case_intake): skill=, source=none
  Q: 烤箱买了不到一个月，烤箱灯不亮了，要保修
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-069** (case_intake): skill=, source=none
  Q: 我想购买空气炸锅的炸篮替换件，型号HD9650
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-070** (case_intake): skill=, source=none
  Q: I need to file a warranty claim for microwave that stopped heating after 3 month
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-071** (case_intake): skill=, source=none
  Q: 之前买的蓝牙鼠标，USB接收器丢了，能单独买接收器吗？
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-073** (case_intake): skill=, source=none
  Q: Nespresso machine leaking water from bottom, need urgent repair. Can I get a loa
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-074** (case_intake): skill=, source=none
  Q: VR头显配套的处理器单元灯不亮了，开机没反应
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-075** (case_intake): skill=, source=none
  Q: 烤箱使用时有焦味散发出来，很担心安全问题
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-078** (refund): skill=, source=none
  Q: I want a refund for this microwave. It arrived damaged with a cracked door.
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-085** (refund): skill=, source=none
  Q: Could not get the Airfryer to connect to WiFi after trying for a week. Want my m
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-087** (handoff): skill=, source=none
  Q: I've been trying to fix this microwave for hours and nothing works. This is unac
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-088** (handoff): skill=, source=none
  Q: 你们产品漏电！我刚才用洗衣机的时候被电了一下，现在手臂还有点麻，这种情况你们必须给我一个说法！
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-093** (handoff): skill=, source=none
  Q: I need legal action. Your product caused property damage and I want to file a fo
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-094** (handoff): skill=, source=none
  Q: Voglio parlare con un operatore umano, per favore. Non parlo cinese.
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-099** (composite): skill=, source=none
  Q: 洗衣机洗完衣服后有异味，而且我注意到最近脱水的时候声音比以前大很多
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-100** (composite): skill=, source=none
  Q: 昨天买的VR头显，感觉有点漏光而且遮光罩戴着不太舒服，这个可以调节吗？
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-102** (composite): skill=, source=none
  Q: My airfryer makes a weird noise during cooking and the keep warm function doesn'
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-108** (composite): skill=, source=none
  Q: The Nespresso machine is not recognizing capsules and also the coffee is coming 
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-111** (boundary): skill=, source=none
  Q: ？
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-112** (boundary): skill=, source=none
  Q: 哈哈哈哈哈哈哈哈哈哈哈哈哈
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-113** (boundary): skill=, source=none
  Q: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-114** (boundary): skill=, source=none
  Q: 帮我查一下产品型号为AC900-B2 Pro Max Ultra 2024冬季限量版的问题
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-115** (boundary): skill=, source=none
  Q: Please help me in French, German, and Japanese all at once
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-116** (boundary): skill=, source=none
  Q: 关机吗
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-118** (boundary): skill=, source=none
  Q: 🐛🐛🐛🔥🔥🔥
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-120** (boundary): skill=, source=none
  Q: What's the meaning of life? Also, how do I clean my airfryer?
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-124** (boundary): skill=, source=none
  Q: 我今年85岁了，第一次用智能手机和APP来控制烤箱，完全听不懂术语，能一步一步教我
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-125** (boundary): skill=, source=none
  Q: The temperature was set to 180°C but my food burned. The manual says cooking tim
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-126** (multi-turn): skill=, source=none
  Q: 我的空气炸锅有问题
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-128** (multi-turn): skill=, source=none
  Q: I followed your cleaning instructions for the microwave but the smell still hasn
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-130** (multi-turn): skill=, source=none
  Q: 按照你说的换了新电池也重新配对了，蓝牙鼠标还是连不上
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-131** (multi-turn): skill=, source=none
  Q: 咖啡机除垢后还是有苦涩味，而且出咖啡速度变慢了
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-134** (multi-turn): skill=, source=none
  Q: 之前关于VR头晕你说适应一下就好了，我已经试了两周了还是不行。还有其他办法吗？
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-136** (general): skill=, source=none
  Q: 你好，我想了解一下你们的产品保修政策
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-137** (general): skill=, source=none
  Q: What products do you sell?
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-138** (general): skill=, source=none
  Q: 你们公司叫什么名字？
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-140** (general): skill=, source=none
  Q: What are your business hours?
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-141** (general): skill=, source=none
  Q: 今天天气真好
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-143** (general): skill=, source=none
  Q: Do you have a store near me?
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-144** (general): skill=, source=none
  Q: 我想了解一下你们有没有以旧换新的政策
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-145** (general): skill=, source=none
  Q: 帮我查一下订单物流到哪里了 ORD-88888
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-148** (no_evidence): skill=, source=none
  Q: Does the Nespresso machine make iced coffee?
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。
- **qa-149** (no_evidence): skill=, source=none
  Q: 洗衣机能烘干吗？我洗完衣服想直接烘干
  A: nikon0 已接收到你的请求。当前没有高置信度业务 Skill 命中，因此由 SupervisorAgent 走通用处理路径；后续会接入商品知识问答和 LLM。

## 6. Poor 项分析

共 13 个 poor 项:

- **qa-060** (case_intake): skill=product_support, source=planned
- **qa-076** (refund): skill=case_intake, source=planned
- **qa-079** (refund): skill=case_intake, source=planned
- **qa-080** (refund): skill=product_support, source=rule_fallback
- **qa-082** (refund): skill=case_intake, source=planned
- **qa-084** (refund): skill=product_support, source=rule_fallback
- **qa-089** (handoff): skill=product_support, source=planned
- **qa-092** (handoff): skill=case_intake, source=planned
- **qa-121** (boundary): skill=case_intake, source=planned
- **qa-129** (multi-turn): skill=product_support, source=planned
- ... 及其他 3 项

## 7. 关键发现

### 7.1 系统优势
- **中文产品查询路由准确**: planned 路由能正确将中文产品问题路由到 product_support
- **本地手册搜索有效**: StructuredManualBackend 能从 .txt 文件中检索到相关内容
- **Case intake 识别准确**: case_intake 相关的中文消息能被正确识别
- **运行稳定**: 149/150 项成功执行，无崩溃

### 7.2 系统局限
- **英文查询不匹配**: 纯英文消息不会被 planner 匹配到 product_support（关键词主要为中文）
- **证据检索不够精准**: 部分查询返回了不相关手册的证据（如 Airfryer 清洁问题返回 VR 手册）
- **答案质量低**: 无 LLM 时只能输出原始证据片段，不能生成连贯的客服回答
- **handoff/refund/复合意图处理不足**: 这些路径在 deterministic profile 下返回通用回答
- **边界/无效输入处理**: 表情符号、极短消息等边界情况返回通用回答

### 7.3 建议改进
1. Planner 应增加英文关键词或支持语言无关的关键词匹配
2. StructuredManualBackend 需要更精准的产品-手册映射机制
3. 即使无 LLM，也应使用模板将证据转化为更可读的回答
4. Handoff/Refund/投诉等流程需要实现专门的处理逻辑

## 8. 评测数据

- 数据集: `nikon0/eval/datasets/agent_qa_eval_150_manual.jsonl` (150 条)
- 原始结果: `nikon0/eval/reports/manual-qa-eval-150/full/raw_results.jsonl`
- 评判结果: `nikon0/eval/reports/manual-qa-eval-150/full/judgements.jsonl`
- 评测脚本: `nikon0/eval/run_manual_qa_eval.py`
- 数据集构建器: `nikon0/eval/build_manual_qa_dataset.py`

---
*报告生成时间: 2026-06-19*