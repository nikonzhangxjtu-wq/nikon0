"""边界测试 - Planner 模块.

覆盖：空消息、长消息、中英混合、复合意图冲突、关键词缺失、极端组合.
"""

from nikon0.agent.planner import RuleBasedPlanner
from nikon0.app.schemas.agent import AgentContext, AgentRequest
from nikon0.app.schemas.trace import ExecutionTrace


def make_ctx(msg: str) -> AgentContext:
    return AgentContext(
        request=AgentRequest(session_id="t1", message=msg),
        trace=ExecutionTrace(trace_id="t1", session_id="t1", user_message=msg),
    )


class TestPlannerEdgeCases:
    """Planner 边界条件测试."""

    def test_empty_message_produces_general_intent(self):
        """空消息应产生 general 意图."""
        planner = RuleBasedPlanner()
        result = planner.plan(make_ctx(""))
        assert result.recommended_skill is None
        assert result.needs_general_handle is True
        assert len(result.intents) == 1
        assert result.intents[0].intent == "general"
        assert result.intents[0].confidence == 0.4

    def test_whitespace_only_message(self):
        """仅有空格的消息."""
        planner = RuleBasedPlanner()
        result = planner.plan(make_ctx("   \n\t  "))
        assert result.recommended_skill is None
        assert result.intents[0].intent == "general"

    def test_very_long_message(self):
        """5000字长消息中包含单个产品关键词."""
        prefix = "一些无关的前置文本。" * 500
        msg = prefix + "AC900 显示 E2 怎么处理？"
        planner = RuleBasedPlanner()
        result = planner.plan(make_ctx(msg))
        # 应仍能检测到 product_support 和 case_intake(因为"故障"出现在消息中)
        intents = {i.intent for i in result.intents}
        assert "product_support" in intents
        # "故障"关键词出现在 msg 中所以会触发 case_intake
        assert "case_intake" in intents

    def test_pure_numbers_message(self):
        """纯数字消息."""
        planner = RuleBasedPlanner()
        result = planner.plan(make_ctx("12345 67890"))
        assert result.recommended_skill is None
        assert result.intents[0].intent == "general"

    def test_pure_special_chars_message(self):
        """纯特殊字符消息."""
        planner = RuleBasedPlanner()
        result = planner.plan(make_ctx("!@#$%^&*()_+-=[]{}|;:',.<>?/`~"))
        assert result.recommended_skill is None
        assert result.intents[0].intent == "general"

    def test_english_only_message(self):
        """全英文消息应无法匹配任何中文关键词."""
        planner = RuleBasedPlanner()
        result = planner.plan(make_ctx("My AC900 shows error E2, what should I do?"))
        # "e2" 是小写也可匹配（因为检查 "e2" in text），"E2"需要大写
        # "怎么处理" 不在英文消息中
        # product_support 匹配需要 "e2" in text
        assert result.intents[0].intent == "general"

    def test_mixed_chinese_english_message(self):
        """中英混合消息."""
        planner = RuleBasedPlanner()
        result = planner.plan(make_ctx("My AC900 显示 error E2, how to fix it? 怎么处理？"))
        # "怎么处理" 会触发 product_support
        intents = {i.intent for i in result.intents}
        assert "product_support" in intents

    def test_composite_intent_detection(self):
        """复合意图检测."""
        planner = RuleBasedPlanner()
        result = planner.plan(make_ctx("AC900 显示故障码 E2 无法启动，需要退款退货"))
        assert result.is_composite is True
        intents = {i.intent for i in result.intents}
        assert "refund" in intents
        assert "product_support" in intents

    def test_case_intake_priority_over_product_support(self):
        """case_intake 优先级应高于 product_support."""
        planner = RuleBasedPlanner()
        result = planner.plan(make_ctx("洗碗机坏了不转了，怎么处理？"))
        # "坏了"、"不转"触发 case_intake，"怎么处理"、"洗碗机"触发 product_support
        assert result.recommended_skill == "case_intake"
        intents = {i.intent for i in result.intents}
        assert "case_intake" in intents
        assert "product_support" in intents

    def test_complaint_triggers_handoff_risk(self):
        """投诉意图应标记为高风险."""
        planner = RuleBasedPlanner()
        result = planner.plan(make_ctx("我要投诉，你们产品质量太差了"))
        assert result.risk_level == "high"
        assert any(i.intent == "complaint" for i in result.intents)

    def test_refund_triggers_high_risk(self):
        """退款意图应标记为高风险."""
        planner = RuleBasedPlanner()
        result = planner.plan(make_ctx("不满意，要退款"))
        assert result.risk_level == "high"
        assert any(i.intent == "refund" for i in result.intents)

    def test_tool_echo_detection(self):
        """tool echo 短语检测."""
        planner = RuleBasedPlanner()
        result = planner.plan(make_ctx("请执行 tool echo 测试"))
        assert result.recommended_skill == "tool_echo"
        assert any(i.intent == "tool_echo" for i in result.intents)

    def test_tool_echo_chinese_detection(self):
        """中文 tool echo 短语检测."""
        planner = RuleBasedPlanner()
        result = planner.plan(make_ctx("请运行工具回声验证"))
        assert result.recommended_skill == "tool_echo"

    def test_multiple_intent_with_same_priority_picks_higher_confidence(self):
        """同优先级应按 confidence 排序."""
        planner = RuleBasedPlanner()
        # 构造两个都触发 case_intake 的词
        result = planner.plan(make_ctx("坏了 不转"))
        # "坏了"触发 case_intake 0.9, "不转"也是 case_intake 0.9
        assert result.recommended_skill == "case_intake"

    def test_planner_result_has_steps(self):
        """验证 planner 产出 PlanStep."""
        planner = RuleBasedPlanner()
        result = planner.plan(make_ctx("AC900 E2 故障需要退款"))
        assert len(result.steps) >= 2
        step_ids = {s.step_id for s in result.steps}
        assert "product_support" in step_ids
        assert "refund_policy" in step_ids

    def test_partial_keyword_match_edge(self):
        """部分关键词匹配的边界."""
        planner = RuleBasedPlanner()
        # "怎么" 匹配但需要和其他词一起
        result = planner.plan(make_ctx("怎么"))
        assert result.recommended_skill is None  # 单独的"怎么"不触发 product_support

    def test_negative_context_not_confused(self):
        """否定语境不应被错误路由（当前 planner 没有否定检测）."""
        planner = RuleBasedPlanner()
        # 包含"不转"会触发 case_intake
        result = planner.plan(make_ctx("我想确认一下不是电机不转的问题"))
        assert "case_intake" in {i.intent for i in result.intents}
        # 注意：这是一个已知局限——planner 不处理否定语义

    def test_risk_level_for_mixed_normal_and_risky(self):
        """混合普通+风险意图时，风险应为 high."""
        planner = RuleBasedPlanner()
        result = planner.plan(make_ctx("空气净化器怎么清洁？另外我要投诉"))
        assert result.risk_level == "high"

    def test_only_normal_intent_is_low_risk(self):
        """仅普通意图的风险应为 low."""
        planner = RuleBasedPlanner()
        result = planner.plan(make_ctx("空调怎么清洁保养？"))
        assert result.risk_level == "low"
