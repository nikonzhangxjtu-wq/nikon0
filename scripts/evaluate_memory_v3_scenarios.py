"""Memory v3 场景评测脚本。

用法：
    conda run -n kefu python scripts/evaluate_memory_v3_scenarios.py

输出：
    eval/results/memory_v3_scenario_report.json
    eval/results/memory_v3_scenario_report.md

这个脚本不是普通单元测试：它会把“当前实现是否满足理想业务期待”分成
pass / weak / fail，方便发现设计短板，而不是让已知短板直接阻塞 pytest。
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.memory.v3.episodic_store import InMemoryEpisodicMemoryV3Store
from app.services.memory.v3.llm_judge import LlmMemoryJudge
from app.services.memory.v3.manager import MemoryManagerV3
from app.services.memory.v3.profile_store import InMemoryUserProfileV3Store
from app.services.memory.v3.read_planner import MemoryReadPlanner
from app.services.memory.v3.session_store import InMemorySessionMemoryV3Store
from app.services.memory.v3.types import LlmMemoryJudgement, ObservationCandidate, TurnEvidencePacket


RESULT_DIR = Path("eval/results")
JSON_PATH = RESULT_DIR / "memory_v3_scenario_report.json"
MD_PATH = RESULT_DIR / "memory_v3_scenario_report.md"


@dataclass
class ScenarioResult:
    scenario_id: str
    name: str
    status: str
    passed_checks: list[str] = field(default_factory=list)
    weak_points: list[str] = field(default_factory=list)
    failed_checks: list[str] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)


class StaticJudge:
    """测试用 LLM Judge：返回预设 judgement，避免真实模型影响评测稳定性。"""

    def __init__(self, judgement: LlmMemoryJudgement | None) -> None:
        self.judgement = judgement

    def judge(self, **_: Any) -> LlmMemoryJudgement | None:
        return self.judgement


def make_manager(*, judge: object | None = None) -> MemoryManagerV3:
    return MemoryManagerV3(
        session_store=InMemorySessionMemoryV3Store(),
        profile_store=InMemoryUserProfileV3Store(),
        episodic_store=InMemoryEpisodicMemoryV3Store(),
        llm_judge=judge,
        enabled=True,
    )


def packet(
    question: str,
    *,
    answer: str = "已记录。",
    session_id: str = "sid",
    user_id: str | None = None,
    turn_id: str = "turn",
    route_domain_hint: str = "customer_service",
    route_needs_rag: bool = False,
    branch_name: str = "no_rag",
    recent_history: str = "",
    visual_context: str = "",
    rag_context: str = "",
    branch_result: dict[str, Any] | None = None,
    tool_results: list[dict[str, Any]] | None = None,
) -> TurnEvidencePacket:
    return TurnEvidencePacket(
        session_id=session_id,
        user_id=user_id,
        turn_id=turn_id,
        timestamp=time.time(),
        question=question,
        answer=answer,
        route_domain_hint=route_domain_hint,
        route_needs_rag=route_needs_rag,
        branch_name=branch_name,
        recent_history=recent_history,
        visual_context=visual_context,
        rag_context=rag_context,
        branch_result=branch_result,
        tool_results=tool_results or [],
    )


def snapshot(manager: MemoryManagerV3, *, session_id: str = "sid", user_id: str | None = None) -> dict[str, Any]:
    session = manager.session_store.get(session_id)
    profile = manager.profile_store.get(user_id) if user_id else None
    episodic = manager.episodic_store.search(user_id, "", top_k=10) if user_id else []
    return {
        "active_issue_thread_id": session.active_issue_thread_id,
        "atoms": [
            {
                "kind": atom.kind,
                "value": atom.value,
                "scope": atom.scope,
                "source": atom.source,
                "status": atom.status,
                "pii_level": atom.pii_level,
            }
            for atom in session.atoms.values()
        ],
        "issue_threads": [
            {
                "thread_id": thread.thread_id,
                "status": thread.status,
                "category": thread.category,
                "product_model": thread.product_model,
                "fault_codes": list(thread.fault_codes),
                "symptoms": list(thread.symptoms),
                "attempted_actions": list(thread.attempted_actions),
                "missing_slots": list(thread.missing_slots),
            }
            for thread in session.issue_threads.values()
        ],
        "profile_atoms": [
            {
                "kind": atom.kind,
                "value": atom.value,
                "status": atom.status,
                "pii_level": atom.pii_level,
            }
            for atom in (profile.stable_atoms.values() if profile else [])
        ],
        "episodic_events": [
            {
                "event_type": event.event_type,
                "title": event.title,
                "summary": event.summary,
            }
            for event in episodic
        ],
    }


def values(state: dict[str, Any], kind: str, *, active_only: bool = True) -> list[str]:
    out = []
    for atom in state["atoms"]:
        if atom["kind"] == kind and (not active_only or atom["status"] == "active"):
            out.append(atom["value"])
    return out


def profile_values(state: dict[str, Any], kind: str) -> list[str]:
    return [atom["value"] for atom in state["profile_atoms"] if atom["kind"] == kind and atom["status"] == "active"]


def make_result(
    scenario_id: str,
    name: str,
    *,
    passed: list[str] | None = None,
    weak: list[str] | None = None,
    failed: list[str] | None = None,
    state: dict[str, Any] | None = None,
) -> ScenarioResult:
    passed = passed or []
    weak = weak or []
    failed = failed or []
    if failed:
        status = "fail"
    elif weak:
        status = "weak"
    else:
        status = "pass"
    return ScenarioResult(
        scenario_id=scenario_id,
        name=name,
        status=status,
        passed_checks=passed,
        weak_points=weak,
        failed_checks=failed,
        state=state or {},
    )


def scenario_non_tool_session_write() -> ScenarioResult:
    manager = make_manager()
    manager.observe_and_write(packet("我的 AC900 显示 E2，我已经断电重启过了，还是不行"))
    state = snapshot(manager)
    passed, failed = [], []
    if "AC900" in values(state, "product_model"):
        passed.append("非工单场景能从用户当前轮写入产品型号")
    else:
        failed.append("未写入产品型号 AC900")
    if "E2" in values(state, "fault_code"):
        passed.append("非工单场景能写入故障码")
    else:
        failed.append("未写入故障码 E2")
    if "断电重启" in values(state, "attempted_action"):
        passed.append("非工单场景能写入用户已尝试动作")
    else:
        failed.append("未写入断电重启动作")
    return make_result("S01", "非工单当前轮事实写入", passed=passed, failed=failed, state=state)


def scenario_pure_howto_memory_noise() -> ScenarioResult:
    manager = make_manager()
    manager.observe_and_write(
        packet(
            "AC900 的滤网怎么清洗？",
            answer="根据手册，先打开后盖再取出滤网。",
            branch_name="rag_manual",
            route_needs_rag=True,
            rag_context="[手册] 先打开后盖再取出滤网。",
        )
    )
    state = snapshot(manager)
    weak = []
    if values(state, "product_model"):
        weak.append("纯手册 how-to 问题也会写入 product_model 并创建 issue thread，可能造成 session 记忆噪音")
    if not any(atom["kind"] in {"manual_step", "manual_knowledge"} for atom in state["atoms"]):
        passed = ["未把 RAG 手册步骤原文写成用户记忆"]
    else:
        return make_result("S02", "纯手册问答禁写与噪音", failed=["RAG 手册知识泄漏进用户记忆"], state=state)
    return make_result("S02", "纯手册问答禁写与噪音", passed=passed, weak=weak, state=state)


def scenario_rag_feedback_write() -> ScenarioResult:
    manager = make_manager()
    manager.observe_and_write(
        packet(
            "我已经清洗滤网了，但红灯还闪",
            branch_name="rag_manual",
            route_needs_rag=True,
            rag_context="[手册] 清洗滤网步骤...",
        )
    )
    state = snapshot(manager)
    passed, failed = [], []
    if "清洗滤网" in values(state, "attempted_action"):
        passed.append("用户对 RAG 建议的反馈能写为 attempted_action")
    else:
        failed.append("未记录用户已清洗滤网")
    if "红灯还闪" in values(state, "symptom"):
        passed.append("用户反馈的新现象能写入 session")
    else:
        failed.append("未记录红灯还闪现象")
    return make_result("S03", "RAG 反馈写入", passed=passed, failed=failed, state=state)


def scenario_profile_phone_gate() -> ScenarioResult:
    manager = make_manager()
    user_id = "alice"
    manager.observe_and_write(packet("手机号 13800138000", user_id=user_id, turn_id="t1"))
    plain_state = snapshot(manager, user_id=user_id)
    manager.observe_and_write(packet("记住，以后默认用 13800138000 联系我", user_id=user_id, turn_id="t2"))
    remembered_state = snapshot(manager, user_id=user_id)
    passed, failed = [], []
    if not profile_values(plain_state, "phone"):
        passed.append("普通手机号不会直接进入 profile")
    else:
        failed.append("普通手机号误写入 profile")
    if "13800138000" in profile_values(remembered_state, "phone"):
        passed.append("显式记住手机号可写入 profile")
    else:
        failed.append("显式记住手机号未进入 profile")
    return make_result("S04", "手机号 profile 门控", passed=passed, failed=failed, state=remembered_state)


def scenario_forget_profile_gap() -> ScenarioResult:
    manager = make_manager()
    user_id = "alice"
    manager.observe_and_write(packet("记住，以后默认用 13800138000 联系我", user_id=user_id, turn_id="t1"))
    manager.observe_and_write(packet("不要保存我的手机号 13800138000", user_id=user_id, turn_id="t2"))
    state = snapshot(manager, user_id=user_id)
    weak = []
    if profile_values(state, "phone"):
        weak.append("forget/delete 当前只作用于 session，未删除 profile 中已保存的手机号")
    return make_result(
        "S05",
        "忘记指令跨 scope 删除",
        passed=[] if weak else ["profile 手机号已删除"],
        weak=weak,
        state=state,
    )


def scenario_correction() -> ScenarioResult:
    manager = make_manager()
    manager.observe_and_write(packet("我的 AC900 显示 E2", turn_id="t1"))
    manager.observe_and_write(packet("刚才说错了，不是 AC900，是 AC901", turn_id="t2"))
    state = snapshot(manager)
    active_models = values(state, "product_model")
    passed, failed = [], []
    if "AC901" in active_models and "AC900" not in active_models:
        passed.append("纠错后 active 产品型号替换为 AC901")
    else:
        failed.append(f"纠错后 active 产品型号异常: {active_models}")
    return make_result("S06", "用户纠错 supersede", passed=passed, failed=failed, state=state)


def scenario_multi_issue_mixing() -> ScenarioResult:
    manager = make_manager()
    manager.observe_and_write(packet("我的 AC900 显示 E2，我已经断电重启过了", turn_id="t1"))
    manager.observe_and_write(packet("另一个 DW200 显示 F1，我清洗滤网了也不行", turn_id="t2"))
    state = snapshot(manager)
    weak = []
    if len(state["issue_threads"]) < 2:
        weak.append("多产品/多故障仍被合并到单一 active issue thread，E2/F1 可能串线")
    return make_result(
        "S07",
        "多产品多问题隔离",
        passed=[] if weak else ["成功生成多个 issue thread"],
        weak=weak,
        state=state,
    )


def scenario_llm_bad_json_fallback() -> ScenarioResult:
    judge = LlmMemoryJudge(call_model=lambda _prompt: "不是 JSON")
    manager = make_manager(judge=judge)
    manager.observe_and_write(packet("我的 AC900 显示 E2，这个还是不行"))
    state = snapshot(manager)
    passed, failed = [], []
    if "AC900" in values(state, "product_model") and "E2" in values(state, "fault_code"):
        passed.append("LLM 输出坏 JSON 时，规则候选仍能写入 session")
    else:
        failed.append("LLM 坏 JSON 导致规则候选丢失")
    return make_result("S08", "LLM 坏 JSON 回退", passed=passed, failed=failed, state=state)


def scenario_llm_hallucinated_action_gap() -> ScenarioResult:
    judgement = LlmMemoryJudgement(
        should_write=True,
        write_intent="observe",
        target_scope="session",
        confidence=0.9,
        reason="模拟 LLM 幻觉",
        observations=[
            ObservationCandidate(
                kind="attempted_action",
                value="更换主板",
                source="llm_judge",
                confidence=0.9,
                evidence_text="LLM 幻觉",
                scope_hint="session",
            )
        ],
        resolved_references={},
    )
    manager = make_manager(judge=StaticJudge(judgement))
    manager.observe_and_write(packet("这个还是不行", recent_history="用户: AC900 显示 E2"))
    state = snapshot(manager)
    weak = []
    if "更换主板" in values(state, "attempted_action"):
        weak.append("LLM verifier 允许 attempted_action 不出现在原文中，存在幻觉动作写入风险")
    return make_result(
        "S09",
        "LLM 幻觉候选防护",
        passed=[] if weak else ["幻觉动作被拦截"],
        weak=weak,
        state=state,
    )


def scenario_tool_episodic_no_session_mirror() -> ScenarioResult:
    manager = make_manager()
    manager.observe_and_write(
        packet(
            "帮我提交报修",
            user_id="alice",
            branch_name="case_intake",
            branch_result={"status": "submitted", "case_id": "CASE-1001", "product_model": "AC900"},
        )
    )
    state = snapshot(manager, user_id="alice")
    weak = []
    if state["episodic_events"]:
        passed = ["工单 submitted 能写入 episodic"]
    else:
        return make_result("S10", "工具事件 episodic 与 session 镜像", failed=["未写入 episodic"], state=state)
    if "CASE-1001" not in values(state, "case_id"):
        weak.append("case_id/status 作为 episodic 写入后没有同步镜像到 session issue thread")
    return make_result("S10", "工具事件 episodic 与 session 镜像", passed=passed, weak=weak, state=state)


def scenario_read_noise() -> ScenarioResult:
    manager = make_manager()
    manager.observe_and_write(packet("我的 AC900 显示 E2", turn_id="t1"))
    manager.observe_and_write(packet("另一个 DW200 显示 F1", turn_id="t2"))
    planner = MemoryReadPlanner()
    request = planner.plan(
        session_id="sid",
        user_id=None,
        question="AC900 这个故障怎么继续处理？",
        recent_history="",
        route_domain_hint="customer_service",
    )
    rendered = manager.read(request).rendered_context
    weak = []
    if "DW200" in rendered or "F1" in rendered:
        weak.append("读取阶段未按 query 实体强过滤，AC900 问题可能带出 DW200/F1 噪音")
    return make_result(
        "S11",
        "读取相关性与噪音控制",
        passed=[] if weak else ["读取结果未包含无关产品"],
        weak=weak,
        state={"rendered_context": rendered},
    )


def scenario_preference_requires_llm() -> ScenarioResult:
    judgement = LlmMemoryJudgement(
        should_write=True,
        write_intent="remember",
        target_scope="profile",
        confidence=0.86,
        reason="用户表达长期回答偏好",
        observations=[
            ObservationCandidate(
                kind="user_preference",
                value="回答尽量简短",
                source="llm_judge",
                confidence=0.86,
                evidence_text="以后回答简单点",
                scope_hint="profile",
                write_intent="remember",
            )
        ],
        resolved_references={},
    )
    manager = make_manager(judge=StaticJudge(judgement))
    manager.observe_and_write(packet("以后回答简单点", user_id="alice"))
    state = snapshot(manager, user_id="alice")
    passed, failed = [], []
    if "回答尽量简短" in profile_values(state, "user_preference"):
        passed.append("LLM 辅助能把自然语言长期偏好写入 profile")
    else:
        failed.append("长期偏好未写入 profile")
    return make_result("S12", "LLM 辅助 profile 偏好", passed=passed, failed=failed, state=state)


SCENARIOS: list[Callable[[], ScenarioResult]] = [
    scenario_non_tool_session_write,
    scenario_pure_howto_memory_noise,
    scenario_rag_feedback_write,
    scenario_profile_phone_gate,
    scenario_forget_profile_gap,
    scenario_correction,
    scenario_multi_issue_mixing,
    scenario_llm_bad_json_fallback,
    scenario_llm_hallucinated_action_gap,
    scenario_tool_episodic_no_session_mirror,
    scenario_read_noise,
    scenario_preference_requires_llm,
]


def summarize(results: list[ScenarioResult]) -> dict[str, Any]:
    counts = {"pass": 0, "weak": 0, "fail": 0}
    for result in results:
        counts[result.status] += 1
    limitations = []
    for result in results:
        for item in result.weak_points + result.failed_checks:
            limitations.append({"scenario_id": result.scenario_id, "issue": item})
    return {
        "scenario_count": len(results),
        "counts": counts,
        "limitations": limitations,
    }


def write_reports(results: list[ScenarioResult]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    summary = summarize(results)
    JSON_PATH.write_text(
        json.dumps(
            {
                "summary": summary,
                "results": [asdict(result) for result in results],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    lines = [
        "# Memory v3 Scenario Report",
        "",
        f"- 场景数: {summary['scenario_count']}",
        f"- pass: {summary['counts']['pass']}",
        f"- weak: {summary['counts']['weak']}",
        f"- fail: {summary['counts']['fail']}",
        "",
        "## 设计不足汇总",
    ]
    if summary["limitations"]:
        for item in summary["limitations"]:
            lines.append(f"- `{item['scenario_id']}` {item['issue']}")
    else:
        lines.append("- 暂未发现 weak/fail 项。")
    lines.append("")
    lines.append("## 场景明细")
    for result in results:
        lines.extend(
            [
                "",
                f"### {result.scenario_id} {result.name}",
                f"- status: `{result.status}`",
            ]
        )
        for item in result.passed_checks:
            lines.append(f"- pass: {item}")
        for item in result.weak_points:
            lines.append(f"- weak: {item}")
        for item in result.failed_checks:
            lines.append(f"- fail: {item}")
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    results = [scenario() for scenario in SCENARIOS]
    write_reports(results)
    summary = summarize(results)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nWrote: {JSON_PATH}")
    print(f"Wrote: {MD_PATH}")


if __name__ == "__main__":
    main()
