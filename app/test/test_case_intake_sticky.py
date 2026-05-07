"""Case intake 粘性路由与 has_pending 行为。"""

from __future__ import annotations

from app.services.skills.case_intake_skill import CaseIntakeSkill
def test_has_pending_after_partial_intake():
    from app.services.skills.case_intake_redis_store import MemoryCaseIntakeStore

    store = MemoryCaseIntakeStore()
    skill = CaseIntakeSkill(state_store=store)
    sid = "sess_sticky_1"
    r = skill.run(
        question="电钻坏了帮我报修",
        session_id=sid,
        conversation_history="",
        enrichment="",
    )
    assert r.completed is False
    assert skill.has_pending_intake(sid) is True

    r2 = skill.run(
        question="型号 DW100",
        session_id=sid,
        conversation_history="",
        enrichment="",
    )
    assert r2.completed is False
    assert skill.has_pending_intake(sid) is True

    r3 = skill.run(
        question="电话13800138000",
        session_id=sid,
        conversation_history="",
        enrichment="",
    )
    assert r3.completed is True
    assert skill.has_pending_intake(sid) is False


def test_try_cancel_clears_pending():
    from app.services.skills.case_intake_redis_store import MemoryCaseIntakeStore

    store = MemoryCaseIntakeStore()
    skill = CaseIntakeSkill(state_store=store)
    sid = "sess_sticky_2"
    skill.run(question="机器故障报修", session_id=sid, conversation_history="", enrichment="")
    assert skill.has_pending_intake(sid) is True
    assert skill.try_cancel_intake(sid, "算了，先不报修了") is True
    assert skill.has_pending_intake(sid) is False


if __name__ == "__main__":
    test_has_pending_after_partial_intake()
    test_try_cancel_clears_pending()
    print("[OK] test_case_intake_sticky passed")
