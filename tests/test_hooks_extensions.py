import pytest
from jarvis.models.base import ToolCall
from jarvis.tools import ToolResult
from jarvis.hooks import BudgetGuardHook, ToolApprovalHook, HookResult

class MockContext:
    def __init__(self, max_consec=2, require_approval=True):
        from types import SimpleNamespace
        self.config = SimpleNamespace(
            max_consecutive_tools=max_consec,
            require_tool_approval=require_approval
        )
        self.approval_handler = None

@pytest.mark.asyncio
async def test_budget_guard_hook() -> None:
    hook = BudgetGuardHook()
    ctx = MockContext(max_consec=2)
    call = ToolCall(call_id="c1", tool_name="ls", arguments={})
    
    # Reset counter for a mock session
    from types import SimpleNamespace
    session = SimpleNamespace(id="s1")
    ctx.session = session
    await hook.before_model(ctx, [])
    
    # First call should succeed
    r1 = await hook.before_tool(ctx, call)
    assert r1.stop is False
    await hook.after_tool(ctx, call, ToolResult("c1", "ls", "ok"))
    
    # Second call should succeed
    r2 = await hook.before_tool(ctx, call)
    assert r2.stop is False
    await hook.after_tool(ctx, call, ToolResult("c2", "ls", "ok"))
    
    # Third call should hit budget limit
    r3 = await hook.before_tool(ctx, call)
    assert r3.stop is True
    assert "limit" in r3.reason.lower()

@pytest.mark.asyncio
async def test_tool_approval_hook() -> None:
    hook = ToolApprovalHook()
    ctx = MockContext(require_approval=True)
    call = ToolCall(call_id="c1", tool_name="Bash", arguments={"command": "rm -rf"})
    
    # Rejects when handler returns False
    ctx.approval_handler = lambda tc: False
    res_reject = await hook.before_tool(ctx, call)
    assert res_reject.skip_tool is True
    
    # Accepts when handler returns True
    ctx.approval_handler = lambda tc: True
    res_accept = await hook.before_tool(ctx, call)
    assert res_accept.skip_tool is False


@pytest.mark.asyncio
async def test_tool_approval_hook_async() -> None:
    hook = ToolApprovalHook()
    ctx = MockContext(require_approval=True)
    call = ToolCall(call_id="c1", tool_name="Bash", arguments={"command": "rm -rf"})
    
    # Async approval handler
    async def async_handler(tc):
        return True
        
    ctx.approval_handler = async_handler
    res = await hook.before_tool(ctx, call)
    assert res.skip_tool is False
