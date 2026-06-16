from __future__ import annotations

import typing
from jarvis.models.base import NativeAction

class MockQQAPI:
    def __init__(self) -> None:
        self.replies: list[tuple[str, str, str]] = []
        self.markdowns: list[tuple[str, str, str | None]] = []
        self.keyboards: list[tuple[str, dict[str, typing.Any]]] = []

    async def reply(self, target_id: str, content: str, message_id: str) -> None:
        self.replies.append((target_id, content, message_id))

    async def send_markdown(self, target_id: str, content: str, template_id: str | None = None) -> None:
        self.markdowns.append((target_id, content, template_id))

    async def send_keyboard(self, target_id: str, keyboard_data: dict[str, typing.Any]) -> None:
        self.keyboards.append((target_id, keyboard_data))


class QQTransport:
    def __init__(self, client: typing.Any = None) -> None:
        self.client = client or MockQQAPI()

    async def execute_native_action(self, target_id: str, action: NativeAction) -> None:
        if isinstance(self.client, MockQQAPI) or hasattr(self.client, "replies"):
            # Mock implementation
            if action.action_type == "qq_reply":
                msg_id = action.params["message_id"]
                content = action.params["content"]
                await self.client.reply(target_id, content, msg_id)
            elif action.action_type == "qq_send_markdown":
                content = action.params["content"]
                template_id = action.params.get("template_id")
                await self.client.send_markdown(target_id, content, template_id)
            elif action.action_type == "qq_send_keyboard":
                await self.client.send_keyboard(target_id, action.params)
        else:
            # Real implementation using botpy
            # botpy uses post_message or direct message
            import botpy
            
            if action.action_type == "qq_reply":
                msg_id = action.params["message_id"]
                content = action.params["content"]
                await self.client.api.post_message(
                    channel_id=target_id,
                    content=content,
                    msg_id=msg_id
                )
            elif action.action_type == "qq_send_markdown":
                content = action.params["content"]
                template_id = action.params.get("template_id")
                markdown_params = {"content": content}
                if template_id:
                    markdown_params["template_id"] = template_id
                await self.client.api.post_message(
                    channel_id=target_id,
                    markdown=markdown_params
                )
            elif action.action_type == "qq_send_keyboard":
                await self.client.api.post_message(
                    channel_id=target_id,
                    keyboard=action.params
                )
