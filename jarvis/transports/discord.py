from __future__ import annotations

import typing
from jarvis.models.base import NativeAction

class MockDiscordAPI:
    def __init__(self) -> None:
        self.sent_messages: list[tuple[str, str, str | None]] = []
        self.reactions: list[tuple[str, str, str, str]] = []
        self.embeds: list[tuple[str, dict[str, typing.Any]]] = []
        self.threads: list[tuple[str, str, str | None]] = []

    async def send_message(self, channel_id: str, content: str, reply_to: str | None = None) -> None:
        self.sent_messages.append((channel_id, content, reply_to))

    async def add_reaction(self, channel_id: str, message_id: str, emoji: str, action: str = "add") -> None:
        self.reactions.append((channel_id, message_id, emoji, action))

    async def send_embed(self, channel_id: str, embed_dict: dict[str, typing.Any]) -> None:
        self.embeds.append((channel_id, embed_dict))

    async def create_thread(self, channel_id: str, name: str, message_id: str | None = None) -> None:
        self.threads.append((channel_id, name, message_id))


class DiscordTransport:
    def __init__(self, client: typing.Any = None) -> None:
        self.client = client or MockDiscordAPI()

    async def execute_native_action(self, channel_id: str, action: NativeAction) -> None:
        if isinstance(self.client, MockDiscordAPI) or hasattr(self.client, "sent_messages"):
            # Mock implementation
            if action.action_type == "discord_reply":
                msg_id = action.params["message_id"]
                content = action.params["content"]
                await self.client.send_message(channel_id, content, reply_to=msg_id)
            elif action.action_type == "discord_reaction":
                msg_id = action.params["message_id"]
                emoji = action.params["emoji"]
                action_name = action.params.get("action", "add")
                await self.client.add_reaction(channel_id, msg_id, emoji, action_name)
            elif action.action_type == "discord_send_embed":
                await self.client.send_embed(channel_id, action.params)
            elif action.action_type == "discord_create_thread":
                name = action.params["name"]
                msg_id = action.params.get("message_id")
                await self.client.create_thread(channel_id, name, msg_id)
        else:
            # Real implementation (lazy import discord)
            import discord
            # Get channel (handles both integer IDs for discord.py and string/channel retrieval)
            try:
                channel_int = int(channel_id)
                channel = self.client.get_channel(channel_int)
                if channel is None:
                    channel = await self.client.fetch_channel(channel_int)
            except ValueError:
                channel = self.client.get_channel(channel_id)
                if channel is None:
                    channel = await self.client.fetch_channel(channel_id)

            if action.action_type == "discord_reply":
                msg_id = action.params["message_id"]
                content = action.params["content"]
                try:
                    ref = discord.MessageReference(message_id=int(msg_id), channel_id=channel.id)
                except ValueError:
                    ref = discord.MessageReference(message_id=msg_id, channel_id=channel.id)
                await channel.send(content, reference=ref)
            elif action.action_type == "discord_reaction":
                msg_id = action.params["message_id"]
                emoji = action.params["emoji"]
                action_name = action.params.get("action", "add")
                try:
                    message = await channel.fetch_message(int(msg_id))
                except ValueError:
                    message = await channel.fetch_message(msg_id)
                if action_name == "add":
                    await message.add_reaction(emoji)
                elif action_name == "remove":
                    await message.remove_reaction(emoji, self.client.user)
            elif action.action_type == "discord_send_embed":
                embed = discord.Embed.from_dict(action.params)
                await channel.send(embed=embed)
            elif action.action_type == "discord_create_thread":
                name = action.params["name"]
                msg_id = action.params.get("message_id")
                if msg_id:
                    try:
                        message = await channel.fetch_message(int(msg_id))
                    except ValueError:
                        message = await channel.fetch_message(msg_id)
                    await message.create_thread(name=name)
                else:
                    await channel.create_thread(name=name)
