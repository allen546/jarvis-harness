from jarvis.models.base import Message


def test_message_metadata_preserves_transport_fields() -> None:
    message = Message(role="user", content="hi", metadata={"channel_id": "c1", "message_id": "m1"})
    assert message.model_dump()["metadata"] == {"channel_id": "c1", "message_id": "m1"}
