# Deploying Jarvis to Raspberry Pi

## Prerequisites

- Remote Pi at `192.168.0.159` (user: `allen`)
- SSH key-based auth configured
- HTTP proxy running on Pi at `127.0.0.1:7890` (HTTP) / `7891` (SOCKS)

## 1. First-time setup on Pi

```bash
ssh allen@192.168.0.159

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create directory
mkdir -p ~/jarvis/config/sessions ~/jarvis/storage/heartbeat ~/jarvis/storage/sessions
```

## 2. Deploy code

From your Mac (project root):

```bash
rsync -avz --progress \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude 'storage' \
  --exclude '*.pyc' \
  --exclude '.pytest_cache' \
  --exclude 'node_modules' \
  --exclude '.DS_Store' \
  --exclude 'config/' \
  ./ allen@192.168.0.159:/mnt/pi-data/jarvis/
```

> **Important:** `config/` is excluded because the remote has its own configs (QQ keys, MCP secrets, proxy settings). Never rsync local config over remote config.

## 3. Install dependencies on Pi

```bash
ssh allen@192.168.0.159 'cd ~/jarvis && ~/.local/bin/uv sync'
```

Verify:
```bash
ssh allen@192.168.0.159 'cd ~/jarvis && .venv/bin/python -c "import fastapi, mcp, croniter, pydantic, uvicorn, botpy, jarvis; print(\"ok\")"'
```

## 4. Remote config files

These live on the Pi and are **not** synced from local. Edit them on the remote directly.

### `config/global.json`

```json
{
  "model": {
    "provider": "openai_compatible",
    "model_name": "mimo-v2.5",
    "extra_params": {
      "api_key": "<your-api-key>",
      "base_url": "https://token-plan-cn.xiaomimimo.com/v1",
      "thinking": "medium"
    }
  },
  "harness": {
    "system_prompt": "You are Jarvis, a helpful AI assistant.",
    "heartbeat": {
      "enabled": true,
      "interval_secs": 300
    }
  },
  "channels": {
    "qq": {
      "enabled": true,
      "app_id": "<qq-app-id>",
      "app_secret": "<qq-app-secret>"
    }
  },
  "gateway": {
    "host": "0.0.0.0",
    "port": 8000
  },
  "proxy": {
    "http_proxy": "http://127.0.0.1:7890",
    "https_proxy": "http://127.0.0.1:7890",
    "no_proxy": "localhost,127.0.0.1,::1"
  }
}
```

> **Note:** `all_proxy` / SOCKS is deliberately omitted — `socksio` is not installed.

### `config/mcp_settings.json`

```json
{
  "mcpServers": {
    "amap-maps": {
      "url": "https://mcp.amap.com/mcp?key=<your-amap-key>",
      "transport": "streamable_http"
    },
    "calendar": {
      "command": "/mnt/pi-data/tools/.venv/bin/cal-mcp",
      "env": {
        "CALDAV_URL": "https://raspberrypi.tail350de6.ts.net:5232",
        "CALDAV_USER": "allen",
        "CALDAV_PASSWORD": "<cal-password>",
        "CALDAV_CALENDAR": "allen/calendar.ics",
        "CALDAV_REMINDERS": "allen/reminders",
        "CALDAV_VERIFY_SSL": "false"
      }
    },
    "managebac": {
      "command": "/mnt/pi-data/tools/.venv/bin/mb-mcp"
    },
    "playwright": {
      "command": "node",
      "args": ["/mnt/pi-data/tools/invisible-pw-mcp/server.js"]
    }
  }
}
```

> **Note:** Stdio MCP servers receive `HTTP_PROXY`/`HTTPS_PROXY` automatically from the proxy config. SOCKS `ALL_PROXY` is filtered out (no `socksio`).

## 5. Systemd service

The service file is at `jarvis.service` in the project root. Install it:

```bash
ssh allen@192.168.0.159 '
  sudo ln -sf /mnt/pi-data/jarvis/jarvis.service /etc/systemd/system/jarvis.service
  sudo systemctl daemon-reload
  sudo systemctl enable jarvis
  sudo systemctl start jarvis
'
```

### Service management

```bash
# Status
ssh allen@192.168.0.159 'systemctl status jarvis'

# Restart after code update
ssh allen@192.168.0.159 'sudo systemctl restart jarvis'

# View logs (live)
ssh allen@192.168.0.159 'journalctl -u jarvis -f'

# View recent logs
ssh allen@192.168.0.159 'journalctl -u jarvis --since "10 min ago"'
```

## 6. Full redeploy checklist

```bash
# From Mac project root:
rsync -avz --quiet \
  --exclude '.git' --exclude '.venv' --exclude '__pycache__' --exclude 'storage' \
  --exclude '*.pyc' --exclude '.pytest_cache' --exclude 'node_modules' --exclude '.DS_Store' \
  --exclude 'config/' \
  ./ allen@192.168.0.159:/mnt/pi-data/jarvis/ && \
ssh allen@192.168.0.159 'sudo systemctl restart jarvis'
```

## 7. Test

```bash
# Gateway health check
curl -X POST http://192.168.0.159:8000/sessions/test/turns \
  -H "Content-Type: application/json" \
  -d '{"content": "hello"}'

# Should return: {"session_id":"test","content":"Hello!","tool_calls":[]}
```

## Architecture notes

- **QQ bot** runs in a dedicated thread with its own event loop (botpy requirement)
- Messages are bridged to the main asyncio loop via `run_coroutine_threadsafe`
- All three QQ message types are handled: C2C DMs (`on_c2c_message_create`), group @mentions (`on_group_at_message_create`), guild @mentions (`on_at_message_create`)
- Replies are sent as markdown (`msg_type=2`)
- MCP proxy injection: only `HTTP_PROXY`/`HTTPS_PROXY` are passed to stdio processes; SOCKS vars are filtered out
- Gateway returns HTTP 500 on agent errors (via `ErrorEvent`)
