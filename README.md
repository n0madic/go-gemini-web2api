# go-gemini-web2api

A zero-dependency **Gemini Web → OpenAI API** proxy written in Go.
It exposes Google Gemini's web interface as an OpenAI-compatible
(and Google-native-compatible) HTTP API, so tools like Cherry Studio, ChatBox,
the OpenAI SDK, Codex CLI, or the Gemini CLI can talk to it directly.

This project draws ideas from [`gemini_web2api.py`](https://github.com/Sophomoresty/gemini-web2api)
and [HanaokaYuzu/Gemini-API](https://github.com/HanaokaYuzu/Gemini-API) — the latter
being the source of the header-based model selection and the account-tier capacity logic.

## Features

- **Zero cost, zero install** — single static Go binary, stdlib only, no paid subscription
- **OpenAI compatible** — drop-in `/v1/chat/completions` and `/v1/models`
- **Anthropic compatible** — `/v1/messages` for Claude Code and Anthropic SDK clients
- **Responses API** — `/v1/responses` for OpenAI Codex CLI
- **Google native API** — `/v1beta/models` for Gemini CLI
- **Streaming** — true SSE token streaming
- **Tool calling** — OpenAI / Anthropic / Google function calling, with choice constraints (none/auto/required/specific)
- **Vision / image input** — OpenAI `image_url`, Anthropic `image` blocks, Google `inlineData`, Responses `input_image` (requires an authenticated cookie)
- **Web search** — Gemini's built-in internet access is inherited
- **Dynamic models** — the model list is read live from your account (depends on its tier); `/v1/models` reflects what you can actually use
- **Optional API keys** — open by default, Bearer/`x-api-key` auth when configured

## How it works

Requests are translated into Gemini's internal `StreamGenerate` (batchexecute)
endpoint — the same one the Gemini web app uses — converting between OpenAI's API
format and Gemini's internal protobuf-like array format. The backend does not
require authentication for basic anonymous text generation. The model is selected
per request via HTTP headers (`x-goog-ext-525001261-jspb` carries the model id and
account capacity), and the available-model list is fetched live from the account
via the `otAQ7b` batchexecute RPC. Cookie / SAPISIDHASH auth is optional for plain
text, but required to read the real model list, to route to non-default models, and
for image input (and it raises the prompt-size limit). Without authentication the
proxy serves a small static fallback list and generation falls back to the default
(Flash) model.

> ⚠️ This relies on reverse-engineered web endpoints. It can break when Google
> changes them. The date-stamped `GEMINI_BL` build label is **auto-resolved** from
> the Gemini page at startup (and refreshed periodically), so it stays current on
> its own; set `GEMINI_BL` explicitly only to pin a specific value.

## Build & run

```sh
go build -o go-gemini-web2api .
./go-gemini-web2api
# or just:
go run .
```

Configuration is read from the environment, optionally seeded from a `.env` file
(copy `.env.example` to `.env`). Process environment variables always take
precedence over `.env`.

```sh
cp .env.example .env   # optional
go run .                       # listens on 127.0.0.1:8081 by default

# Expose on all interfaces / a different port:
GEMINI_LISTEN=:8081 go run .
./go-gemini-web2api -listen 0.0.0.0:9090

# Point at a specific .env file and/or cookie file:
./go-gemini-web2api -config /etc/gemini/.env -cookie-file /etc/gemini/cookie.txt
```

By default the server binds to **localhost only** (`127.0.0.1:8081`); pass
`:8081`, `0.0.0.0:8081`, or a bare port to listen on all interfaces. The `-listen`
flag overrides `GEMINI_LISTEN`. The `.env` path is resolved in priority order: the
`-config` flag, then `$GEMINI_ENV_FILE`, then `./.env`. The `-cookie-file` flag overrides
`GEMINI_COOKIE_FILE`.

## Client configuration

### Cherry Studio / ChatBox / any OpenAI client

| Field | Value |
|---|---|
| Base URL | `http://localhost:8081/v1` |
| API Key | any value if `GEMINI_API_KEYS` is unset; otherwise one of your keys |
| Model | `gemini-3.5-flash` (or any returned by `/v1/models`) |

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8081/v1", api_key="sk-your-key")
resp = client.chat.completions.create(
    model="gemini-3.5-flash",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
)
print(resp.choices[0].message.content)
```

### Claude Code / Anthropic clients

The proxy speaks the Anthropic Messages API (`/v1/messages`), so Claude Code and
the Anthropic SDK work against it:

```sh
export ANTHROPIC_BASE_URL=http://localhost:8081
export ANTHROPIC_API_KEY=sk-anything          # or one of your GEMINI_API_KEYS
export ANTHROPIC_MODEL=gemini-3.5-flash        # optional; unknown names fall back to GEMINI_DEFAULT_MODEL
export ANTHROPIC_SMALL_FAST_MODEL=gemini-3.1-flash-lite   # optional, for background tasks
claude
```

`claude-*` model names sent by the client are accepted and routed to
`GEMINI_DEFAULT_MODEL` (set `ANTHROPIC_MODEL` to a Gemini model name to choose
explicitly). The `/v1/models` endpoint returns Anthropic-formatted output when the
request carries an `anthropic-version` header, and OpenAI-formatted output otherwise.

### Gemini CLI

```sh
export GEMINI_API_KEY=none
export GOOGLE_GEMINI_BASE_URL=http://localhost:8081
gemini
```

### curl

```sh
# Non-streaming
curl http://localhost:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemini-3.5-flash","messages":[{"role":"user","content":"Hello!"}]}'

# Streaming
curl -N http://localhost:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemini-3.5-flash","messages":[{"role":"user","content":"Hi"}],"stream":true}'

# With client auth enabled (GEMINI_API_KEYS=sk-secret)
curl http://localhost:8081/v1/chat/completions \
  -H 'Authorization: Bearer sk-secret' \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemini-3.5-flash","messages":[{"role":"user","content":"Hi"}]}'
```

## Configuration (environment variables)

| Variable | Default | Description |
|---|---|---|
| `GEMINI_LISTEN` | `127.0.0.1:8081` | Bind address; accepts `host:port`, `:port` (all interfaces), or a bare port. Overridden by `-listen` |
| `GEMINI_RETRY_ATTEMPTS` | `3` | Upstream retry attempts |
| `GEMINI_RETRY_DELAY_SEC` | `2` | Delay between retries |
| `GEMINI_REQUEST_TIMEOUT_SEC` | `180` | Per-request timeout |
| `GEMINI_BL` | `auto` | Gemini web build label; `auto`/empty resolves it from the Gemini page, or pin an explicit value |
| `GEMINI_DEFAULT_MODEL` | `gemini-3.5-flash` | Model used when none is given |
| `GEMINI_LOG_REQUESTS` | `false` | Log incoming requests at INFO (warnings/errors always shown) |
| `GEMINI_AUTH_USER` | – | Google account index (`/u/N`) |
| `GEMINI_COOKIE` | – | Inline cookie string |
| `GEMINI_COOKIE_FILE` | – | Path to a cookie file (raw `k=v; …` string; overridden by `-cookie-file`) |
| `GEMINI_COOKIE_REFRESH_MIN` | `9` | Minutes between `__Secure-1PSIDTS` auto-rotations for a file-backed cookie (0 = off) |
| `GEMINI_PROXY` | – | HTTP(S) proxy URL |
| `GEMINI_API_KEYS` | – | Comma-separated accepted client keys (empty = open) |
| `GEMINI_ENV_FILE` | `.env` | Path to the dotenv file (overridden by `-config`) |

When `GEMINI_API_KEYS` is empty, authentication is disabled. When one or more keys
are set, the generation endpoints require the key via `Authorization: Bearer <key>`,
`x-api-key`, `x-goog-api-key`, or a `?key=` query parameter. Health (`/`) and the
model lists stay open.

> ⚠️ Binding to a non-loopback address (`:8081`, `0.0.0.0`, or a LAN IP) **with no
> API keys** exposes your authenticated Google session to anyone who can reach it.
> Set `GEMINI_API_KEYS` whenever you don't bind to `127.0.0.1` — the proxy prints a
> loud `OPEN PROXY` warning at startup if you forget.

## Models

The model list is **not hardcoded** — it is read live from your Google account at
startup (and refreshed after each cookie rotation) via the `otAQ7b` batchexecute
RPC, so it reflects the models your account's tier actually exposes. Query the
current list at any time:

```sh
curl http://localhost:8081/v1/models
```

Model names come straight from Gemini's versioned label (e.g. `3.1 Flash-Lite` →
`gemini-3.1-flash-lite`). A typical free account today exposes:

| Model | Description |
|---|---|
| `gemini-3.5-flash` | Fast general-purpose model (default) |
| `gemini-3.1-pro` | Pro model (needs an authenticated cookie) |
| `gemini-3.1-flash-lite` | Lightweight fast model |

The exact names and count depend on your account, so treat the table above as an
example — trust `/v1/models`. The startup banner shows the resolved list and marks
its source as `(dynamic)` (from the account) or `(fallback)`.

**Fallback list.** When the proxy runs anonymously, or the cookie no longer
authenticates, or the RPC fails, it cannot read the real catalog and the banner
shows `(fallback)`. In that mode the backend only serves its default (Flash)
model — there are no real model ids to select anything else — so the list is
honestly just the single configured `GEMINI_DEFAULT_MODEL` (default
`gemini-3.5-flash`). Requesting any other name returns `400` rather than silently
answering with Flash.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Health / status |
| `GET` | `/v1/models` | List models (OpenAI or Anthropic format, by `anthropic-version` header) |
| `GET` | `/v1beta/models` | List models (Google format) |
| `POST` | `/v1/chat/completions` | OpenAI Chat Completions (stream + tools) |
| `POST` | `/v1/responses` | OpenAI Responses API (Codex CLI) |
| `POST` | `/v1/messages` | Anthropic Messages API (Claude Code; stream + tools) |
| `POST` | `/v1/messages/count_tokens` | Anthropic token counting |
| `POST` | `/v1beta/models/{model}:generateContent` | Google native (non-stream) |
| `POST` | `/v1beta/models/{model}:streamGenerateContent` | Google native (stream) |

## Tool calling

```python
resp = client.chat.completions.create(
    model="gemini-3.5-flash",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }],
)
```

Tool calls are emulated by injecting a `tool_call` instruction block into the
prompt and parsing the model's response back into OpenAI/Anthropic tool calls.
Reliability depends on the model following the instruction — heavily agentic
clients like Claude Code (which lean on tool use for every action) may behave
inconsistently. Plain chat works well.

`tool_choice` is honored: `none` (no tools), `auto` (default), `required` /
Anthropic `any` (must call a tool), and a specific function (`{"type":"function",
"function":{"name":"…"}}` / Anthropic `{"type":"tool","name":"…"}`) all add the
corresponding constraint to the prompt.

The Google native API supports the full function-calling cycle:
`functionDeclarations` (in `tools`), model `functionCall` parts in the response,
client `functionResponse` parts in `contents`, and `toolConfig.functionCallingConfig`
(`AUTO` / `ANY` / `NONE`, with `allowedFunctionNames`). Streaming
(`streamGenerateContent`) emits incremental candidate chunks followed by a final
chunk carrying `finishReason` and `usageMetadata`.

## Cookie for Pro routing

Anonymous access works for plain text, but it cannot read the real account model
list — it falls back to just the default Flash model, and any other model name
returns `400`. [Image input](#vision--image-input) is also unavailable. For the
full model list, non-default models, and vision, provide a Google cookie.

Authentication uses Google's `SAPISIDHASH` scheme. For the session to actually
authenticate you need **all** of these (they belong together):

- **`__Secure-1PSID`** + **`__Secure-1PSIDTS`** — the session cookie and its
  rotating timestamp token. **`__Secure-1PSIDTS` is required** — `__Secure-1PSID`
  alone is treated as a stale/anonymous session. (`__Secure-1PSIDCC` may also help.)
- **`SAPISID`** — used to compute the `SAPISIDHASH` Authorization header.

The legacy `SID`, `HSID`, `SSID`, and `APISID` cookies are **not required**.

> ⚠️ If you omit `__Secure-1PSIDTS`, requests silently fall back to anonymous
> (Google ignores invalid auth rather than erroring), so everything still
> "works" — but you are not actually authenticated. See *Verifying auth* below.

**How to get them:**

1. Open Chrome, go to [gemini.google.com](https://gemini.google.com), and sign in
   with any free Google account.
2. Open DevTools (F12) → Application → Cookies → `https://gemini.google.com`.
3. Copy `__Secure-1PSID`, `__Secure-1PSIDTS`, and `SAPISID`.

Then either pass them inline:

```sh
export GEMINI_COOKIE="__Secure-1PSID=...; __Secure-1PSIDTS=...; SAPISID=..."
```

…or point at a file via `GEMINI_COOKIE_FILE=cookie.txt` (or the `-cookie-file
cookie.txt` flag). The file holds the raw single-line cookie string — the same
`k=v; k=v` format as `GEMINI_COOKIE`:

```
__Secure-1PSID=...; __Secure-1PSIDTS=...; SAPISID=...
```

`SAPISID` is parsed automatically from the cookie string, so make sure it is one
of the `k=v` pairs (it normally is, since it's sent on every Google request). A
free Google account is sufficient — no paid subscription required.

### Keeping the cookie fresh (auto-rotation)

`__Secure-1PSIDTS` is a short-lived token that Google rotates frequently, so a
copied cookie goes stale within minutes. The proxy keeps it alive automatically by
rotating `__Secure-1PSIDTS` against `accounts.google.com/RotateCookies` at startup
and every `GEMINI_COOKIE_REFRESH_MIN` minutes (default `9`):

- **File-backed** (`GEMINI_COOKIE_FILE` / `-cookie-file`) — the rotated token is
  written back to the file in place, so it survives restarts.
- **Inline** (`GEMINI_COOKIE`) — the token is rotated **in memory**: the session
  stays valid for as long as the proxy runs, but the rotated value is not persisted.
  After a restart, the `GEMINI_COOKIE` value must still be fresh (prefer a cookie
  file if you restart often).

The banner shows `authenticated (auto-refresh 9m)` when rotation is active; set
`GEMINI_COOKIE_REFRESH_MIN=0` to disable it. Tip: don't keep gemini.google.com open
in a browser with the same account, or the two will fight over rotating the same
session token.

### Verifying auth actually works

Because invalid auth silently falls back to anonymous, "it returns a response" is
**not** proof that the cookie authenticated. To check:

- **Startup banner** — when a cookie is configured, the proxy checks it at startup
  (via the `SNlM0e` token, see below) and prints the result:

  ```
  Cookie:    yes (cookie.txt) — authenticated
  Cookie:    yes (cookie.txt) — NOT authenticated
  ```

  A `NOT authenticated` line also logs a warning hinting at a missing/expired
  `__Secure-1PSIDTS`.
- **Signed-in token** — fetch the page with your cookie and look for the `SNlM0e`
  token, which is issued only to a signed-in session:

  ```sh
  curl -s -H "Cookie: <your cookie string>" https://gemini.google.com/app | grep -o '"SNlM0e":"[^"]*"' | head -1
  ```

  A match means the cookie authenticates; no match means it's being treated as
  anonymous (usually a missing/expired `__Secure-1PSIDTS`).
- **History** — send a request through the proxy, then open
  [gemini.google.com](https://gemini.google.com) → *Recent*. The conversation
  appears there only if the request was authenticated.

### Authenticated account path and XSRF token

If the signed-in Gemini URL contains an account index, e.g.
`https://gemini.google.com/u/1/app/...`, set `GEMINI_AUTH_USER=1`.

Authenticated requests **require** the page XSRF token (`SNlM0e` in the rendered
page source), sent as the `at` form field. Without it a signed-in request is
rejected and Gemini returns an empty body — even though the cookie itself is valid.
The server scrapes `SNlM0e` from `/app` at startup and re-scrapes it after each
cookie rotation (the token rotates with the session), so this is fully automatic —
there's nothing to configure. If authenticated requests still fail, refresh Gemini
Web and make sure `GEMINI_AUTH_USER` matches the `/u/<index>/` part of the URL. You
may also need to update `GEMINI_BL` to the current build label.

## Proxy

If you cannot reach `gemini.google.com` directly, set a proxy:

```sh
GEMINI_PROXY=http://127.0.0.1:7890 ./go-gemini-web2api
```

If `GEMINI_PROXY` is unset, the standard `HTTP_PROXY` / `HTTPS_PROXY` environment
variables are auto-detected. Works with Clash, V2Ray, Shadowsocks, or any HTTP proxy.

## Vision / image input

Image input works across every API surface and is attached to the prompt by
uploading each image to Gemini's push endpoint, then referencing it in the
generation payload:

| API | Accepted image form |
|---|---|
| OpenAI Chat (`/v1/chat/completions`) | content part `{"type":"image_url","image_url":{"url":"data:…"｜"https://…"}}` |
| OpenAI Responses (`/v1/responses`) | content part `{"type":"input_image","image_url":"data:…"｜"https://…"}` |
| Anthropic (`/v1/messages`) | block `{"type":"image","source":{"type":"base64"｜"url",…}}` |
| Google native (`/v1beta`) | part `{"inlineData":{"mimeType":"image/…","data":"<base64>"}}` |

Both base64/`data:` URIs (decoded inline) and remote `http(s)` URLs (fetched at
request time) are supported. Notes:

- **Requires an authenticated cookie.** Anonymous sessions (and cookies that no
  longer sign in) cannot attach images — such a request is rejected immediately
  with `400` and a clear message, rather than wasting retries. Text-only requests
  are unaffected. See [Cookie for Pro routing](#cookie-for-pro-routing).
- Uploaded images use a ~24h server-side TTL, so each image is uploaded fresh on
  every request (never cached across requests).
- An image that fails to download or upload is logged and skipped; the request
  still proceeds with the remaining images and text.

## Use as a library

The Gemini web client is importable on its own — `package gemini` has no HTTP/CLI
dependencies, so you can drive generation directly without running the proxy:

```go
import (
	"context"
	"fmt"
	"log/slog"
	"os"

	"github.com/n0madic/go-gemini-web2api/pkg/gemini"
)

func main() {
	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	// Every field is optional: New fills unset fields with sensible defaults
	// (RequestTimeout 180s, RetryAttempts 3, RetryDelaySec 2, DefaultModel
	// gemini-3.5-flash). When no explicit GeminiBL is given the build label is
	// auto-resolved from the Gemini page (GeminiBLAuto defaults on), and a nil
	// logger falls back to slog.Default(). gemini.New(gemini.Config{}, nil) just
	// works; pass a cookie to authenticate.
	client, err := gemini.New(gemini.Config{
		CookieFile: "cookie.txt", // optional; anonymous works for plain text
	}, logger)
	if err != nil {
		panic(err)
	}

	ctx := context.Background()
	client.ResolveBuildLabel(ctx) // optional: refresh the build label up front
	client.CheckAuth(ctx)         // optional: scrape the XSRF token for auth/models
	client.ResolveModels(ctx)     // optional: fetch the live model catalog

	m, _ := client.ResolveModelOrDefault("gemini-3.5-flash")

	// Blocking generation:
	text, err := client.Generate(ctx, m.Params("Say hello in one word."))
	fmt.Println(text, err)

	// Streaming generation (cumulative deltas):
	_ = client.GenerateStream(ctx, m.Params("Count to three."), func(delta string) {
		fmt.Print(delta)
	})
}
```

Call `client.SetModels([]*gemini.AvailableModel{...})` to seed the catalog offline
instead of fetching it. The format-translation layer (OpenAI / Anthropic / Google
↔ Gemini) lives in a second importable package,
`github.com/n0madic/go-gemini-web2api/pkg/apiconv`, if you want the request DTOs
and response builders without the bundled HTTP server.

## Limitations

- **Account-tier bound** — the available models are whatever your Google account's
  tier exposes via the live model list; the proxy can't unlock models your account
  doesn't have. Without an authenticated cookie only the default Flash model is
  available, and other model names return `400`.
- **Single-turn** — each request is independent; multi-turn context is simulated
  by folding previous messages into the prompt.
- **Prompt size limit** — the anonymous Gemini Web endpoint stops returning text
  somewhere around ~30–40k tokens of input and replies empty. The proxy surfaces
  this as a `502` (`empty response from Gemini …`) instead of a silent empty answer.
  This is why heavily-loaded agentic clients (e.g. Claude Code with many tools/MCP
  servers, whose system prompt alone can exceed the limit) may fail to get a reply.
- **Rate limits** — Google may throttle high-frequency requests. The server retries
  automatically, but sustained heavy use may be blocked.
- **Approximate token counts** — usage is estimated as `chars / 4` (a rough heuristic; the web backend reports no real token usage).

## Testing

```sh
go test ./...
```

## License

MIT
