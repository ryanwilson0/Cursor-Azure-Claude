/**
 * server.js - Azure Anthropic (Claude) -> OpenAI-compatible proxy for Cursor
 *
 * Key behavior:
 * - Always call Azure Anthropic endpoint NON-streaming
 * - If client requested stream=true, we stream back via OpenAI SSE format
 * - Tool calling:
 *   * We strongly instruct the model to emit Cursor XML tool-call markup (<function_calls>...</function_calls>)
 *   * We parse that into OpenAI tool_calls so Cursor actually executes tools.
 *   * If Azure ever returns native Anthropic tool_use blocks, we convert them too.
 *   * Fallback: parse "Calling read_file on ... with lines ..." lines.
 */

const express = require("express");
const axios = require("axios");

const app = express();
app.use(express.json({ limit: "50mb" }));

/**
 * ENV VARS REQUIRED
 * - AZURE_ENDPOINT   (full Anthropic messages endpoint)
 * - AZURE_API_KEY
 * - SERVICE_API_KEY  (what you paste into Cursor "OpenAI API Key")
 *
 * Optional:
 * - AZURE_DEPLOYMENT_NAME (defaults to "claude-opus-4-5")
 * - ANTHROPIC_VERSION (defaults to "2023-06-01")
 * - PORT (defaults to 8080)
 * - DEBUG_LOG_BODY ("true" to log request/response bodies; be careful)
 */
const CONFIG = {
  AZURE_ENDPOINT: process.env.AZURE_ENDPOINT,
  AZURE_API_KEY: process.env.AZURE_API_KEY,
  SERVICE_API_KEY: process.env.SERVICE_API_KEY,
  PORT: process.env.PORT || 8080,
  ANTHROPIC_VERSION: process.env.ANTHROPIC_VERSION || "2023-06-01",
  AZURE_DEPLOYMENT_NAME: process.env.AZURE_DEPLOYMENT_NAME || "claude-opus-4-5",
  DEBUG_LOG_BODY: String(process.env.DEBUG_LOG_BODY || "false").toLowerCase() === "true",
};

const MODEL_NAMES_TO_MAP = [
  "gpt-4",
  "gpt-4.1",
  "gpt-4o",
  "claude-opus-4-5",
  "claude-sonnet-4-5",
  "claude-4.5-opus-high",
  "claude-4-opus",
  "claude-3-opus",
  "claude-3-sonnet",
  "claude-3-haiku",
];

function mapModelToDeployment(modelName) {
  if (!modelName) return CONFIG.AZURE_DEPLOYMENT_NAME;
  if (MODEL_NAMES_TO_MAP.includes(modelName)) return CONFIG.AZURE_DEPLOYMENT_NAME;
  if (process.env.AZURE_DEPLOYMENT_NAME) return CONFIG.AZURE_DEPLOYMENT_NAME;
  return modelName;
}

function makeReqId() {
  return "req_" + Date.now().toString(36) + "_" + Math.random().toString(36).slice(2, 8);
}
function makeChatCmplId() {
  return "chatcmpl-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 10);
}

// -------------------- CORS --------------------
app.use((req, res, next) => {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.header("Access-Control-Allow-Headers", "Content-Type, Authorization, x-api-key, anthropic-version");
  if (req.method === "OPTIONS") return res.sendStatus(200);
  next();
});

// -------------------- Logging --------------------
app.use((req, res, next) => {
  console.log(`[${req.method}] ${req.path}`);
  next();
});

// -------------------- Auth --------------------
function requireAuth(req, res, next) {
  if (req.method === "OPTIONS" || req.path === "/health" || req.path === "/") return next();

  if (!CONFIG.SERVICE_API_KEY) {
    return res.status(500).json({ error: { message: "SERVICE_API_KEY not configured", type: "configuration_error" } });
  }

  const authHeader = req.headers.authorization;
  if (!authHeader) {
    return res.status(401).json({
      error: {
        type: "authentication_error",
        message:
          "Missing Authorization header.\n\n" +
          "Cursor Settings > Models > API Keys > OpenAI API Key\n" +
          "must equal SERVICE_API_KEY on the proxy.",
      },
    });
  }

  const token = authHeader.startsWith("Bearer ") ? authHeader.slice(7) : authHeader;
  if (token !== CONFIG.SERVICE_API_KEY) {
    return res.status(401).json({
      error: {
        type: "authentication_error",
        message:
          "Invalid API key.\n\n" +
          "Cursor Settings > Models > API Keys > OpenAI API Key\n" +
          "must equal SERVICE_API_KEY on the proxy.",
      },
    });
  }

  next();
}

// -------------------- Helpers --------------------
function toAnthropicContentBlocks(content) {
  if (Array.isArray(content)) return content;
  if (typeof content === "string") return [{ type: "text", text: content }];
  if (content == null) return [];
  return [{ type: "text", text: String(content) }];
}

function openaiToolsToAnthropic(tools = []) {
  // OpenAI tools: [{type:"function", function:{name, description, parameters}}]
  // Anthropic tools: [{name, description, input_schema}]
  return (tools || [])
    .filter((t) => t?.type === "function" && t.function?.name)
    .map((t) => ({
      name: t.function.name,
      description: t.function.description || "",
      input_schema: t.function.parameters || { type: "object", properties: {} },
    }));
}

function buildToolIdToNameMap(messages) {
  const map = new Map();
  if (!Array.isArray(messages)) return map;

  for (const m of messages) {
    if (!m || m.role !== "assistant") continue;
    if (Array.isArray(m.tool_calls)) {
      for (const tc of m.tool_calls) {
        const id = tc?.id;
        const name = tc?.function?.name;
        if (id && name) map.set(id, name);
      }
    }
  }
  return map;
}

function getToolNameSet(reqBody) {
  const set = new Set();
  const tools = Array.isArray(reqBody?.tools) ? reqBody.tools : [];
  for (const t of tools) {
    if (t?.function?.name) set.add(t.function.name);
    else if (t?.name) set.add(t.name);
  }
  return set;
}

function getToolParamSchemaMap(reqBody) {
  // name -> array of param keys (best-effort)
  const map = new Map();
  const tools = Array.isArray(reqBody?.tools) ? reqBody.tools : [];
  for (const t of tools) {
    const name = t?.function?.name || t?.name;
    const props = t?.function?.parameters?.properties;
    if (!name) continue;
    if (props && typeof props === "object") map.set(name, Object.keys(props));
    else map.set(name, []);
  }
  return map;
}

function pickFirstKey(keys, candidates) {
  for (const c of candidates) if (keys.includes(c)) return c;
  return null;
}

function safeParseValue(v) {
  if (typeof v !== "string") return v;
  const s = v.trim();
  if (!s) return s;

  if ((s.startsWith("{") && s.endsWith("}")) || (s.startsWith("[") && s.endsWith("]"))) {
    try {
      return JSON.parse(s);
    } catch (_) {}
  }

  if (/^-?\d+$/.test(s)) return parseInt(s, 10);
  if (/^-?\d+\.\d+$/.test(s)) return parseFloat(s);

  return s;
}

/**
 * IMPORTANT:
 * Cursor tool execution is most reliable when the assistant produces structured tool calls.
 * We enforce a strict XML tool-call format that we can parse into OpenAI tool_calls.
 */
const TOOL_CALL_INSTRUCTION = `
When you need to use a tool, you MUST output tool calls ONLY in this exact XML format, with NO extra wrappers:

<function_calls>
  <invoke name="TOOL_NAME">
    <parameter name="param1">value</parameter>
    <parameter name="param2">value</parameter>
  </invoke>
</function_calls>

Rules:
- Do NOT write "Calling read_file ..." lines.
- Do NOT claim you ran a tool unless you output an <invoke ...> block.
- Put only the tool calls inside <function_calls> ... </function_calls>.
- After tool results are returned, continue normally.
`.trim();

function openaiToolMessageToAnthropicUserMessage(msg, toolIdToName) {
  // Cursor sends tool results as role=tool with tool_call_id + content.
  // Azure may not honor Anthropic tool_result blocks reliably, so we inject results as user-visible text.
  const toolUseId = msg.tool_call_id || msg.tool_callId || msg.id;
  const toolName = toolUseId ? toolIdToName.get(toolUseId) : null;

  let resultText;
  if (typeof msg.content === "string") resultText = msg.content;
  else {
    try {
      resultText = JSON.stringify(msg.content);
    } catch {
      resultText = String(msg.content);
    }
  }

  const wrapped =
    (toolName ? `TOOL_RESULT for ${toolName}` : "TOOL_RESULT") +
    (toolUseId ? ` (tool_call_id=${toolUseId})` : "") +
    ":\n" +
    resultText;

  return { role: "user", content: toAnthropicContentBlocks(wrapped) };
}

// Parse Cursor/Claude XML tool calls
function parseFunctionCallsFromText(text, allowedToolNames) {
  const tool_calls = [];
  let cleaned = typeof text === "string" ? text : "";

  const isAllowed = (name) => {
    if (!name) return false;
    if (!allowedToolNames || allowedToolNames.size === 0) return true;
    return allowedToolNames.has(name);
  };

  const fcStart = cleaned.indexOf("<function_calls>");
  const fcEnd = cleaned.indexOf("</function_calls>");
  if (fcStart === -1 || fcEnd === -1 || fcEnd <= fcStart) {
    // Strip thinking tags anyway
    cleaned = cleaned.replace(/<thinking>[\s\S]*?<\/thinking>/gi, "").trim();
    return { tool_calls: [], cleanedText: cleaned };
  }

  const block = cleaned.slice(fcStart, fcEnd + "</function_calls>".length);
  cleaned = (cleaned.slice(0, fcStart) + cleaned.slice(fcEnd + "</function_calls>".length))
    .replace(/<thinking>[\s\S]*?<\/thinking>/gi, "")
    .trim();

  const invokeRegex = /<invoke\s+name="([^"]+)"\s*>([\s\S]*?)<\/invoke>/gi;
  const paramRegex = /<parameter\s+name="([^"]+)"\s*>([\s\S]*?)<\/parameter>/gi;

  let m;
  while ((m = invokeRegex.exec(block)) !== null) {
    const toolName = (m[1] || "").trim();
    if (!isAllowed(toolName)) continue;

    const invokeBody = m[2] || "";
    const args = {};

    let p;
    while ((p = paramRegex.exec(invokeBody)) !== null) {
      const key = (p[1] || "").trim();
      const rawVal = (p[2] || "").trim();
      if (!key) continue;
      args[key] = safeParseValue(rawVal);
    }

    tool_calls.push({
      id: "call_" + Date.now().toString(36) + "_" + Math.random().toString(36).slice(2, 10),
      type: "function",
      function: { name: toolName, arguments: JSON.stringify(args || {}) },
    });
  }

  return { tool_calls, cleanedText: cleaned };
}

// Fallback parser: "Calling read_file on path with lines A to B"
function parseCallingLineToolCalls(text, allowedToolNames, toolParamSchemaMap) {
  const tool_calls = [];
  let cleaned = typeof text === "string" ? text : "";

  const isAllowed = (name) => {
    if (!name) return false;
    if (!allowedToolNames || allowedToolNames.size === 0) return true;
    return allowedToolNames.has(name);
  };

  const mkToolCall = (name, argsObj) => ({
    id: "call_" + Date.now().toString(36) + "_" + Math.random().toString(36).slice(2, 10),
    type: "function",
    function: { name, arguments: JSON.stringify(argsObj ?? {}) },
  });

  const lines = cleaned.split(/\r?\n/);
  const out = [];

  // Example patterns we try to handle:
  // "Calling read_file on backend/foo.py with lines 245 to 290"
  // "Calling view_file on /abs/path with lines 10-50"
  // "Calling shell_command cat /path | head -20"
  const callRe1 = /^\s*Calling\s+([A-Za-z0-9_-]+)\s+on\s+(.+?)\s+with\s+lines\s+(\d+)\s*(?:to|-)\s*(\d+)\s*$/i;
  const callRe2 = /^\s*Calling\s+shell_command\s+(.+)\s*$/i;
  const callRe3 = /^\s*Calling\s+([A-Za-z0-9_-]+)\s+(.+)\s*$/i;

  for (const line of lines) {
    let m;

    // shell_command direct
    m = line.match(callRe2);
    if (m) {
      const cmd = (m[1] || "").trim();
      if (isAllowed("shell_command")) {
        tool_calls.push(mkToolCall("shell_command", { command: cmd }));
        continue;
      }
    }

    // read/view_file with lines
    m = line.match(callRe1);
    if (m) {
      const toolName = (m[1] || "").trim();
      const path = (m[2] || "").trim();
      const a = parseInt(m[3], 10);
      const b = parseInt(m[4], 10);

      if (isAllowed(toolName)) {
        const keys = toolParamSchemaMap.get(toolName) || [];

        const pathKey =
          pickFirstKey(keys, ["file_path", "path", "filepath", "filename", "file"]) || "file_path";
        const startKey =
          pickFirstKey(keys, ["start_line", "startLine", "from_line", "line_start", "start"]) || "start_line";
        const endKey =
          pickFirstKey(keys, ["end_line", "endLine", "to_line", "line_end", "end"]) || "end_line";

        const args = {};
        args[pathKey] = path;
        args[startKey] = a;
        args[endKey] = b;

        tool_calls.push(mkToolCall(toolName, args));
        continue;
      }
    }

    // generic "Calling X ..." (weak)
    m = line.match(callRe3);
    if (m) {
      const toolName = (m[1] || "").trim();
      const rest = (m[2] || "").trim();
      if (isAllowed(toolName)) {
        if (toolName === "shell_command") {
          tool_calls.push(mkToolCall("shell_command", { command: rest }));
          continue;
        }
        // For unknown formats, we do not create a tool call because args will almost certainly be wrong.
        // Keep line visible.
      }
    }

    out.push(line);
  }

  cleaned = out.join("\n").replace(/<thinking>[\s\S]*?<\/thinking>/gi, "").trim();
  return { tool_calls, cleanedText: cleaned };
}

// Convert Anthropic content blocks (if Azure ever returns tool_use)
function anthropicContentToTextAndToolCalls(anthropicContent) {
  const textParts = [];
  const toolCalls = [];

  for (const b of anthropicContent || []) {
    if (b?.type === "text" && typeof b.text === "string") {
      textParts.push(b.text);
    } else if (b?.type === "tool_use") {
      toolCalls.push({
        id: b.id,
        type: "function",
        function: { name: b.name, arguments: JSON.stringify(b.input || {}) },
      });
    }
  }

  return { text: textParts.join(""), tool_calls: toolCalls };
}

function mapFinishReason(stopReason) {
  switch (stopReason) {
    case "end_turn":
    case "stop_sequence":
      return "stop";
    case "max_tokens":
      return "length";
    case "tool_use":
      return "tool_calls";
    default:
      return "stop";
  }
}

function transformRequestToAnthropic(openAIRequest) {
  const { messages, model, max_tokens, temperature, tools, ...rest } = openAIRequest;

  const toolIdToName = buildToolIdToNameMap(messages);

  let anthropicMessages = [];
  let systemTextParts = [];

  if (!Array.isArray(messages)) throw new Error("Invalid request format: expected messages[]");

  for (const msg of messages) {
    if (!msg) continue;

    if (msg.role === "system") {
      if (msg.content != null) systemTextParts.push(typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content));
      continue;
    }

    if (msg.role === "tool") {
      anthropicMessages.push(openaiToolMessageToAnthropicUserMessage(msg, toolIdToName));
      continue;
    }

    const roleMapped = msg.role === "assistant" ? "assistant" : "user";
    anthropicMessages.push({ role: roleMapped, content: toAnthropicContentBlocks(msg.content) });
  }

  if (!anthropicMessages.length) throw new Error("Invalid request: no messages");

  const azureModelName = mapModelToDeployment(model);

  // Prepend our tool-call discipline instruction to system prompt.
  const systemFinal = [TOOL_CALL_INSTRUCTION, ...systemTextParts].filter(Boolean).join("\n\n");

  const anthropicRequest = {
    model: azureModelName,
    messages: anthropicMessages,
    max_tokens: max_tokens || 4096,
    system: systemFinal,
  };

  if (temperature !== undefined) anthropicRequest.temperature = temperature;

  // Try passing tools through. If Azure ignores them, we still parse XML.
  const anthTools = openaiToolsToAnthropic(tools);
  if (anthTools.length) anthropicRequest.tools = anthTools;

  // Copy a small set of supported fields
  const supportedFields = ["metadata", "stop_sequences", "top_p", "top_k"];
  for (const f of supportedFields) {
    if (rest[f] !== undefined) anthropicRequest[f] = rest[f];
  }

  return anthropicRequest;
}

function transformAzureToOpenAI(azureResp, requestedModel, allowedToolNames, toolParamSchemaMap) {
  const { content, stop_reason, usage } = azureResp || {};

  // 1) Native tool_use blocks
  const native = anthropicContentToTextAndToolCalls(content);
  let text = native.text || "";
  let toolCalls = native.tool_calls || [];

  // 2) If no native tool_use, parse XML tool calls
  let extractedToolCalls = [];
  if (!toolCalls.length) {
    const parsedXml = parseFunctionCallsFromText(text, allowedToolNames);
    extractedToolCalls = parsedXml.tool_calls;
    text = parsedXml.cleanedText;

    if (extractedToolCalls.length) toolCalls = extractedToolCalls;
  }

  // 3) Fallback parse "Calling ... on ... with lines ..."
  let extractedCalling = [];
  if (!toolCalls.length) {
    const parsedCalling = parseCallingLineToolCalls(text, allowedToolNames, toolParamSchemaMap);
    extractedCalling = parsedCalling.tool_calls;
    text = parsedCalling.cleanedText;

    if (extractedCalling.length) toolCalls = extractedCalling;
  }

  const hasToolCalls = toolCalls.length > 0;

  const msg = {
    role: "assistant",
    content: typeof text === "string" ? text : "",
  };
  if (hasToolCalls) msg.tool_calls = toolCalls;

  const openaiResp = {
    id: makeChatCmplId(),
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: requestedModel || "claude-opus-4-5",
    choices: [
      {
        index: 0,
        message: msg,
        finish_reason: hasToolCalls ? "tool_calls" : mapFinishReason(stop_reason),
      },
    ],
    usage: {
      prompt_tokens: usage?.input_tokens ?? 0,
      completion_tokens: usage?.output_tokens ?? 0,
      total_tokens: (usage?.input_tokens ?? 0) + (usage?.output_tokens ?? 0),
    },
    _debug: {
      stop_reason: stop_reason || null,
      native_tool_calls: native.tool_calls?.length || 0,
      extracted_tool_calls: extractedToolCalls.length || extractedCalling.length || 0,
    },
  };

  return openaiResp;
}

// ---------- SSE (OpenAI streaming) ----------
function createSSE(reqId, res, model) {
  const id = makeChatCmplId();
  const created = Math.floor(Date.now() / 1000);
  const m = model || "claude-opus-4-5";

  res.status(200);
  res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Accel-Buffering", "no");
  res.flushHeaders?.();

  let closed = false;
  let finished = false;

  res.on("finish", () => {
    finished = true;
    console.log(`[${reqId}] [DEBUG] response_finished`);
  });

  res.on("close", () => {
    closed = true;
    // Distinguish early client close vs normal close after end()
    if (!finished && !res.writableEnded) {
      console.log(`[${reqId}] [DEBUG] response_closed_early_by_client`);
    } else {
      console.log(`[${reqId}] [DEBUG] response_closed`);
    }
  });

  const writeChunk = (delta, finish_reason = null) => {
    if (closed || res.destroyed) return;
    const chunk = {
      id,
      object: "chat.completion.chunk",
      created,
      model: m,
      choices: [{ index: 0, delta: delta || {}, finish_reason }],
    };
    res.write(`data: ${JSON.stringify(chunk)}\n\n`);
  };

  const done = (finish_reason = "stop") => {
    if (closed || res.destroyed) return;
    writeChunk({}, finish_reason);
    res.write("data: [DONE]\n\n");
    res.end();
  };

  const error = (message) => {
    // Send an assistant-visible error and terminate cleanly.
    writeChunk({ content: message ? String(message) : "Proxy error" }, null);
    done("stop");
  };

  return {
    id,
    created,
    model: m,
    writeChunk,
    done,
    error,
    isClosed: () => closed || res.destroyed || res.writableEnded,
  };
}

function sseSendToolCalls(reqId, sse, toolCalls) {
  // OpenAI streaming tool_calls format is picky:
  // 1) announce id/name with empty arguments
  // 2) stream arguments in chunks for that same index
  const ARG_CHUNK = 900;

  for (let idx = 0; idx < toolCalls.length; idx++) {
    const tc = toolCalls[idx];
    const name = tc?.function?.name || "unknown_tool";
    const args = tc?.function?.arguments || "";
    const tcId = tc?.id || ("call_" + Date.now().toString(36) + "_" + Math.random().toString(36).slice(2, 10));

    console.log(`[${reqId}] [SSE] tool_call idx=${idx} name=${name} id=${tcId} args_len=${args.length}`);

    // declare call
    sse.writeChunk(
      {
        tool_calls: [
          {
            index: idx,
            id: tcId,
            type: "function",
            function: { name, arguments: "" },
          },
        ],
      },
      null
    );

    // stream args
    for (let i = 0; i < args.length; i += ARG_CHUNK) {
      sse.writeChunk(
        {
          tool_calls: [
            {
              index: idx,
              function: { arguments: args.slice(i, i + ARG_CHUNK) },
            },
          ],
        },
        null
      );
    }
  }
}

// -------------------- Core handler --------------------
async function handleChatCompletions(req, res) {
  const reqId = makeReqId();
  const requestedModel = req.body?.model || "claude-opus-4-5";
  const wantStream = req.body?.stream === true;

  const toolsCount = Array.isArray(req.body?.tools) ? req.body.tools.length : 0;
  const roles = Array.isArray(req.body?.messages) ? req.body.messages.map((m) => m.role).join(",") : "";

  console.log(`[${reqId}] [REQUEST /chat/completions] ${new Date().toISOString()}`);
  console.log(`[${reqId}] Model=${requestedModel} Stream=${wantStream}`);
  console.log(`[${reqId}] Tools present=${toolsCount}`);
  console.log(`[${reqId}] Roles=${roles}`);

  if (CONFIG.DEBUG_LOG_BODY) {
    try {
      console.log(`[${reqId}] [DEBUG] request_body=${JSON.stringify(req.body).slice(0, 20000)}`);
    } catch (_) {}
  }

  const allowedToolNames = getToolNameSet(req.body);
  const toolParamSchemaMap = getToolParamSchemaMap(req.body);

  let sse = null;
  let keepalive = null;

  // True abort signal (client aborts request). This is NOT the same as req "close".
  req.on("aborted", () => {
    console.log(`[${reqId}] [DEBUG] request_aborted_by_client`);
    if (keepalive) clearInterval(keepalive);
  });

  try {
    if (!CONFIG.AZURE_API_KEY) {
      const msg = "Azure API key not configured";
      if (wantStream) {
        sse = createSSE(reqId, res, requestedModel);
        sse.error(msg);
        return;
      }
      return res.status(500).json({ error: { message: msg, type: "configuration_error" } });
    }
    if (!CONFIG.AZURE_ENDPOINT) {
      const msg = "Azure endpoint not configured";
      if (wantStream) {
        sse = createSSE(reqId, res, requestedModel);
        sse.error(msg);
        return;
      }
      return res.status(500).json({ error: { message: msg, type: "configuration_error" } });
    }
    if (!req.body || !Array.isArray(req.body.messages)) {
      const msg = "Invalid request: expected messages[]";
      if (wantStream) {
        sse = createSSE(reqId, res, requestedModel);
        sse.error(msg);
        return;
      }
      return res.status(400).json({ error: { message: msg, type: "invalid_request_error" } });
    }

    // If streaming, start SSE immediately so Cursor doesn't time out.
    if (wantStream) {
      sse = createSSE(reqId, res, requestedModel);

      // initial role chunk
      sse.writeChunk({ role: "assistant" }, null);

      // keepalive as real SSE data chunks (not comments)
      keepalive = setInterval(() => {
        try {
          if (!sse.isClosed()) sse.writeChunk({}, null);
          else clearInterval(keepalive);
        } catch (_) {
          clearInterval(keepalive);
        }
      }, 15000);
    }

    // Always call Azure non-streaming.
    const reqForAzure = { ...req.body, stream: false };

    const anthropicRequest = transformRequestToAnthropic(reqForAzure);
    anthropicRequest.stream = false;

    console.log(`[${reqId}] [AZURE] POST ${CONFIG.AZURE_ENDPOINT}`);
    const response = await axios.post(CONFIG.AZURE_ENDPOINT, anthropicRequest, {
      headers: {
        "Content-Type": "application/json",
        "x-api-key": CONFIG.AZURE_API_KEY,
        "anthropic-version": CONFIG.ANTHROPIC_VERSION,
      },
      timeout: 120000,
      responseType: "json",
      validateStatus: (s) => s < 600,
    });

    console.log(`[${reqId}] [AZURE] Response status=${response.status}`);

    if (keepalive) clearInterval(keepalive);

    if (response.status >= 400) {
      const msg = response.data?.error?.message || response.data?.message || "Azure API error";
      console.error(`[${reqId}] [ERROR] Azure error: ${msg}`);

      if (wantStream && sse) {
        sse.error(msg);
        return;
      }
      return res.status(response.status).json({ error: { message: msg, type: "api_error", code: response.status } });
    }

    if (CONFIG.DEBUG_LOG_BODY) {
      try {
        console.log(`[${reqId}] [DEBUG] azure_body=${JSON.stringify(response.data).slice(0, 20000)}`);
      } catch (_) {}
    }

    const openAIResponse = transformAzureToOpenAI(
      response.data,
      requestedModel,
      allowedToolNames,
      toolParamSchemaMap
    );

    const dbg = openAIResponse._debug || {};
    const toolCalls = openAIResponse?.choices?.[0]?.message?.tool_calls || [];
    const toolNames = toolCalls.map((tc) => tc?.function?.name).filter(Boolean);

    console.log(
      `[${reqId}] [DEBUG] stop_reason=${dbg.stop_reason} native_tool_calls=${dbg.native_tool_calls} extracted_tool_calls=${dbg.extracted_tool_calls}`
    );
    if (toolNames.length) console.log(`[${reqId}] [DEBUG] tool_call_names=${toolNames.join(",")}`);

    // Non-streaming JSON response
    if (!wantStream) {
      res.setHeader("Content-Type", "application/json; charset=utf-8");
      console.log(`[${reqId}] [RESPONSE] Sending JSON`);
      // Remove internal debug field for clients (optional; comment out to keep)
      delete openAIResponse._debug;
      return res.status(200).json(openAIResponse);
    }

    // Streaming SSE response
    console.log(`[${reqId}] [RESPONSE] Sending SSE`);

    if (!sse) sse = createSSE(reqId, res, requestedModel);

    const msg0 = openAIResponse?.choices?.[0]?.message || { role: "assistant", content: "" };
    const text = typeof msg0.content === "string" ? msg0.content : "";
    const hasToolCalls = Array.isArray(msg0.tool_calls) && msg0.tool_calls.length > 0;

    // Stream content (if any)
    if (text.length) {
      // Chunk text to avoid huge single event
      const CHUNK = 1500;
      for (let i = 0; i < text.length; i += CHUNK) {
        sse.writeChunk({ content: text.slice(i, i + CHUNK) }, null);
      }
    }

    // Stream tool calls (if any)
    if (hasToolCalls) {
      sseSendToolCalls(reqId, sse, msg0.tool_calls);
      sse.done("tool_calls");
      return;
    }

    sse.done("stop");
  } catch (e) {
    if (keepalive) clearInterval(keepalive);

    const errMsg = e?.message || String(e);
    console.error(`[${reqId}] [ERROR] /chat/completions exception: ${errMsg}`);

    // If streaming already started, cannot send JSON
    if (wantStream) {
      try {
        if (!sse) sse = createSSE(reqId, res, requestedModel);
        sse.error(errMsg);
      } catch (_) {
        try {
          res.end();
        } catch (_) {}
      }
      return;
    }

    return res.status(500).json({ error: { message: errMsg, type: "proxy_error" } });
  }
}

// -------------------- Endpoints --------------------
app.get("/", (req, res) => {
  res.json({
    status: "running",
    name: "Azure Anthropic Proxy for Cursor",
    config: {
      AZURE_ENDPOINT_set: !!CONFIG.AZURE_ENDPOINT,
      AZURE_API_KEY_set: !!CONFIG.AZURE_API_KEY,
      SERVICE_API_KEY_set: !!CONFIG.SERVICE_API_KEY,
      ANTHROPIC_VERSION: CONFIG.ANTHROPIC_VERSION,
      AZURE_DEPLOYMENT_NAME: CONFIG.AZURE_DEPLOYMENT_NAME,
      DEBUG_LOG_BODY: CONFIG.DEBUG_LOG_BODY,
    },
    endpoints: {
      health: "/health",
      chat_cursor: "/chat/completions",
      chat_openai: "/v1/chat/completions",
      models: "/v1/models",
    },
  });
});

app.get("/health", (req, res) => {
  res.json({
    status: "ok",
    timestamp: new Date().toISOString(),
    AZURE_ENDPOINT_set: !!CONFIG.AZURE_ENDPOINT,
    AZURE_API_KEY_set: !!CONFIG.AZURE_API_KEY,
    SERVICE_API_KEY_set: !!CONFIG.SERVICE_API_KEY,
    port: CONFIG.PORT,
  });
});

// OpenAI-compatible models endpoints (Cursor often calls this)
app.get("/v1/models", requireAuth, (req, res) => {
  const now = Math.floor(Date.now() / 1000);
  res.json({
    object: "list",
    data: [
      { id: "claude-opus-4-5", object: "model", created: now, owned_by: "proxy" },
      { id: "claude-sonnet-4-5", object: "model", created: now, owned_by: "proxy" },
    ],
  });
});
app.get("/models", requireAuth, (req, res) => {
  const now = Math.floor(Date.now() / 1000);
  res.json({
    object: "list",
    data: [
      { id: "claude-opus-4-5", object: "model", created: now, owned_by: "proxy" },
      { id: "claude-sonnet-4-5", object: "model", created: now, owned_by: "proxy" },
    ],
  });
});

// Cursor uses this
app.post("/chat/completions", requireAuth, handleChatCompletions);
// Some clients call this
app.post("/v1/chat/completions", requireAuth, handleChatCompletions);

// Optional: Anthropic-native passthrough for debugging
app.post("/v1/messages", async (req, res) => {
  try {
    if (!CONFIG.AZURE_API_KEY) throw new Error("Azure API key not configured");
    if (!CONFIG.AZURE_ENDPOINT) throw new Error("Azure endpoint not configured");

    const response = await axios.post(CONFIG.AZURE_ENDPOINT, req.body, {
      headers: {
        "Content-Type": "application/json",
        "x-api-key": CONFIG.AZURE_API_KEY,
        "anthropic-version": req.headers["anthropic-version"] || CONFIG.ANTHROPIC_VERSION,
      },
      timeout: 120000,
      responseType: "json",
      validateStatus: (s) => s < 600,
    });

    res.status(response.status).json(response.data);
  } catch (e) {
    res.status(500).json({ error: { message: e?.message || "proxy_error", type: "proxy_error" } });
  }
});

// 404
app.use((req, res) => {
  res.status(404).json({
    error: {
      message: "Endpoint not found. Available: GET /, GET /health, GET /v1/models, POST /chat/completions, POST /v1/chat/completions, POST /v1/messages",
      type: "not_found",
    },
  });
});

// Start
app.listen(CONFIG.PORT, "0.0.0.0", () => {
  console.log("\n" + "=".repeat(80));
  console.log("Azure Anthropic Proxy - Running");
  console.log("=".repeat(80));
  console.log(`Listening on 0.0.0.0:${CONFIG.PORT}`);
  console.log(`AZURE_ENDPOINT set: ${!!CONFIG.AZURE_ENDPOINT}`);
  console.log(`AZURE_API_KEY set: ${!!CONFIG.AZURE_API_KEY}`);
  console.log(`SERVICE_API_KEY set: ${!!CONFIG.SERVICE_API_KEY}`);
  console.log(`ANTHROPIC_VERSION: ${CONFIG.ANTHROPIC_VERSION}`);
  console.log(`AZURE_DEPLOYMENT_NAME: ${CONFIG.AZURE_DEPLOYMENT_NAME}`);
  console.log(`DEBUG_LOG_BODY: ${CONFIG.DEBUG_LOG_BODY}`);
  console.log("Endpoints:");
  console.log("  GET  /health");
  console.log("  GET  /v1/models");
  console.log("  POST /chat/completions");
  console.log("  POST /v1/chat/completions");
  console.log("  POST /v1/messages");
  console.log("=".repeat(80) + "\n");
});

