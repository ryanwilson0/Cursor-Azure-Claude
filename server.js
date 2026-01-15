// server.js
const express = require("express");
const axios = require("axios");

const app = express();
app.use(express.json({ limit: "50mb" }));

/**
 * ENV VARS REQUIRED
 * - AZURE_ENDPOINT   (full Anthropic /v1/messages endpoint URL for your Azure Anthropic deployment)
 * - AZURE_API_KEY
 * - SERVICE_API_KEY  (paste into Cursor: Settings > Models > API Keys > OpenAI API Key)
 *
 * Optional:
 * - AZURE_DEPLOYMENT_NAME (defaults to "claude-opus-4-5")
 * - ANTHROPIC_VERSION (defaults to "2023-06-01")
 * - PORT (defaults to 8080)
 * - DEBUG_LOG_BODY ("1" to log more request/response details)
 */
const CONFIG = {
  AZURE_ENDPOINT: process.env.AZURE_ENDPOINT,
  AZURE_API_KEY: process.env.AZURE_API_KEY,
  SERVICE_API_KEY: process.env.SERVICE_API_KEY,
  PORT: process.env.PORT || 8080,
  ANTHROPIC_VERSION: process.env.ANTHROPIC_VERSION || "2023-06-01",
  AZURE_DEPLOYMENT_NAME: process.env.AZURE_DEPLOYMENT_NAME || "claude-opus-4-5",
  DEBUG_LOG_BODY: process.env.DEBUG_LOG_BODY === "1",
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

// -------------------- CORS --------------------
app.use((req, res, next) => {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.header(
    "Access-Control-Allow-Headers",
    "Content-Type, Authorization, x-api-key, anthropic-version, OpenAI-Organization, OpenAI-Project"
  );
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

// -------------------- Utilities --------------------
function makeReqId() {
  return "req_" + Date.now().toString(36) + "_" + Math.random().toString(36).slice(2, 8);
}

function makeChatCmplId() {
  return "chatcmpl-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 10);
}

function summarizeTools(tools) {
  const out = [];
  const arr = Array.isArray(tools) ? tools : [];
  for (const t of arr) {
    const name = t?.function?.name || t?.name || "unknown";
    const params = t?.function?.parameters || t?.input_schema || null;
    const keys =
      params && params.properties && typeof params.properties === "object"
        ? Object.keys(params.properties)
        : [];
    out.push({ name, paramKeys: keys });
  }
  return out;
}

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
  // Map tool_call_id -> tool name from prior assistant tool_calls (OpenAI format)
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

function safeParseValue(v) {
  if (typeof v !== "string") return v;
  const s = v.trim();
  if (!s) return s;

  // JSON
  if ((s.startsWith("{") && s.endsWith("}")) || (s.startsWith("[") && s.endsWith("]"))) {
    try {
      return JSON.parse(s);
    } catch (_) {}
  }

  // int
  if (/^-?\d+$/.test(s)) return parseInt(s, 10);

  // float
  if (/^-?\d+\.\d+$/.test(s)) return parseFloat(s);

  return s;
}

function getAllowedToolNames(reqBody) {
  const set = new Set();
  const tools = Array.isArray(reqBody?.tools) ? reqBody.tools : [];
  for (const t of tools) {
    if (t?.type === "function" && t?.function?.name) set.add(t.function.name);
  }
  return set;
}

function openaiToolMessageToAnthropicToolResult(msg, toolIdToName, reqId) {
  // OpenAI tool message: { role:"tool", tool_call_id:"...", content:"..." }
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

  console.log(`[${reqId}] [TOOL_RESULT IN] tool_call_id=${toolUseId || "missing"} tool=${toolName || "unknown"} len=${resultText.length}`);

  // Preferred: real Anthropic tool_result (lets Claude reliably continue tool loop)
  if (toolUseId) {
    return {
      role: "user",
      content: [
        {
          type: "tool_result",
          tool_use_id: toolUseId,
          content: [{ type: "text", text: resultText }],
        },
      ],
    };
  }

  // Fallback if no tool_use_id
  const wrapped = (toolName ? `TOOL_RESULT for ${toolName}` : "TOOL_RESULT") + ":\n" + resultText;
  return { role: "user", content: toAnthropicContentBlocks(wrapped) };
}

function mapFinishReason(stopReason) {
  // Anthropic stop_reason -> OpenAI finish_reason
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

/**
 * Tool-use coercion prompt:
 * If Azure ignores tools, Claude will often narrate "Calling read_file..." instead of producing tool calls.
 * This prompt instructs it to emit a strict XML tool-call block we can parse into OpenAI tool_calls.
 */
function buildToolUseCoercionSystem(tools) {
  const arr = Array.isArray(tools) ? tools : [];
  if (!arr.length) return "";

  const lines = [];
  lines.push("TOOL CALLING PROTOCOL (IMPORTANT):");
  lines.push("You have tools available. When you need to use a tool, you MUST output a <function_calls> XML block exactly in this format:");
  lines.push("");
  lines.push("<function_calls>");
  lines.push('  <invoke name="TOOL_NAME">');
  lines.push('    <parameter name="param1">value</parameter>');
  lines.push("    ...");
  lines.push("  </invoke>");
  lines.push("</function_calls>");
  lines.push("");
  lines.push("Rules:");
  lines.push("1) Do NOT write 'Calling tool...' in plain text. Use the XML block only.");
  lines.push("2) Include ALL required parameters for the tool, using the exact parameter names.");
  lines.push("3) Put ONLY one <function_calls> block at the end of your message when calling tools.");
  lines.push("4) Do NOT include <thinking> tags.");
  lines.push("");

  lines.push("Available tools and parameter schemas (use these exact names):");
  for (const t of arr) {
    if (t?.type !== "function" || !t.function?.name) continue;
    const name = t.function.name;
    const params = t.function.parameters || {};
    const props = params.properties && typeof params.properties === "object" ? params.properties : {};
    const required = Array.isArray(params.required) ? params.required : [];
    const propKeys = Object.keys(props);

    lines.push(`- ${name}`);
    if (required.length) lines.push(`  required: ${required.join(", ")}`);
    if (propKeys.length) {
      lines.push(`  params: ${propKeys.join(", ")}`);
    } else {
      lines.push("  params: (none specified)");
    }
  }

  return lines.join("\n");
}

/**
 * Extract tool calls from assistant text (fallback when upstream doesn't return tool_use blocks).
 * Handles:
 *  - Cursor XML <function_calls> ... </function_calls>
 *  - Plain "Calling read_file on PATH with lines A to B"
 *  - Plain "Calling shell_command CMD"
 */
function extractToolCallsFromAssistantText(text, allowedToolNames) {
  const tool_calls = [];
  let cleaned = typeof text === "string" ? text : "";

  const mkToolCall = (name, argsObj) => ({
    id: "call_" + Math.random().toString(36).slice(2, 10),
    type: "function",
    function: {
      name,
      arguments: JSON.stringify(argsObj ?? {}),
    },
  });

  const isAllowed = (name) => {
    if (!name) return false;
    if (!allowedToolNames || allowedToolNames.size === 0) return true;
    return allowedToolNames.has(name);
  };

  // Strip <thinking> from visible content
  cleaned = cleaned.replace(/<thinking>[\s\S]*?<\/thinking>/gi, "");

  // 1) Parse XML tool calls
  if (cleaned.includes("<function_calls>") && cleaned.includes("</function_calls>")) {
    const blockRe = /<function_calls>[\s\S]*?<\/function_calls>/gi;
    const invokeRe = /<invoke\s+name="([^"]+)"\s*>([\s\S]*?)<\/invoke>/gi;
    const paramRe = /<parameter\s+name="([^"]+)"\s*>([\s\S]*?)<\/parameter>/gi;

    const blocks = cleaned.match(blockRe) || [];
    for (const block of blocks) {
      let m;
      while ((m = invokeRe.exec(block))) {
        const toolName = (m[1] || "").trim();
        if (!isAllowed(toolName)) continue;

        const body = m[2] || "";
        const args = {};
        let pm;
        while ((pm = paramRe.exec(body))) {
          const k = (pm[1] || "").trim();
          const v = safeParseValue(pm[2] || "");
          if (k) args[k] = v;
        }
        tool_calls.push(mkToolCall(toolName, args));
      }
    }

    // Remove all function_calls blocks from visible content
    cleaned = cleaned.replace(blockRe, "").trim();
  }

  // 2) Parse "Calling read_file on X with lines A to B" (common hallucinated phrasing)
  {
    const lines = cleaned.split(/\r?\n/);
    const kept = [];
    for (const line of lines) {
      const m = line.match(/^\s*Calling\s+([A-Za-z0-9_-]+)\s+on\s+(.+?)\s+with\s+lines\s+(\d+)\s+to\s+(\d+)\s*$/i);
      if (m) {
        const toolName = (m[1] || "").trim();
        const filePath = (m[2] || "").trim();
        const startLine = parseInt(m[3], 10);
        const endLine = parseInt(m[4], 10);
        if (isAllowed(toolName)) {
          tool_calls.push(mkToolCall(toolName, { file_path: filePath, start_line: startLine, end_line: endLine }));
          continue; // drop this line from visible content
        }
      }
      kept.push(line);
    }
    cleaned = kept.join("\n").trim();
  }

  // 3) Parse "Calling shell_command <cmd>" lines
  {
    const lines = cleaned.split(/\r?\n/);
    const kept = [];
    for (const line of lines) {
      const m = line.match(/^\s*Calling\s+([A-Za-z0-9_-]+)\s+(.+?)\s*$/i);
      if (!m) {
        kept.push(line);
        continue;
      }
      const toolName = (m[1] || "").trim();
      const rest = (m[2] || "").trim();

      if (!isAllowed(toolName)) {
        kept.push(line);
        continue;
      }

      if (toolName === "shell_command") {
        tool_calls.push(mkToolCall("shell_command", { command: rest }));
        continue; // drop line
      }

      // If it’s a known tool but we can’t parse args safely, keep line visible.
      // The coercion system prompt is what should make args appear in XML for these tools.
      kept.push(line);
    }
    cleaned = kept.join("\n").trim();
  }

  return { cleaned, tool_calls };
}

// -------------------- Transforms --------------------
function transformRequestToAnthropic(openAIRequest, reqId) {
  const { messages, model, max_tokens, temperature, tools, ...rest } = openAIRequest;

  if (!Array.isArray(messages)) {
    throw new Error("Invalid request format: expected messages[]");
  }

  const toolIdToName = buildToolIdToNameMap(messages);

  const anthropicMessages = [];
  const systemTextParts = [];

  for (const msg of messages) {
    if (!msg) continue;

    if (msg.role === "system") {
      if (msg.content != null) {
        systemTextParts.push(typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content));
      }
      continue;
    }

    if (msg.role === "tool") {
      anthropicMessages.push(openaiToolMessageToAnthropicToolResult(msg, toolIdToName, reqId));
      continue;
    }

    const roleMapped = msg.role === "assistant" ? "assistant" : "user";
    anthropicMessages.push({
      role: roleMapped,
      content: toAnthropicContentBlocks(msg.content),
    });
  }

  if (!anthropicMessages.length) throw new Error("Invalid request: no messages");

  const azureModelName = mapModelToDeployment(model);

  const anthropicRequest = {
    model: azureModelName,
    messages: anthropicMessages,
    max_tokens: max_tokens || 4096,
  };

  // Build system string
  const baseSystem =
    systemTextParts.length ? systemTextParts.join("\n\n") : rest.system !== undefined ? rest.system : undefined;

  const coercion = buildToolUseCoercionSystem(tools);
  if (baseSystem && coercion) anthropicRequest.system = String(baseSystem) + "\n\n" + coercion;
  else if (baseSystem) anthropicRequest.system = String(baseSystem);
  else if (coercion) anthropicRequest.system = coercion;

  if (temperature !== undefined) anthropicRequest.temperature = temperature;

  // Pass tools through to Anthropic if supported upstream
  const anthTools = openaiToolsToAnthropic(tools);
  if (anthTools.length) anthropicRequest.tools = anthTools;

  // Copy a small set of fields that are commonly accepted
  const supportedFields = ["metadata", "stop_sequences", "top_p", "top_k"];
  for (const f of supportedFields) {
    if (rest[f] !== undefined) anthropicRequest[f] = rest[f];
  }

  if (CONFIG.DEBUG_LOG_BODY) {
    console.log(`[${reqId}] [DEBUG] anthropicRequest.model=${anthropicRequest.model}`);
    console.log(`[${reqId}] [DEBUG] anthropicRequest.messages=${anthropicRequest.messages.length}`);
    console.log(`[${reqId}] [DEBUG] anthropicRequest.system_len=${(anthropicRequest.system || "").length}`);
    console.log(`[${reqId}] [DEBUG] anthropicRequest.tools=${Array.isArray(anthropicRequest.tools) ? anthropicRequest.tools.length : 0}`);
  }

  return anthropicRequest;
}

function anthropicContentToOpenAIMessage(contentBlocks) {
  const textParts = [];
  const toolCalls = [];

  for (const b of contentBlocks || []) {
    if (b?.type === "text" && typeof b.text === "string") {
      textParts.push(b.text);
    } else if (b?.type === "tool_use") {
      toolCalls.push({
        id: b.id,
        type: "function",
        function: {
          name: b.name,
          arguments: JSON.stringify(b.input || {}),
        },
      });
    }
  }

  const msg = {
    role: "assistant",
    content: textParts.join("") || "",
  };

  if (toolCalls.length) msg.tool_calls = toolCalls;

  return msg;
}

function transformAnthropicToOpenAI(anthropicResp, requestedModel, allowedToolNames, reqId) {
  const { content, stop_reason, usage } = anthropicResp || {};

  const assistantMessage = anthropicContentToOpenAIMessage(content);

  const nativeToolCalls = Array.isArray(assistantMessage.tool_calls) ? assistantMessage.tool_calls : [];
  const nativeCount = nativeToolCalls.length;

  // Fallback parsing when tool calls show up as text
  if (nativeCount === 0) {
    const parsed = extractToolCallsFromAssistantText(assistantMessage.content || "", allowedToolNames);
    if (parsed.tool_calls.length > 0) {
      assistantMessage.content = parsed.cleaned || "";
      assistantMessage.tool_calls = parsed.tool_calls;
    }
  }

  const finalToolCalls = Array.isArray(assistantMessage.tool_calls) ? assistantMessage.tool_calls : [];
  const hasToolCalls = finalToolCalls.length > 0;

  console.log(
    `[${reqId}] [DEBUG] stop_reason=${stop_reason || "unknown"} native_tool_calls=${nativeCount} extracted_tool_calls=${finalToolCalls.length}`
  );
  if (finalToolCalls.length) {
    console.log(
      `[${reqId}] [DEBUG] tool_call_names=${finalToolCalls.map((t) => t?.function?.name || "unknown").join(", ")}`
    );
  }

  return {
    id: makeChatCmplId(),
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: requestedModel || "claude-opus-4-5",
    choices: [
      {
        index: 0,
        message: assistantMessage,
        finish_reason: hasToolCalls ? "tool_calls" : mapFinishReason(stop_reason),
      },
    ],
    usage: {
      prompt_tokens: usage?.input_tokens ?? 0,
      completion_tokens: usage?.output_tokens ?? 0,
      total_tokens: (usage?.input_tokens ?? 0) + (usage?.output_tokens ?? 0),
    },
  };
}

// -------------------- SSE Streaming --------------------
function createSSE(res, model, reqId) {
  const id = makeChatCmplId();
  const created = Math.floor(Date.now() / 1000);
  const m = model || "claude-opus-4-5";

  res.status(200);
  res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Accel-Buffering", "no");
  res.flushHeaders?.();

  const writeChunk = (delta, finish_reason = null) => {
    const chunk = {
      id,
      object: "chat.completion.chunk",
      created,
      model: m,
      choices: [{ index: 0, delta: delta || {}, finish_reason }],
    };
    res.write(`data: ${JSON.stringify(chunk)}\n\n`);
  };

  const writeComment = (comment) => {
    res.write(`: ${comment || "keepalive"}\n\n`);
  };

  const done = (finish_reason = "stop") => {
    writeChunk({}, finish_reason);
    res.write("data: [DONE]\n\n");
    res.end();
  };

  const error = (message) => {
    console.log(`[${reqId}] [SSE] error=${String(message || "proxy_error")}`);
    writeChunk({ content: message ? String(message) : "Proxy error" }, null);
    done("stop");
  };

  return { id, created, model: m, writeChunk, writeComment, done, error };
}

function sseSendToolCalls(sse, toolCalls, reqId) {
  // Cursor is picky: stream tool_calls as OpenAI-style deltas with indices and chunked arguments.
  const ARG_CHUNK = 1200;

  for (let idx = 0; idx < toolCalls.length; idx++) {
    const tc = toolCalls[idx];
    const name = tc?.function?.name || "unknown_tool";
    const args = tc?.function?.arguments || "";
    const tcId = tc?.id || ("call_" + Math.random().toString(36).slice(2, 10));

    console.log(`[${reqId}] [SSE] tool_call idx=${idx} name=${name} id=${tcId} args_len=${args.length}`);

    // announce id/name
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

// -------------------- Main Handler --------------------
async function handleChatCompletions(req, res) {
  const reqId = makeReqId();

  console.log(`[${reqId}] [REQUEST /chat/completions] ${new Date().toISOString()}`);
  console.log(`[${reqId}] Model=${req.body?.model} Stream=${req.body?.stream}`);

  const toolsCount = Array.isArray(req.body?.tools) ? req.body.tools.length : 0;
  console.log(`[${reqId}] Tools present=${toolsCount}`);
  console.log(
    `[${reqId}] Roles=${Array.isArray(req.body?.messages) ? req.body.messages.map((m) => m.role).join(",") : "n/a"}`
  );

  if (CONFIG.DEBUG_LOG_BODY) {
    console.log(`[${reqId}] [DEBUG] tools_summary=${JSON.stringify(summarizeTools(req.body?.tools || []))}`);
    console.log(`[${reqId}] [DEBUG] body_keys=${req.body ? Object.keys(req.body).join(",") : "null"}`);
  }

  const requestedModel = req.body?.model || "claude-opus-4-5";
  const wantStream = req.body?.stream === true;
  const allowedToolNames = getAllowedToolNames(req.body);

  let sse = null;
  let keepalive = null;
  let clientClosed = false;

  // If streaming, open SSE immediately to prevent Cursor timeouts.
  if (wantStream) {
    sse = createSSE(res, requestedModel, reqId);
    sse.writeChunk({ role: "assistant" }, null);

    keepalive = setInterval(() => {
      try {
        sse.writeComment("keepalive");
      } catch (_) {
        clearInterval(keepalive);
      }
    }, 15000);

    req.on("close", () => {
      clientClosed = true;
      if (keepalive) clearInterval(keepalive);
      console.log(`[${reqId}] [DEBUG] client_disconnected`);
    });
  }

  try {
    if (!CONFIG.AZURE_API_KEY) {
      const msg = "Azure API key not configured";
      if (wantStream && sse) return sse.error(msg);
      return res.status(500).json({ error: { message: msg, type: "configuration_error" } });
    }
    if (!CONFIG.AZURE_ENDPOINT) {
      const msg = "Azure endpoint not configured";
      if (wantStream && sse) return sse.error(msg);
      return res.status(500).json({ error: { message: msg, type: "configuration_error" } });
    }
    if (!req.body || !Array.isArray(req.body.messages)) {
      const msg = "Invalid request: expected messages[]";
      if (wantStream && sse) return sse.error(msg);
      return res.status(400).json({ error: { message: msg, type: "invalid_request_error" } });
    }

    // Always call Azure non-streaming; we stream to Cursor ourselves.
    const reqForAzure = { ...req.body, stream: false };

    const anthropicRequest = transformRequestToAnthropic(reqForAzure, reqId);
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

    if (clientClosed) {
      console.log(`[${reqId}] [DEBUG] client_closed_before_response_processing`);
      return;
    }

    if (response.status >= 400) {
      const msg = response.data?.error?.message || response.data?.message || "Azure API error";
      console.error(`[${reqId}] [ERROR] Azure error: ${msg}`);
      if (wantStream && sse) return sse.error(msg);
      return res.status(response.status).json({ error: { message: msg, type: "api_error", code: response.status } });
    }

    if (CONFIG.DEBUG_LOG_BODY) {
      const contentTypes = Array.isArray(response.data?.content) ? response.data.content.map((b) => b?.type) : [];
      console.log(`[${reqId}] [DEBUG] azure.stop_reason=${response.data?.stop_reason}`);
      console.log(`[${reqId}] [DEBUG] azure.content_types=${JSON.stringify(contentTypes)}`);
      const txt = Array.isArray(response.data?.content)
        ? response.data.content.filter((b) => b?.type === "text").map((b) => b.text).join("")
        : "";
      console.log(`[${reqId}] [DEBUG] azure.text_preview=${JSON.stringify((txt || "").slice(0, 400))}`);
    }

    const openAIResponse = transformAnthropicToOpenAI(response.data, requestedModel, allowedToolNames, reqId);

    // Non-stream response
    if (!wantStream) {
      res.setHeader("Content-Type", "application/json; charset=utf-8");
      console.log(`[${reqId}] [RESPONSE] Sending JSON`);
      return res.status(200).json(openAIResponse);
    }

    // Stream response
    console.log(`[${reqId}] [RESPONSE] Sending SSE`);

    const choice0 = openAIResponse?.choices?.[0] || {};
    const msg0 = choice0.message || { role: "assistant", content: "" };
    const toolCalls = Array.isArray(msg0.tool_calls) ? msg0.tool_calls : [];
    const hasToolCalls = toolCalls.length > 0;

    const text = typeof msg0.content === "string" ? msg0.content : "";
    if (text.length) {
      const CHUNK = 1500;
      for (let i = 0; i < text.length; i += CHUNK) {
        sse.writeChunk({ content: text.slice(i, i + CHUNK) }, null);
      }
    }

    if (hasToolCalls) {
      sseSendToolCalls(sse, toolCalls, reqId);
      return sse.done("tool_calls");
    }

    return sse.done("stop");
  } catch (e) {
    if (keepalive) clearInterval(keepalive);

    const errMsg = e?.message || String(e);
    console.error(`[${reqId}] [ERROR] /chat/completions exception: ${errMsg}`);

    if (wantStream && sse) return sse.error(errMsg);

    return res.status(500).json({ error: { message: errMsg, type: "proxy_error" } });
  }
}

// -------------------- Endpoints --------------------
app.get("/", (req, res) => {
  res.json({
    status: "running",
    name: "Azure Anthropic Proxy for Cursor",
    endpoints: {
      health: "/health",
      chat_cursor: "/chat/completions",
      chat_openai: "/v1/chat/completions",
      models: "/v1/models",
      debug_messages: "/v1/messages",
    },
    config: {
      apiKeyConfigured: !!CONFIG.AZURE_API_KEY,
      endpointConfigured: !!CONFIG.AZURE_ENDPOINT,
      deployment: CONFIG.AZURE_DEPLOYMENT_NAME,
      anthropicVersion: CONFIG.ANTHROPIC_VERSION,
      debugLogging: CONFIG.DEBUG_LOG_BODY,
    },
  });
});

app.get("/health", (req, res) => {
  res.json({
    status: "ok",
    timestamp: new Date().toISOString(),
    apiKeyConfigured: !!CONFIG.AZURE_API_KEY,
    endpointConfigured: !!CONFIG.AZURE_ENDPOINT,
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

// Optional: Anthropic-native passthrough for debugging (auth-protected)
app.post("/v1/messages", requireAuth, async (req, res) => {
  const reqId = makeReqId();
  console.log(`[${reqId}] [REQUEST /v1/messages] ${new Date().toISOString()}`);

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

    console.log(`[${reqId}] [AZURE] status=${response.status}`);
    res.status(response.status).json(response.data);
  } catch (e) {
    console.error(`[${reqId}] [ERROR] /v1/messages: ${e?.message || e}`);
    res.status(500).json({ error: { message: e?.message || "proxy_error", type: "proxy_error" } });
  }
});

// 404
app.use((req, res) => {
  res.status(404).json({
    error: {
      message:
        "Endpoint not found. Available: GET /, GET /health, GET /v1/models, POST /chat/completions, POST /v1/chat/completions, POST /v1/messages",
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
  console.log("  POST /v1/messages (debug passthrough)");
  console.log("=".repeat(80) + "\n");
});
