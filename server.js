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
 * - PORT (defaults to 8080)
 */
const CONFIG = {
  AZURE_ENDPOINT: process.env.AZURE_ENDPOINT,
  AZURE_API_KEY: process.env.AZURE_API_KEY,
  SERVICE_API_KEY: process.env.SERVICE_API_KEY,
  PORT: process.env.PORT || 8080,
  ANTHROPIC_VERSION: process.env.ANTHROPIC_VERSION || "2023-06-01",
  AZURE_DEPLOYMENT_NAME: process.env.AZURE_DEPLOYMENT_NAME || "claude-opus-4-5",
};

// Model name mapping: common model names that should be mapped to Azure deployment
const MODEL_NAMES_TO_MAP = [
  "gpt-4",
  "gpt-4.1",
  "gpt-4o",
  "claude-opus-4-5",
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
  if (req.method === "OPTIONS" || req.path === "/health" || req.path === "/" ) return next();

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

  let token = authHeader.startsWith("Bearer ") ? authHeader.slice(7) : authHeader;
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
function parseCallingToolLines(text) {
  if (!text || typeof text !== "string") return { tool_calls: [], cleanedText: text || "" };

  const tool_calls = [];
  const outLines = [];

  const lines = text.split(/\r?\n/);
  for (const line of lines) {
    const m = line.match(/^\s*Calling\s+([a-zA-Z0-9_]+)\s+([\s\S]+)\s*$/);
    if (!m) {
      outLines.push(line);
      continue;
    }

    const toolName = m[1];
    const rest = m[2];

    // Cursor's shell_command tool almost always wants { "command": "..." }
    let argsObj;
    if (toolName === "shell_command") {
      argsObj = { command: rest };
    } else {
      // Attempt: if it contains JSON, parse it; else pass as { input: "..." }
      const jsonStart = rest.indexOf("{");
      const jsonEnd = rest.lastIndexOf("}");
      if (jsonStart !== -1 && jsonEnd !== -1 && jsonEnd > jsonStart) {
        const jsonStr = rest.slice(jsonStart, jsonEnd + 1);
        try {
          argsObj = JSON.parse(jsonStr);
        } catch {
          argsObj = { input: rest };
        }
      } else {
        argsObj = { input: rest };
      }
    }

    tool_calls.push({
      id: "call_" + Date.now().toString(36) + "_" + Math.random().toString(36).slice(2, 10),
      type: "function",
      function: { name: toolName, arguments: JSON.stringify(argsObj) },
    });
  }

  return { tool_calls, cleanedText: outLines.join("\n").trim() };
}

function openaiToolMessageToAnthropicUserMessage(msg, toolIdToName) {
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

  // IMPORTANT: We do NOT rely on Anthropic-native tool_result because Azure may not honor it.
  // We pass tool results as plain user-visible text so Claude can continue correctly.
  const wrapped =
    (toolName ? `TOOL_RESULT for ${toolName}` : "TOOL_RESULT") +
    (toolUseId ? ` (tool_call_id=${toolUseId})` : "") +
    ":\n" +
    resultText;

  return { role: "user", content: toAnthropicContentBlocks(wrapped) };
}

// Parse Cursor/Claude text tool markup like:
// <function_calls>
//   <invoke name="read_file">
//     <parameter name="file_path">...</parameter>
//   </invoke>
// </function_calls>
function parseFunctionCallsFromText(text) {
  if (!text || typeof text !== "string") return { tool_calls: [], cleanedText: text || "" };

  const toolCalls = [];

  const fcStart = text.indexOf("<function_calls>");
  const fcEnd = text.indexOf("</function_calls>");
  let cleaned = text;

  let block = null;
  if (fcStart !== -1 && fcEnd !== -1 && fcEnd > fcStart) {
    block = text.slice(fcStart, fcEnd + "</function_calls>".length);
    cleaned = (text.slice(0, fcStart) + text.slice(fcEnd + "</function_calls>".length)).trim();
  }

  // Also strip <thinking>...</thinking> (Cursor users do not want it surfaced)
  cleaned = cleaned.replace(/<thinking>[\s\S]*?<\/thinking>/g, "").trim();

  if (!block) return { tool_calls: [], cleanedText: cleaned };

  const invokeRegex = /<invoke\s+name="([^"]+)">([\s\S]*?)<\/invoke>/g;
  const paramRegex = /<parameter\s+name="([^"]+)">([\s\S]*?)<\/parameter>/g;

  let m;
  while ((m = invokeRegex.exec(block)) !== null) {
    const toolName = (m[1] || "").trim();
    const invokeBody = m[2] || "";
    const args = {};

    let p;
    while ((p = paramRegex.exec(invokeBody)) !== null) {
      const key = (p[1] || "").trim();
      const rawVal = (p[2] || "").trim();

      // Try to parse JSON-looking values; otherwise keep string
      let val = rawVal;
      if ((rawVal.startsWith("{") && rawVal.endsWith("}")) || (rawVal.startsWith("[") && rawVal.endsWith("]"))) {
        try {
          val = JSON.parse(rawVal);
        } catch {
          val = rawVal;
        }
      }
      args[key] = val;
    }

    if (toolName) {
      const id = "call_" + Date.now().toString(36) + "_" + Math.random().toString(36).slice(2, 10);
      toolCalls.push({
        id,
        type: "function",
        function: {
          name: toolName,
          arguments: JSON.stringify(args || {}),
        },
      });
    }
  }

  return { tool_calls: toolCalls, cleanedText: cleaned };
}

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
        function: {
          name: b.name,
          arguments: JSON.stringify(b.input || {}),
        },
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

function makeChatCmplId() {
  return "chatcmpl-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 10);
}

function transformRequestToAnthropic(openAIRequest) {
  const { messages, model, max_tokens, temperature, tools, ...rest } = openAIRequest;

  const toolIdToName = buildToolIdToNameMap(messages);

  let anthropicMessages = [];
  let systemTextParts = [];

  if (Array.isArray(messages)) {
    for (const msg of messages) {
      if (!msg) continue;

      if (msg.role === "system") {
        if (msg.content != null) {
          systemTextParts.push(typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content));
        }
        continue;
      }

      if (msg.role === "tool") {
        anthropicMessages.push(openaiToolMessageToAnthropicUserMessage(msg, toolIdToName));
        continue;
      }

      const roleMapped = msg.role === "assistant" ? "assistant" : "user";
      anthropicMessages.push({
        role: roleMapped,
        content: toAnthropicContentBlocks(msg.content),
      });
    }
  } else {
    throw new Error("Invalid request format: expected messages[]");
  }

  if (!anthropicMessages.length) throw new Error("Invalid request: no messages");

  const azureModelName = mapModelToDeployment(model);

  const anthropicRequest = {
    model: azureModelName,
    messages: anthropicMessages,
    max_tokens: max_tokens || 4096,
  };

  if (systemTextParts.length) {
    anthropicRequest.system = systemTextParts.join("\n\n");
  } else if (rest.system !== undefined) {
    anthropicRequest.system = rest.system;
  }

  if (temperature !== undefined) anthropicRequest.temperature = temperature;

  // Try passing tools through (safe if Azure supports it). If Azure ignores it, we still parse tool markup from text.
  const anthTools = openaiToolsToAnthropic(tools);
  if (anthTools.length) anthropicRequest.tools = anthTools;

  // Copy a small set of supported fields
  const supportedFields = ["metadata", "stop_sequences", "top_p", "top_k"];
  for (const f of supportedFields) {
    if (rest[f] !== undefined) anthropicRequest[f] = rest[f];
  }

  return anthropicRequest;
}

function transformAzureToOpenAI(azureResp, requestedModel) {
  const { content, stop_reason, usage } = azureResp || {};

  // 1) If Azure returns Anthropic-native tool_use blocks, convert those directly.
  const { text, tool_calls: nativeToolCalls } = anthropicContentToTextAndToolCalls(content);

  // 2) If no native tool_use blocks, parse Cursor/Claude text markup (<function_calls><invoke ...>)
  let toolCalls = nativeToolCalls;
  let cleanedText = text;

  if (!toolCalls.length) {
    const parsed = parseCallingToolLines(text);
    toolCalls = parsed.tool_calls;
    cleanedText = parsed.cleanedText;
  } else {
    // still strip thinking tags if present
    cleanedText = (cleanedText || "").replace(/<thinking>[\s\S]*?<\/thinking>/g, "").trim();
  }

  const hasToolCalls = toolCalls.length > 0;

  const msg = {
    role: "assistant",
    // IMPORTANT: If tool calls exist, keep the user-facing text (but without tool markup).
    // Cursor can handle both content + tool_calls.
    content: cleanedText || "",
  };

  if (hasToolCalls) msg.tool_calls = toolCalls;

  return {
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
  };
}

function sendOpenAISSE(res, openAIResponse) {
  const choice = openAIResponse?.choices?.[0] || {};
  const message = choice.message || { role: "assistant", content: "" };

  const toolCalls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  const hasToolCalls = toolCalls.length > 0;

  const id = openAIResponse.id || makeChatCmplId();
  const created = openAIResponse.created || Math.floor(Date.now() / 1000);
  const model = openAIResponse.model || "claude-opus-4-5";

  // IMPORTANT: If handleChatCompletions already started SSE, do NOT set headers again.
  if (!res.headersSent) {
    res.status(200);
    res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
    res.setHeader("Cache-Control", "no-cache, no-transform");
    res.setHeader("Connection", "keep-alive");
    res.setHeader("X-Accel-Buffering", "no");
    res.flushHeaders?.();
  }

  const writeChunk = (delta, finish_reason = null) => {
    const chunk = {
      id,
      object: "chat.completion.chunk",
      created,
      model,
      choices: [{ index: 0, delta, finish_reason }],
    };
    res.write(`data: ${JSON.stringify(chunk)}\n\n`);
  };

  // role chunk (safe to send twice; Cursor tolerates it)
  writeChunk({ role: "assistant" }, null);

  // content chunks
  const text = typeof message.content === "string" ? message.content : "";
  if (text.length) {
    const CHUNK = 1500;
    for (let i = 0; i < text.length; i += CHUNK) {
      writeChunk({ content: text.slice(i, i + CHUNK) }, null);
    }
  }

  // tool_calls in incremental OpenAI streaming format (Cursor requires this pattern)
  if (hasToolCalls) {
    const ARG_CHUNK = 1200;

    for (let idx = 0; idx < toolCalls.length; idx++) {
      const tc = toolCalls[idx];
      const name = tc?.function?.name || "unknown_tool";
      const args = tc?.function?.arguments || "";
      const tcId =
        tc?.id || ("call_" + Date.now().toString(36) + "_" + Math.random().toString(36).slice(2, 10));

      // announce tool call id/name
      writeChunk(
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

      // stream arguments in chunks
      for (let i = 0; i < args.length; i += ARG_CHUNK) {
        writeChunk(
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

    writeChunk({}, "tool_calls");
    res.write("data: [DONE]\n\n");
    return res.end();
  }

  writeChunk({}, "stop");
  res.write("data: [DONE]\n\n");
  return res.end();
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
    },
    config: {
      apiKeyConfigured: !!CONFIG.AZURE_API_KEY,
      endpointConfigured: !!CONFIG.AZURE_ENDPOINT,
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
function makeChatCmplId() {
  return "chatcmpl-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 10);
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

function safeParseValue(v) {
  if (typeof v !== "string") return v;
  const s = v.trim();
  if (!s) return s;

  // Try JSON
  if ((s.startsWith("{") && s.endsWith("}")) || (s.startsWith("[") && s.endsWith("]"))) {
    try { return JSON.parse(s); } catch (_) {}
  }

  // Try int
  if (/^-?\d+$/.test(s)) return parseInt(s, 10);

  // Try float
  if (/^-?\d+\.\d+$/.test(s)) return parseFloat(s);

  return s;
}

function extractToolCallsFromText(text, allowedToolNames) {
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

  // 1) Parse Cursor-style XML tool calls:
  // <function_calls>
  //   <invoke name="read_file">
  //     <parameter name="file_path">...</parameter>
  //   </invoke>
  // </function_calls>
  if (cleaned.includes("<invoke") && cleaned.includes("name=")) {
    const invokeRe = /<invoke\s+name="([^"]+)"\s*>([\s\S]*?)<\/invoke>/gi;
    const paramRe = /<parameter\s+name="([^"]+)"\s*>([\s\S]*?)<\/parameter>/gi;

    let m;
    while ((m = invokeRe.exec(cleaned))) {
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

    // Remove function_calls + thinking blocks from visible content (keep any surrounding explanation)
    cleaned = cleaned
      .replace(/<function_calls>[\s\S]*?<\/function_calls>/gi, "")
      .replace(/<thinking>[\s\S]*?<\/thinking>/gi, "")
      .trim();
  }

  // 2) Parse simple "Calling shell_command <cmd>" lines
  // Example: "Calling shell_command cat /path | head -20"
  {
    const callLineRe = /^Calling\s+([A-Za-z0-9_-]+)\s+(.+)$/gmi;
    let cm;
    while ((cm = callLineRe.exec(cleaned))) {
      const toolName = (cm[1] || "").trim();
      const rest = (cm[2] || "").trim();
      if (!isAllowed(toolName)) continue;

      // We only safely auto-parse shell_command here; others are too ambiguous in free text.
      if (toolName === "shell_command") {
        tool_calls.push(mkToolCall("shell_command", { command: rest }));
      }
    }
  }

  return { cleaned, tool_calls };
}

function createSSE(res, model) {
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

  const done = (finish_reason = "stop") => {
    writeChunk({}, finish_reason);
    res.write("data: [DONE]\n\n");
    res.end();
  };

  const error = (message) => {
    // Send an assistant-visible error and terminate the stream cleanly.
    writeChunk({ content: message ? String(message) : "Proxy error" }, null);
    done("stop");
  };

  return { writeChunk, done, error, id, created, model: m };
}

function sseSendToolCalls(sse, toolCalls) {
  // OpenAI streaming expects tool_calls with indices; some clients are picky.
  // We send two deltas: one declares calls + names; second sends arguments.
  const declared = toolCalls.map((tc, idx) => ({
    index: idx,
    id: tc.id,
    type: "function",
    function: { name: tc.function?.name, arguments: "" },
  }));
  sse.writeChunk({ tool_calls: declared }, null);

  const withArgs = toolCalls.map((tc, idx) => ({
    index: idx,
    function: { arguments: tc.function?.arguments || "" },
  }));
  sse.writeChunk({ tool_calls: withArgs }, null);
}

async function handleChatCompletions(req, res) {
  console.log("[REQUEST /chat/completions]", new Date().toISOString());
  console.log("Model:", req.body?.model, "Stream:", req.body?.stream);

  const toolsCount = Array.isArray(req.body?.tools) ? req.body.tools.length : 0;
  console.log("Tools present:", toolsCount);
  console.log("Roles:", Array.isArray(req.body?.messages) ? req.body.messages.map((m) => m.role) : []);

  const requestedModel = req.body?.model || "claude-opus-4-5";
  const wantStream = req.body?.stream === true;
  const allowedToolNames = getToolNameSet(req.body);

  let sse = null;
  let keepalive = null;

  try {
    if (!CONFIG.AZURE_API_KEY) {
      const msg = "Azure API key not configured";
      if (wantStream) {
        sse = createSSE(res, requestedModel);
        sse.error(msg);
        return;
      }
      return res.status(500).json({ error: { message: msg, type: "configuration_error" } });
    }

    if (!CONFIG.AZURE_ENDPOINT) {
      const msg = "Azure endpoint not configured";
      if (wantStream) {
        sse = createSSE(res, requestedModel);
        sse.error(msg);
        return;
      }
      return res.status(500).json({ error: { message: msg, type: "configuration_error" } });
    }

    if (!req.body || !Array.isArray(req.body.messages)) {
      const msg = "Invalid request: expected messages[]";
      if (wantStream) {
        sse = createSSE(res, requestedModel);
        sse.error(msg);
        return;
      }
      return res.status(400).json({ error: { message: msg, type: "invalid_request_error" } });
    }

    // If streaming, start SSE immediately so Cursor doesn't time out.
    if (wantStream) {
      sse = createSSE(res, requestedModel);

      // Initial role chunk (lets Cursor know provider is alive)
      sse.writeChunk({ role: "assistant" }, null);

      // Keepalive chunk every 15s while Azure is working
      keepalive = setInterval(() => {
        try {
          sse.writeChunk({}, null);
        } catch (_) {
          // If client disconnected, this will throw; stop keepalive.
          clearInterval(keepalive);
        }
      }, 15000);
    }

    // Always call Azure non-streaming; we do streaming to Cursor ourselves.
    const reqForAzure = { ...req.body, stream: false };

    // Use your existing OpenAI->Anthropic transform (must keep tools/messages/system handling you already have)
    const anthropicRequest = transformRequest(reqForAzure);
    anthropicRequest.stream = false;

    console.log("[AZURE] Calling Azure Anthropic API (non-streaming)...");
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

    console.log("[AZURE] Response status:", response.status);

    if (keepalive) clearInterval(keepalive);

    if (response.status >= 400) {
      const msg = response.data?.error?.message || response.data?.message || "Azure API error";
      console.error("[ERROR] Azure error:", msg);

      if (wantStream && sse) {
        sse.error(msg);
        return;
      }
      return res.status(response.status).json({ error: { message: msg, type: "api_error", code: response.status } });
    }

    // Use your existing Anthropic->OpenAI transform (should surface tool_calls when Azure returns tool_use blocks)
    const openAIResponse = transformResponse(response.data, requestedModel);

    // Fallback: If no tool_calls, try extracting them from assistant text (<function_calls>… or "Calling shell_command …")
    const choice0 = openAIResponse?.choices?.[0];
    const msg0 = choice0?.message || { role: "assistant", content: "" };

    const toolCalls0 = Array.isArray(msg0.tool_calls) ? msg0.tool_calls : [];
    const hasToolCalls0 = toolCalls0.length > 0;

    if (!hasToolCalls0) {
      const parsed = extractToolCallsFromText(msg0.content || "", allowedToolNames);
      if (parsed.tool_calls.length > 0) {
        msg0.content = parsed.cleaned || "";
        msg0.tool_calls = parsed.tool_calls;
        choice0.finish_reason = "tool_calls";
        openAIResponse.choices[0].message = msg0;
      }
    }

    // Non-stream response
    if (!wantStream) {
      res.setHeader("Content-Type", "application/json; charset=utf-8");
      console.log("[RESPONSE] Sending JSON response");
      return res.status(200).json(openAIResponse);
    }

    // Stream response
    console.log("[RESPONSE] Sending SSE response");

    const finalChoice = openAIResponse.choices?.[0] || {};
    const finalMsg = finalChoice.message || { role: "assistant", content: "" };
    const finalToolCalls = Array.isArray(finalMsg.tool_calls) ? finalMsg.tool_calls : [];
    const hasToolCalls = finalToolCalls.length > 0;

    if (typeof finalMsg.content === "string" && finalMsg.content.length) {
      sse.writeChunk({ content: finalMsg.content }, null);
    }

    if (hasToolCalls) {
      sseSendToolCalls(sse, finalToolCalls);
      sse.done("tool_calls");
      return;
    }

    sse.done("stop");
  } catch (e) {
    if (keepalive) clearInterval(keepalive);

    const errMsg = e?.message || String(e);
    console.error("[ERROR] /chat/completions exception:", errMsg);

    // If we already started SSE, we cannot send JSON headers.
    if (wantStream) {
      try {
        if (!sse) sse = createSSE(res, req.body?.model || "claude-opus-4-5");
        sse.error(errMsg);
      } catch (_) {
        // If client is gone, just end.
        try { res.end(); } catch (_) {}
      }
      return;
    }

    return res.status(500).json({ error: { message: errMsg, type: "proxy_error" } });
  }
}


// Cursor uses this
app.post("/chat/completions", requireAuth, handleChatCompletions);

// Some clients call this
app.post("/v1/chat/completions", requireAuth, handleChatCompletions);

// Optional: keep Anthropic-native passthrough for debugging
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
  console.log("Endpoints:");
  console.log("  GET  /health");
  console.log("  GET  /v1/models");
  console.log("  POST /chat/completions");
  console.log("  POST /v1/chat/completions");
  console.log("=".repeat(80) + "\n");
});
