const express = require("express");
const axios = require("axios");

const app = express();
app.use(express.json({ limit: "50mb" }));

/**
 * CONFIG
 */
const CONFIG = {
  AZURE_ENDPOINT: process.env.AZURE_ENDPOINT,          // e.g. https://<resource>.openai.azure.com/anthropic/v1/messages OR https://<project>.services.ai.azure.com/anthropic/v1/messages
  AZURE_API_KEY: process.env.AZURE_API_KEY,
  SERVICE_API_KEY: process.env.SERVICE_API_KEY,
  PORT: process.env.PORT || 8080,

  // Anthropic API version header (still required)
  ANTHROPIC_VERSION: process.env.ANTHROPIC_VERSION || "2023-06-01",

  // This is the key one for tool use in many Claude environments
  // (only sent when tools are present)
  ANTHROPIC_BETA_TOOLS: process.env.ANTHROPIC_BETA_TOOLS || "tools-2024-04-04",

  // Default Azure deployment name (can be overridden)
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

/**
 * CORS
 */
app.use((req, res, next) => {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.header("Access-Control-Allow-Headers", "Content-Type, Authorization, x-api-key, anthropic-version, anthropic-beta");
  if (req.method === "OPTIONS") return res.sendStatus(200);
  next();
});

/**
 * Logging
 */
app.use((req, res, next) => {
  console.log(`[${req.method}] ${req.path}`);
  next();
});

/**
 * Auth middleware
 */
function requireAuth(req, res, next) {
  if (req.method === "OPTIONS" || req.path === "/health" || req.path === "/") return next();

  if (!CONFIG.SERVICE_API_KEY) {
    console.error("[ERROR] SERVICE_API_KEY not configured");
    return res.status(500).json({
      error: { message: "SERVICE_API_KEY not configured", type: "configuration_error" },
    });
  }

  const authHeader = req.headers.authorization;
  if (!authHeader) {
    console.error("[ERROR] Missing Authorization header");
    return res.status(401).json({
      error: {
        message:
          "Authentication with Cursor-Azure-Claude-Proxy service failed.\n\n" +
          "Cursor Settings > Models > API Keys > OpenAI API Key\n" +
          "must match SERVICE_API_KEY in your Railway env.\n",
        type: "authentication_error",
      },
    });
  }

  let token = authHeader;
  if (authHeader.startsWith("Bearer ")) token = authHeader.slice(7);

  if (token !== CONFIG.SERVICE_API_KEY) {
    console.error("[ERROR] Invalid API key provided");
    return res.status(401).json({
      error: {
        message:
          "Authentication with Cursor-Azure-Claude-Proxy service failed.\n\n" +
          "Cursor Settings > Models > API Keys > OpenAI API Key\n" +
          "must match SERVICE_API_KEY in your Railway env.\n",
        type: "authentication_error",
      },
    });
  }

  next();
}

/**
 * Helpers: OpenAI <-> Anthropic
 */
function toAnthropicContentBlocks(content) {
  // Normalize to Anthropic content blocks (array)
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

function openaiAssistantToAnthropicAssistantMessage(msg) {
  // IMPORTANT:
  // Cursor will send assistant messages that include tool_calls.
  // Anthropic expects those as "tool_use" blocks in assistant content.
  const blocks = [];

  // Include assistant text if present
  const contentBlocks = toAnthropicContentBlocks(msg.content);
  for (const b of contentBlocks) {
    if (b && (b.type === "text" || b.type)) blocks.push(b);
  }

  // Convert OpenAI tool_calls -> Anthropic tool_use blocks
  if (Array.isArray(msg.tool_calls)) {
    for (const tc of msg.tool_calls) {
      if (!tc?.id || !tc?.function?.name) continue;

      let inputObj = {};
      const args = tc.function.arguments;
      if (typeof args === "string" && args.trim().length) {
        try {
          inputObj = JSON.parse(args);
        } catch {
          // If args isn't valid JSON, pass raw string in a wrapper
          inputObj = { _raw: args };
        }
      }

      blocks.push({
        type: "tool_use",
        id: tc.id,
        name: tc.function.name,
        input: inputObj,
      });
    }
  }

  return { role: "assistant", content: blocks };
}

function openaiToolMessageToAnthropicUserMessage(msg) {
  // OpenAI tool message: { role:"tool", tool_call_id:"...", content:"..." }
  const toolUseId = msg.tool_call_id || msg.tool_callId || msg.id;

  const resultText =
    typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content);

  if (!toolUseId) {
    return { role: "user", content: toAnthropicContentBlocks(resultText) };
  }

  return {
    role: "user",
    content: [
      {
        type: "tool_result",
        tool_use_id: toolUseId,
        // content can be string or blocks; string is safest across implementations
        content: resultText,
      },
    ],
  };
}

function anthropicContentToOpenAIMessage(contentBlocks) {
  const textParts = [];
  const toolCalls = [];

  for (const b of contentBlocks || []) {
    if (b?.type === "text") {
      if (typeof b.text === "string") textParts.push(b.text);
    } else if (b?.type === "tool_use") {
      toolCalls.push({
        id: b.id, // preserve tool_use id so tool_result can refer to it
        type: "function",
        function: {
          name: b.name,
          arguments: JSON.stringify(b.input || {}),
        },
      });
    }
  }

  // If there are tool calls, content is typically null in OpenAI format
  const msg = {
    role: "assistant",
    content: toolCalls.length ? null : (textParts.length ? textParts.join("") : ""),
  };

  if (toolCalls.length) msg.tool_calls = toolCalls;

  return msg;
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

function makeChatCmplId() {
  return "chatcmpl-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 10);
}

function transformRequest(openAIRequest) {
  const {
    messages,
    model,
    max_tokens,
    temperature,
    stream,
    tools,
    role,
    content,
    input,
    user,
    ...rest
  } = openAIRequest;

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
        anthropicMessages.push(openaiToolMessageToAnthropicUserMessage(msg));
        continue;
      }

      if (msg.role === "assistant") {
        anthropicMessages.push(openaiAssistantToAnthropicAssistantMessage(msg));
        continue;
      }

      // user or anything else defaults to user
      anthropicMessages.push({
        role: "user",
        content: toAnthropicContentBlocks(msg.content),
      });
    }
  } else if (role && content != null) {
    if (role === "system") systemTextParts.push(String(content));
    else {
      anthropicMessages = [
        { role: role === "assistant" ? "assistant" : "user", content: toAnthropicContentBlocks(content) },
      ];
    }
  } else if (input != null) {
    if (Array.isArray(input)) {
      for (const msg of input) {
        if (!msg) continue;
        if (msg.role === "system") {
          systemTextParts.push(String(msg.content ?? ""));
          continue;
        }
        if (msg.role === "tool") {
          anthropicMessages.push(openaiToolMessageToAnthropicUserMessage(msg));
          continue;
        }
        if (msg.role === "assistant") {
          anthropicMessages.push(openaiAssistantToAnthropicAssistantMessage(msg));
          continue;
        }
        anthropicMessages.push({
          role: "user",
          content: toAnthropicContentBlocks(msg.content),
        });
      }
    } else {
      anthropicMessages = [{ role: user || "user", content: toAnthropicContentBlocks(input) }];
    }
  } else if (content != null) {
    anthropicMessages = [{ role: "user", content: toAnthropicContentBlocks(content) }];
  } else {
    throw new Error("Invalid request format: missing messages, role/content, input, or content field");
  }

  if (!anthropicMessages.length) throw new Error("Invalid request: no valid messages found");

  const azureModelName = mapModelToDeployment(model);

  const anthropicRequest = {
    model: azureModelName,
    messages: anthropicMessages,
    max_tokens: max_tokens || 4096,
  };

  // System prompt aggregation
  if (systemTextParts.length) {
    anthropicRequest.system = systemTextParts.join("\n\n");
  } else if (rest.system !== undefined) {
    anthropicRequest.system = rest.system;
  }

  // Temperature
  if (temperature !== undefined) anthropicRequest.temperature = temperature;

  // We deliberately control streaming at the route level.
  // But keep stream field if caller provided it (won't be used for Azure request here).
  if (stream !== undefined) anthropicRequest.stream = stream;

  // Tools pass-through
  const anthTools = openaiToolsToAnthropic(tools);
  if (anthTools.length) {
    anthropicRequest.tools = anthTools;

    // Strong hint to stop “pretend tool calling”
    const toolHint =
      "When you need to use a tool, do NOT write 'Calling <tool>' in plain text. " +
      "Use proper tool_use blocks only, with valid JSON inputs.";
    if (typeof anthropicRequest.system === "string") anthropicRequest.system += "\n\n" + toolHint;
    else if (!anthropicRequest.system) anthropicRequest.system = toolHint;
  }

  // Supported optional fields
  const supportedFields = ["metadata", "stop_sequences", "top_p", "top_k"];
  for (const field of supportedFields) {
    if (rest[field] !== undefined) anthropicRequest[field] = rest[field];
  }

  return anthropicRequest;
}

function transformResponse(anthropicResponse, requestedModel) {
  const { content, stop_reason, usage } = anthropicResponse;

  const assistantMessage = anthropicContentToOpenAIMessage(content);

  const hasToolCalls = Array.isArray(assistantMessage.tool_calls) && assistantMessage.tool_calls.length > 0;

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

/**
 * Basic endpoints
 */
app.get("/", (req, res) => {
  res.json({
    status: "running",
    name: "Azure Anthropic Proxy for Cursor",
    endpoints: {
      health: "/health",
      chat_cursor: "/chat/completions",
      chat_openai: "/v1/chat/completions",
      models: "/v1/models",
      anthropic_messages: "/v1/messages",
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

/**
 * OpenAI-compatible model listing (Cursor often calls this)
 */
app.get("/v1/models", requireAuth, (req, res) => {
  const now = Math.floor(Date.now() / 1000);
  res.json({
    object: "list",
    data: [
      { id: "claude-opus-4-5", object: "model", created: now, owned_by: "proxy" },
    ],
  });
});

app.get("/models", requireAuth, (req, res) => {
  const now = Math.floor(Date.now() / 1000);
  res.json({
    object: "list",
    data: [
      { id: "claude-opus-4-5", object: "model", created: now, owned_by: "proxy" },
    ],
  });
});

/**
 * Core handler used by BOTH /chat/completions and /v1/chat/completions
 *
 * Key behavior:
 * - Always call Azure non-streaming (more reliable).
 * - If Cursor asked stream:true, return a minimal OpenAI SSE stream (fake streaming).
 */
async function handleChatCompletions(req, res) {
  console.log("[REQUEST /chat/completions]", new Date().toISOString());
  console.log("Model:", req.body?.model, "Stream:", req.body?.stream);

  const toolsCount = Array.isArray(req.body?.tools) ? req.body.tools.length : 0;
  console.log("Tools present:", toolsCount);
  console.log("Roles:", Array.isArray(req.body?.messages) ? req.body.messages.map((m) => m.role) : []);

  if (!CONFIG.AZURE_API_KEY) {
    console.error("[ERROR] Azure API key not configured");
    return res.status(500).json({ error: { message: "Azure API key not configured", type: "configuration_error" } });
  }
  if (!CONFIG.AZURE_ENDPOINT) {
    console.error("[ERROR] Azure endpoint not configured");
    return res.status(500).json({ error: { message: "Azure endpoint not configured", type: "configuration_error" } });
  }
  if (!req.body) {
    console.error("[ERROR] Empty request body");
    return res.status(400).json({ error: { message: "Invalid request: empty body", type: "invalid_request_error" } });
  }

  const hasMessages = Array.isArray(req.body.messages);
  const hasRoleContent = req.body.role && req.body.content != null;
  const hasInput = req.body.input && (Array.isArray(req.body.input) || typeof req.body.input === "string");
  const hasContent = req.body.content != null;

  if (!hasMessages && !hasRoleContent && !hasInput && !hasContent) {
    console.error("[ERROR] Invalid request body keys:", Object.keys(req.body));
    return res.status(400).json({
      error: {
        message: "Invalid request: must include messages, role/content, input, or content field",
        type: "invalid_request_error",
      },
    });
  }

  const wantStream = req.body.stream === true;

  // Always call Azure with stream:false
  const reqForAzure = { ...req.body, stream: false };

  let anthropicRequest;
  try {
    anthropicRequest = transformRequest(reqForAzure);
    anthropicRequest.stream = false;
  } catch (e) {
    console.error("[ERROR] transformRequest failed:", e);
    return res.status(400).json({ error: { message: "Failed to transform request: " + e.message, type: "transform_error" } });
  }

  // Build Azure headers
  const azureHeaders = {
    "Content-Type": "application/json",
    "x-api-key": CONFIG.AZURE_API_KEY,
    "anthropic-version": CONFIG.ANTHROPIC_VERSION,
  };

  // If tools are present, enable tool use via beta header
  if (toolsCount > 0) {
    azureHeaders["anthropic-beta"] = CONFIG.ANTHROPIC_BETA_TOOLS;
  }

  console.log("[AZURE] Calling Azure Anthropic API (non-streaming)...");
  const response = await axios.post(CONFIG.AZURE_ENDPOINT, anthropicRequest, {
    headers: azureHeaders,
    timeout: 120000,
    responseType: "json",
    validateStatus: (s) => s < 600,
  });

  console.log("[AZURE] Response status:", response.status);

  if (response.status >= 400) {
    const msg = response.data?.error?.message || response.data?.message || "Azure API error";
    console.error("[ERROR] Azure error:", msg);
    return res.status(response.status).json({ error: { message: msg, type: "api_error", code: response.status } });
  }

  const openAIResponse = transformResponse(response.data, req.body?.model);

  const choice = openAIResponse?.choices?.[0] || {};
  const message = choice.message || { role: "assistant", content: "" };

  const toolCalls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  const hasToolCalls = toolCalls.length > 0;

  // If Cursor did NOT request streaming, return JSON
  if (!wantStream) {
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    console.log("[RESPONSE] Sending JSON response");
    return res.status(200).json(openAIResponse);
  }

  // Otherwise return SSE (OpenAI streaming compatible enough for Cursor)
  console.log("[RESPONSE] Sending SSE response");

  const id = openAIResponse.id || makeChatCmplId();
  const created = openAIResponse.created || Math.floor(Date.now() / 1000);
  const model = openAIResponse.model || (req.body?.model || "claude-opus-4-5");

  res.status(200);
  res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders?.();

  const send = (delta, finish_reason = null) => {
    const chunk = {
      id,
      object: "chat.completion.chunk",
      created,
      model,
      choices: [{ index: 0, delta, finish_reason }],
    };
    res.write(`data: ${JSON.stringify(chunk)}\n\n`);
  };

  // 1) initial role chunk
  send({ role: "assistant" }, null);

  // 2) If tool calls exist, stream ONLY tool_calls (no content)
  if (hasToolCalls) {
    const streamedToolCalls = toolCalls.map((tc, idx) => ({
      index: idx,
      id: tc.id,
      type: tc.type || "function",
      function: {
        name: tc.function?.name,
        // In OpenAI streaming this can be chunked; sending full JSON string is acceptable for many clients.
        arguments: tc.function?.arguments || "",
      },
    }));

    send({ tool_calls: streamedToolCalls }, null);

    // Finish as tool_calls so Cursor executes them
    send({}, "tool_calls");
    res.write("data: [DONE]\n\n");
    return res.end();
  }

  // 3) Otherwise stream assistant text (if any)
  if (typeof message.content === "string" && message.content.length) {
    send({ content: message.content }, null);
  }

  // 4) normal stop
  send({}, "stop");
  res.write("data: [DONE]\n\n");
  return res.end();
}

/**
 * Cursor commonly hits either of these depending on configuration.
 */
app.post("/chat/completions", requireAuth, async (req, res) => {
  try {
    await handleChatCompletions(req, res);
  } catch (err) {
    console.error("[ERROR] /chat/completions:", err?.message || err);
    if (!res.headersSent) {
      return res.status(500).json({ error: { message: err?.message || "proxy_error", type: "proxy_error" } });
    }
    return res.end();
  }
});

app.post("/v1/chat/completions", requireAuth, async (req, res) => {
  try {
    await handleChatCompletions(req, res);
  } catch (err) {
    console.error("[ERROR] /v1/chat/completions:", err?.message || err);
    if (!res.headersSent) {
      return res.status(500).json({ error: { message: err?.message || "proxy_error", type: "proxy_error" } });
    }
    return res.end();
  }
});

/**
 * Anthropic-native endpoint (optional / for debugging)
 */
app.post("/v1/messages", async (req, res) => {
  try {
    if (!CONFIG.AZURE_API_KEY) return res.status(500).json({ error: { message: "Azure API key not configured" } });
    if (!CONFIG.AZURE_ENDPOINT) return res.status(500).json({ error: { message: "Azure endpoint not configured" } });

    const isStreaming = req.body?.stream === true;

    const headers = {
      "Content-Type": "application/json",
      "x-api-key": CONFIG.AZURE_API_KEY,
      "anthropic-version": req.headers["anthropic-version"] || CONFIG.ANTHROPIC_VERSION,
    };

    // Allow caller to pass anthropic-beta through (useful for tool debugging)
    if (req.headers["anthropic-beta"]) headers["anthropic-beta"] = req.headers["anthropic-beta"];

    const response = await axios.post(CONFIG.AZURE_ENDPOINT, req.body, {
      headers,
      timeout: 120000,
      responseType: isStreaming ? "stream" : "json",
      validateStatus: (s) => s < 600,
    });

    if (response.status >= 400) {
      return res.status(response.status).json(response.data);
    }

    if (isStreaming) {
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");
      response.data.pipe(res);
      return;
    }

    res.json(response.data);
  } catch (e) {
    console.error("[ERROR] /v1/messages:", e?.message || e);
    res.status(500).json({ error: { message: e?.message || "proxy_error" } });
  }
});

/**
 * 404
 */
app.use((req, res) => {
  res.status(404).json({
    error: {
      message:
        "Endpoint not found. Available endpoints: " +
        "GET /, GET /health, GET /v1/models, POST /chat/completions, POST /v1/chat/completions, POST /v1/messages",
      type: "not_found",
    },
  });
});

/**
 * Start
 */
const server = app.listen(CONFIG.PORT, "0.0.0.0", () => {
  console.log("=".repeat(80));
  console.log("Azure Anthropic Proxy - Server started");
  console.log(`Listening on 0.0.0.0:${CONFIG.PORT}`);
  console.log(`AZURE_ENDPOINT set: ${!!CONFIG.AZURE_ENDPOINT}`);
  console.log(`AZURE_API_KEY set: ${!!CONFIG.AZURE_API_KEY}`);
  console.log(`SERVICE_API_KEY set: ${!!CONFIG.SERVICE_API_KEY}`);
  console.log(`ANTHROPIC_VERSION: ${CONFIG.ANTHROPIC_VERSION}`);
  console.log(`ANTHROPIC_BETA_TOOLS: ${CONFIG.ANTHROPIC_BETA_TOOLS}`);
  console.log("=".repeat(80));
});

process.on("SIGTERM", () => {
  server.close(() => process.exit(0));
});
process.on("SIGINT", () => {
  server.close(() => process.exit(0));
});
