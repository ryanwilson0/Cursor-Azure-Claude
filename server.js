const express = require("express");
const axios = require("axios");
const app = express();

// Middleware
app.use(express.json({ limit: "50mb" }));

// Configuration - from environment variables
const CONFIG = {
  AZURE_ENDPOINT: process.env.AZURE_ENDPOINT,
  AZURE_API_KEY: process.env.AZURE_API_KEY,
  SERVICE_API_KEY: process.env.SERVICE_API_KEY,
  PORT: process.env.PORT || 8080,
  ANTHROPIC_VERSION: "2023-06-01",
  // Default Azure deployment name (can be overridden via AZURE_DEPLOYMENT_NAME)
  AZURE_DEPLOYMENT_NAME: process.env.AZURE_DEPLOYMENT_NAME || "claude-opus-4-5",
};

// Model name mapping: common model names that should be mapped to Azure deployment
// The actual deployment name is determined by AZURE_DEPLOYMENT_NAME env var
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

// Function to map model name to Azure deployment name
function mapModelToDeployment(modelName) {
  if (!modelName) return CONFIG.AZURE_DEPLOYMENT_NAME;

  if (MODEL_NAMES_TO_MAP.includes(modelName)) {
    return CONFIG.AZURE_DEPLOYMENT_NAME;
  }

  if (process.env.AZURE_DEPLOYMENT_NAME) {
    return CONFIG.AZURE_DEPLOYMENT_NAME;
  }

  return modelName;
}

// CORS middleware
app.use((req, res, next) => {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.header(
    "Access-Control-Allow-Headers",
    "Content-Type, Authorization, x-api-key, anthropic-version"
  );

  if (req.method === "OPTIONS") return res.sendStatus(200);
  next();
});

// Log all requests
app.use((req, res, next) => {
  console.log(`[${req.method}] ${req.path}`);
  next();
});

// Authentication middleware - Validate bearer token from Cursor IDE
function requireAuth(req, res, next) {
  // Skip authentication for OPTIONS requests and health check
  if (req.method === "OPTIONS" || req.path === "/health" || req.path === "/") {
    return next();
  }

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
          "These value of:\n" +
          "\tCursor Settings > Models > API Keys > OpenAI API Key\n\n" +
          "Must match the value of:\n" +
          "\tSERVICE_API_KEY in your .env file\n\n" +
          "Ensure the values match exactly, and try again.\n" +
          "If modifying the .env file, restart the service for the changes to apply.",
        type: "authentication_error",
      },
    });
  }

  let token = authHeader;
  if (authHeader.startsWith("Bearer ")) token = authHeader.substring(7);

  if (token !== CONFIG.SERVICE_API_KEY) {
    console.error("[ERROR] Invalid API key provided");
    return res.status(401).json({
      error: {
        message:
          "Authentication with Cursor-Azure-Claude-Proxy service failed.\n\n" +
          "These value of:\n" +
          "\tCursor Settings > Models > API Keys > OpenAI API Key\n\n" +
          "Must match the value of:\n" +
          "\tSERVICE_API_KEY in your .env file\n\n" +
          "Ensure the values match exactly, and try again.\n" +
          "If modifying the .env file, restart the service for the changes to apply.",
        type: "authentication_error",
      },
    });
  }

  next();
}

// ---------- Tool / content helpers ----------

function toAnthropicContentBlocks(content) {
  // Anthropic "messages" expects content as an array of blocks or a string.
  // Normalize to blocks so tool_result handling is consistent.
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

function openaiToolMessageToAnthropicUserMessage(msg) {
  // OpenAI tool message:
  // { role:"tool", tool_call_id:"...", content:"..." }
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
        content: [{ type: "text", text: resultText }],
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
        id: b.id, // preserve tool_use id
        type: "function",
        function: {
          name: b.name,
          arguments: JSON.stringify(b.input || {}),
        },
      });
    }
  }

  return {
    role: "assistant",
    content: textParts.length ? textParts.join("") : null, // IMPORTANT for tool_calls compatibility
    ...(toolCalls.length ? { tool_calls: toolCalls } : {}),
  };
}

function transformRequest(openAIRequest) {
  const {
    messages,
    model,
    max_tokens,
    temperature,
    stream,
    tools,
    tool_choice, // ignored
    role,
    content,
    input,
    user,
    ...rest
  } = openAIRequest;

  let anthropicMessages = [];
  let systemTextParts = [];

  if (messages && Array.isArray(messages)) {
    for (const msg of messages) {
      if (!msg) continue;

      if (msg.role === "system") {
        if (msg.content != null) {
          systemTextParts.push(
            typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content)
          );
        }
        continue;
      }

      if (msg.role === "tool") {
        anthropicMessages.push(openaiToolMessageToAnthropicUserMessage(msg));
        continue;
      }

      const roleMapped = msg.role === "assistant" ? "assistant" : "user";
      anthropicMessages.push({
        role: roleMapped,
        content: toAnthropicContentBlocks(msg.content),
      });
    }
  } else if (role && content) {
    if (role === "system") systemTextParts.push(String(content));
    else {
      anthropicMessages = [
        {
          role: role === "assistant" ? "assistant" : "user",
          content: toAnthropicContentBlocks(content),
        },
      ];
    }
  } else if (input) {
    if (Array.isArray(input)) {
      for (const msg of input) {
        if (!msg) continue;
        if (msg.role === "system") {
          systemTextParts.push(String(msg.content ?? ""));
          continue;
        }
        const roleMapped = msg.role === "assistant" ? "assistant" : "user";
        anthropicMessages.push({
          role: roleMapped,
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

  if (systemTextParts.length) {
    anthropicRequest.system = systemTextParts.join("\n\n");
  } else if (rest.system !== undefined) {
    anthropicRequest.system = rest.system;
  }

  if (temperature !== undefined) anthropicRequest.temperature = temperature;

  if (stream !== undefined) anthropicRequest.stream = stream;

  const anthTools = openaiToolsToAnthropic(tools);
  if (anthTools.length) {
    anthropicRequest.tools = anthTools;
    // Some Anthropic implementations behave better if tool_choice is explicit
    anthropicRequest.tool_choice = { type: "auto" };
  }

  const supportedFields = ["metadata", "stop_sequences", "top_p", "top_k"];
  for (const field of supportedFields) {
    if (rest[field] !== undefined) anthropicRequest[field] = rest[field];
  }

  return anthropicRequest;
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

/**
 * Parse Cursor/LLM "function_calls" XML-like markup from assistant text and convert to tool_calls.
 * This is a compatibility shim for cases where the model emits tool intent as text instead of tool_use blocks.
 *
 * Example:
 * <function_calls>
 * <invoke name="read_file">
 *   <parameter name="file_path">...</parameter>
 *   <parameter name="start_line">1</parameter>
 * </invoke>
 * </function_calls>
 */
function extractToolCallsFromFunctionCallMarkup(text) {
  if (typeof text !== "string") return [];

  if (!text.includes("<function_calls") || !text.includes("<invoke")) return [];

  const toolCalls = [];
  const invokeRe = /<invoke\s+name="([^"]+)">\s*([\s\S]*?)<\/invoke>/g;
  const paramRe = /<parameter\s+name="([^"]+)">\s*([\s\S]*?)\s*<\/parameter>/g;

  let invokeMatch;
  while ((invokeMatch = invokeRe.exec(text)) !== null) {
    const name = invokeMatch[1];
    const inner = invokeMatch[2] || "";

    const params = {};
    let paramMatch;
    while ((paramMatch = paramRe.exec(inner)) !== null) {
      const key = paramMatch[1];
      let val = (paramMatch[2] || "").trim();

      // Coerce simple numbers when appropriate
      if (/^-?\d+$/.test(val)) val = parseInt(val, 10);

      params[key] = val;
    }

    toolCalls.push({
      id: "call_" + Date.now().toString(36) + "_" + Math.random().toString(36).slice(2, 8),
      type: "function",
      function: {
        name,
        arguments: JSON.stringify(params),
      },
    });
  }

  return toolCalls;
}

function transformResponse(anthropicResponse, requestedModel) {
  const { content, stop_reason, usage } = anthropicResponse;

  const assistantMessage = anthropicContentToOpenAIMessage(content);

  let toolCalls =
    Array.isArray(assistantMessage.tool_calls) && assistantMessage.tool_calls.length
      ? assistantMessage.tool_calls
      : [];

  // Fallback: if no tool_calls but assistant emitted markup, convert it
  if (!toolCalls.length && typeof assistantMessage.content === "string") {
    const extracted = extractToolCallsFromFunctionCallMarkup(assistantMessage.content);
    if (extracted.length) {
      toolCalls = extracted;
      assistantMessage.tool_calls = extracted;
    }
  }

  const hasToolCalls = toolCalls.length > 0;

  // IMPORTANT: tool_calls => content must be null (or omitted). Do not coerce to "".
  if (hasToolCalls) assistantMessage.content = null;
  else if (assistantMessage.content == null) assistantMessage.content = "";

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

// ---------- Root / Health ----------

app.get("/", (req, res) => {
  console.log("[INFO] Root endpoint accessed");
  res.json({
    status: "running",
    name: "Azure Anthropic Proxy for Cursor",
    version: "1.0.0",
    endpoints: {
      health: "/health",
      chat_cursor: "/chat/completions",
      chat_openai: "/v1/chat/completions",
      chat_anthropic: "/v1/messages",
    },
    config: { apiKeyConfigured: !!CONFIG.AZURE_API_KEY },
  });
});

app.get("/health", (req, res) => {
  console.log("[HEALTH] Health check requested");
  res.json({
    status: "ok",
    timestamp: new Date().toISOString(),
    apiKeyConfigured: !!CONFIG.AZURE_API_KEY,
    port: CONFIG.PORT,
  });
});

// ---------- Shared OpenAI chat handler ----------

async function handleChatCompletions(req, res) {
  console.log("[REQUEST /chat/completions]", new Date().toISOString());
  console.log("Model:", req.body?.model, "Stream:", req.body?.stream);

  const toolsCount = Array.isArray(req.body?.tools) ? req.body.tools.length : 0;
  console.log("Tools present:", toolsCount);
  console.log("Roles:", Array.isArray(req.body?.messages) ? req.body.messages.map((m) => m.role) : []);

  try {
    if (!CONFIG.AZURE_API_KEY) {
      console.error("[ERROR] Azure API key not configured");
      return res.status(500).json({
        error: { message: "Azure API key not configured", type: "configuration_error" },
      });
    }

    if (!CONFIG.AZURE_ENDPOINT) {
      console.error("[ERROR] Azure endpoint not configured");
      return res.status(500).json({
        error: { message: "Azure endpoint not configured", type: "configuration_error" },
      });
    }

    if (!req.body) {
      console.error("[ERROR] Empty request body");
      return res.status(400).json({
        error: { message: "Invalid request: empty body", type: "invalid_request_error" },
      });
    }

    const hasMessages = Array.isArray(req.body.messages);
    const hasRoleContent = req.body.role && req.body.content;
    const hasInput =
      req.body.input && (Array.isArray(req.body.input) || typeof req.body.input === "string");
    const hasContent = req.body.content;

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

    // If streaming: open SSE immediately to avoid client timeouts ("Empty provider response")
    let sse = null;
    if (wantStream) {
      const id = makeChatCmplId();
      const created = Math.floor(Date.now() / 1000);
      const model = req.body?.model || "claude-opus-4-5";

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

      // Initial role chunk as soon as possible
      send({ role: "assistant" }, null);

      // Periodic keepalive comment (optional but helps some proxies)
      const keepAlive = setInterval(() => {
        try {
          res.write(": keepalive\n\n");
        } catch (_) {}
      }, 15000);

      sse = { id, created, model, send, keepAlive };
    }

    // Always call Azure non-streaming; we wrap into SSE if Cursor requested streaming
    const reqForAzure = { ...req.body, stream: false };

    let anthropicRequest;
    try {
      anthropicRequest = transformRequest(reqForAzure);
      anthropicRequest.stream = false;
    } catch (e) {
      console.error("[ERROR] transformRequest failed:", e);

      if (sse) {
        clearInterval(sse.keepAlive);
        sse.send({ content: `Proxy error (transform): ${e.message}` }, "stop");
        res.write("data: [DONE]\n\n");
        return res.end();
      }

      return res.status(400).json({
        error: { message: "Failed to transform request: " + e.message, type: "transform_error" },
      });
    }

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

    if (response.status >= 400) {
      const msg = response.data?.error?.message || response.data?.message || "Azure API error";
      console.error("[ERROR] Azure error:", msg);

      if (sse) {
        clearInterval(sse.keepAlive);
        sse.send({ content: `Azure error: ${msg}` }, "stop");
        res.write("data: [DONE]\n\n");
        return res.end();
      }

      return res.status(response.status).json({
        error: { message: msg, type: "api_error", code: response.status },
      });
    }

    // Build OpenAI response JSON
    const openAIResponse = transformResponse(response.data, req.body?.model);
    const choice = openAIResponse?.choices?.[0] || {};
    const message = choice.message || { role: "assistant" };
    const toolCalls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
    const hasToolCalls = toolCalls.length > 0;

    // Ensure OpenAI semantics: tool_calls => content null + finish_reason tool_calls
    if (hasToolCalls) {
      message.content = null;
      choice.finish_reason = "tool_calls";
    } else {
      if (message.content == null) message.content = "";
      if (!choice.finish_reason) choice.finish_reason = "stop";
    }

    // Non-streaming JSON response
    if (!wantStream) {
      res.setHeader("Content-Type", "application/json; charset=utf-8");
      console.log("[RESPONSE] Sending JSON response");
      return res.status(200).json(openAIResponse);
    }

    // Streaming SSE response
    console.log("[RESPONSE] Sending SSE response");
    if (!sse) {
      // Should not happen, but fallback
      res.status(500).json({ error: { message: "Streaming setup failed", type: "proxy_error" } });
      return;
    }

    // stop keepalives once we start sending real chunks
    clearInterval(sse.keepAlive);

    // Tool calls path: do NOT stream content; stream tool_calls as OpenAI-style deltas
    if (hasToolCalls) {
      console.log("Proxy response tool_calls:", toolCalls.map((t) => t.function?.name));

      for (let i = 0; i < toolCalls.length; i++) {
        const tc = toolCalls[i];

        // chunk A: announce tool call with name (arguments empty for now)
        sse.send(
          {
            tool_calls: [
              {
                index: i,
                id: tc.id,
                type: tc.type || "function",
                function: {
                  name: tc.function?.name || "",
                  arguments: "",
                },
              },
            ],
          },
          null
        );

        // chunk B: provide arguments (include id again for stricter parsers)
        sse.send(
          {
            tool_calls: [
              {
                index: i,
                id: tc.id,
                type: tc.type || "function",
                function: {
                  name: tc.function?.name || "",
                  arguments: tc.function?.arguments || "",
                },
              },
            ],
          },
          null
        );
      }

      sse.send({}, "tool_calls");
      res.write("data: [DONE]\n\n");
      return res.end();
    }

    // Normal text path: stream content as one chunk
    if (typeof message.content === "string" && message.content.length) {
      sse.send({ content: message.content }, null);
    }

    sse.send({}, "stop");
    res.write("data: [DONE]\n\n");
    return res.end();
  } catch (error) {
    console.error("[ERROR] Exception in /chat/completions:", error.message);

    // If SSE was started, finish it cleanly
    if (req.body?.stream === true) {
      try {
        res.write(
          `data: ${JSON.stringify({
            id: makeChatCmplId(),
            object: "chat.completion.chunk",
            created: Math.floor(Date.now() / 1000),
            model: req.body?.model || "claude-opus-4-5",
            choices: [{ index: 0, delta: { content: `Proxy error: ${error.message}` }, finish_reason: "stop" }],
          })}\n\n`
        );
        res.write("data: [DONE]\n\n");
        return res.end();
      } catch (_) {}
    }

    return res.status(500).json({ error: { message: error.message, type: "proxy_error" } });
  }
}

// Cursor uses this
app.post("/chat/completions", requireAuth, handleChatCompletions);

// Many clients use this
app.post("/v1/chat/completions", requireAuth, handleChatCompletions);

// ---------- Anthropic-native endpoint for direct compatibility ----------

app.post("/v1/messages", async (req, res) => {
  console.log("[REQUEST /v1/messages]", new Date().toISOString());
  console.log("Body:", JSON.stringify(req.body, null, 2));

  try {
    if (!CONFIG.AZURE_API_KEY) {
      console.error("[ERROR] Azure API key not configured");
      throw new Error("Azure API key not configured");
    }

    const isStreaming = req.body.stream === true;
    console.log(`[AZURE] Calling Azure Anthropic API... (streaming: ${isStreaming})`);

    const response = await axios.post(CONFIG.AZURE_ENDPOINT, req.body, {
      headers: {
        "Content-Type": "application/json",
        "x-api-key": CONFIG.AZURE_API_KEY,
        "anthropic-version": req.headers["anthropic-version"] || CONFIG.ANTHROPIC_VERSION,
      },
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

      console.log("[AZURE] Streaming response...");
      response.data.pipe(res);

      response.data.on("end", () => console.log("[AZURE] Stream ended"));
      response.data.on("error", (error) => {
        console.error("[ERROR] Stream error:", error);
        if (!res.headersSent) {
          res.status(500).json({ error: { message: "Streaming error", type: "stream_error" } });
        }
      });
    } else {
      res.json(response.data);
    }
  } catch (error) {
    console.error("[ERROR]", error.message);

    if (error.response) {
      res.status(error.response.status).json(error.response.data);
    } else if (error.request) {
      res.status(503).json({ error: { message: "Unable to reach Azure Anthropic API", type: "connection_error" } });
    } else {
      res.status(500).json({ error: { message: error.message, type: "proxy_error" } });
    }
  }
});

// Catch-all for any Anthropic API requests
app.all("/anthropic/*", async (req, res) => {
  console.log("[CATCH-ALL /anthropic/*]", req.method, req.path);

  try {
    if (!CONFIG.AZURE_API_KEY) throw new Error("Azure API key not configured");

    const isStreaming = req.body?.stream === true;

    const response = await axios({
      method: req.method,
      url: CONFIG.AZURE_ENDPOINT,
      data: req.body,
      headers: {
        "Content-Type": "application/json",
        "x-api-key": CONFIG.AZURE_API_KEY,
        "anthropic-version": req.headers["anthropic-version"] || CONFIG.ANTHROPIC_VERSION,
      },
      timeout: 120000,
      responseType: isStreaming ? "stream" : "json",
      validateStatus: (s) => s < 600,
    });

    if (response.status >= 400) return res.status(response.status).json(response.data);

    if (isStreaming) {
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");
      response.data.pipe(res);
    } else {
      res.json(response.data);
    }
  } catch (error) {
    console.error("[ERROR /anthropic/*]", error.message);
    res.status(error.response?.status || 500).json({
      error: { message: error.message, type: "proxy_error" },
    });
  }
});

// Catch-all for root /v1/* Anthropic-style requests
app.post("/v1/*", async (req, res) => {
  console.log("[CATCH-ALL /v1/*]", req.path);
  console.log("This request did not match specific handlers, proxying to Azure...");

  try {
    if (!CONFIG.AZURE_API_KEY) throw new Error("Azure API key not configured");

    const isStreaming = req.body?.stream === true;

    const response = await axios.post(CONFIG.AZURE_ENDPOINT, req.body, {
      headers: {
        "Content-Type": "application/json",
        "x-api-key": CONFIG.AZURE_API_KEY,
        "anthropic-version": req.headers["anthropic-version"] || CONFIG.ANTHROPIC_VERSION,
      },
      timeout: 120000,
      responseType: isStreaming ? "stream" : "json",
      validateStatus: (s) => s < 600,
    });

    if (response.status >= 400) return res.status(response.status).json(response.data);

    if (isStreaming) {
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");
      response.data.pipe(res);
    } else {
      res.json(response.data);
    }
  } catch (error) {
    console.error("[ERROR /v1/*]", error.message);
    res.status(error.response?.status || 500).json({
      error: { message: error.message, type: "proxy_error" },
    });
  }
});

// OpenAI-compatible models endpoints (Cursor often calls this to validate provider)
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

// 404 handler
app.use((req, res) => {
  console.log("[404] Route not found:", req.method, req.path);
  res.status(404).json({
    error: {
      message:
        "Endpoint not found. Available endpoints: GET /, GET /health, POST /chat/completions, POST /v1/chat/completions, POST /v1/messages",
      type: "not_found",
    },
  });
});

// Start server
const server = app.listen(CONFIG.PORT, "0.0.0.0", () => {
  console.log("\n" + "=".repeat(80));
  console.log("🚀 Azure Anthropic Proxy - Railway Deployment");
  console.log("=".repeat(80));
  console.log(`📍 Server listening on: 0.0.0.0:${CONFIG.PORT}`);
  console.log(`🔑 API Key configured: ${CONFIG.AZURE_API_KEY ? "✅ Yes" : "❌ No - Set AZURE_API_KEY env var!"}`);
  console.log(`📊 Health check: /health`);
  console.log(`💬 Chat endpoints:`);
  console.log(`   - Cursor: /chat/completions`);
  console.log(`   - OpenAI format: /v1/chat/completions`);
  console.log(`   - Anthropic format: /v1/messages`);
  console.log("=".repeat(80) + "\n");

  if (!CONFIG.AZURE_API_KEY) {
    console.error("⚠️  WARNING: AZURE_API_KEY environment variable is not set!");
    console.error("⚠️  The server will not work until you configure this in Railway settings.\n");
  }
});

// Graceful shutdown
process.on("SIGTERM", () => {
  console.log("\n👋 SIGTERM received. Shutting down gracefully...");
  server.close(() => {
    console.log("✅ Server closed");
    process.exit(0);
  });
});

process.on("SIGINT", () => {
  console.log("\n👋 SIGINT received. Shutting down gracefully...");
  server.close(() => {
    console.log("✅ Server closed");
    process.exit(0);
  });
});
