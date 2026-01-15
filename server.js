const express = require("express");
const axios = require("axios");
const app = express();

// Middleware
app.use(express.json({ limit: "50mb" }));

// Configuration - Lấy từ environment variables
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
const MODEL_NAMES_TO_MAP = ["gpt-4", "gpt-4.1", "gpt-4o", "claude-opus-4-5", "claude-4.5-opus-high", "claude-4-opus", "claude-3-opus", "claude-3-sonnet", "claude-3-haiku"];

// Function to map model name to Azure deployment name
function mapModelToDeployment(modelName) {
    if (!modelName) {
        return CONFIG.AZURE_DEPLOYMENT_NAME;
    }

    // If the model name is in our mapping list, use the configured deployment name
    if (MODEL_NAMES_TO_MAP.includes(modelName)) {
        return CONFIG.AZURE_DEPLOYMENT_NAME;
    }

    // If AZURE_DEPLOYMENT_NAME is explicitly set (not the default), use it for all models
    if (process.env.AZURE_DEPLOYMENT_NAME) {
        return CONFIG.AZURE_DEPLOYMENT_NAME;
    }

    // Otherwise, use the model name as-is (in case it's already the deployment name)
    return modelName;
}

// CORS middleware
app.use((req, res, next) => {
    res.header("Access-Control-Allow-Origin", "*");
    res.header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.header("Access-Control-Allow-Headers", "Content-Type, Authorization, x-api-key, anthropic-version");

    if (req.method === "OPTIONS") {
        return res.sendStatus(200);
    }
    next();
});

// Log all requests
app.use((req, res, next) => {
    console.log(`[${req.method}] ${req.path}`);
    // Only log headers in development or when needed for debugging
    // console.log('Headers:', JSON.stringify(req.headers, null, 2));
    next();
});

// Authentication middleware - Validate bearer token from Cursor IDE
function requireAuth(req, res, next) {
    // Skip authentication for OPTIONS requests and health check
    if (req.method === "OPTIONS" || req.path === "/health" || req.path === "/") {
        return next();
    }

    // Check if SERVICE_API_KEY is configured
    if (!CONFIG.SERVICE_API_KEY) {
        console.error("[ERROR] SERVICE_API_KEY not configured");
        return res.status(500).json({
            error: {
                message: "SERVICE_API_KEY not configured",
                type: "configuration_error",
            },
        });
    }

    // Extract bearer token from Authorization header
    const authHeader = req.headers.authorization;
    if (!authHeader) {
        console.error("[ERROR] Missing Authorization header");
        return res.status(401).json({
            error: {
                message: "Authentication with Cursor-Azure-Claude-Proxy service failed.\n\n" +
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

    // Parse bearer token (format: "Bearer <token>" or just "<token>")
    let token = authHeader;
    if (authHeader.startsWith("Bearer ")) {
        token = authHeader.substring(7);
    }

    // Validate token matches SERVICE_API_KEY
    if (token !== CONFIG.SERVICE_API_KEY) {
        console.error("[ERROR] Invalid API key provided");
        return res.status(401).json({
            error: {
                message: "Authentication with Cursor-Azure-Claude-Proxy service failed.\n\n" +
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

    // Authentication successful
    next();
}
function toAnthropicContentBlocks(content) {
  // Anthropic "messages" expects content as an array of blocks or a string.
  // We normalize to blocks to make tool_result handling consistent.
  if (Array.isArray(content)) return content;
  if (typeof content === "string") return [{ type: "text", text: content }];
  if (content == null) return [];
  // Fallback: stringify objects
  return [{ type: "text", text: String(content) }];
}

function openaiToolsToAnthropic(tools = []) {
  // OpenAI tools: [{type:"function", function:{name, description, parameters}}]
  // Anthropic tools: [{name, description, input_schema}]
  return (tools || [])
    .filter(t => t?.type === "function" && t.function?.name)
    .map(t => ({
      name: t.function.name,
      description: t.function.description || "",
      input_schema: t.function.parameters || { type: "object", properties: {} },
    }));
}

function openaiToolMessageToAnthropicUserMessage(msg) {
  // OpenAI tool message usually looks like:
  // { role:"tool", tool_call_id:"...", content:"..." }
  const toolUseId = msg.tool_call_id || msg.tool_callId || msg.id;
  const resultText =
    typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content);

  if (!toolUseId) {
    // If we can't link to a tool_use id, treat as plain user text fallback.
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
        id: b.id, // IMPORTANT: preserve Anthropic tool_use id
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
    content: textParts.length ? textParts.join("") : "",
  };

  if (toolCalls.length) msg.tool_calls = toolCalls;

  return msg;
}

function transformRequest(openAIRequest) {
  const {
    messages,
    model,
    max_tokens,
    temperature,
    stream,
    tools,          // <-- NEW
    tool_choice,    // <-- optional; ignore if unsupported
    role,
    content,
    input,
    user,
    ...rest
  } = openAIRequest;

  let anthropicMessages = [];
  let systemTextParts = [];

  // Build messages from OpenAI messages[]
  if (messages && Array.isArray(messages)) {
    for (const msg of messages) {
      if (!msg) continue;

      if (msg.role === "system") {
        if (msg.content != null) systemTextParts.push(
          typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content)
        );
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
    else anthropicMessages = [{ role: role === "assistant" ? "assistant" : "user", content: toAnthropicContentBlocks(content) }];
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

  // NOTE: We'll handle streaming in Step 4 below (tools + streaming is trickier)
  if (stream !== undefined) anthropicRequest.stream = stream;

  // Pass through tools (OpenAI -> Anthropic)
  const anthTools = openaiToolsToAnthropic(tools);
  if (anthTools.length) anthropicRequest.tools = anthTools;

  const supportedFields = ["metadata", "stop_sequences", "top_p", "top_k"];
  for (const field of supportedFields) {
    if (rest[field] !== undefined) anthropicRequest[field] = rest[field];
  }

  return anthropicRequest;
}

function mapFinishReason(stopReason) {
  // Anthropic -> OpenAI normalization
  // Common Anthropic stop_reason values: "end_turn", "max_tokens", "tool_use", "stop_sequence"
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

function transformResponse(anthropicResponse, requestedModel) {
  const { content, stop_reason, usage } = anthropicResponse;

  const assistantMessage = anthropicContentToOpenAIMessage(content);

  // Cursor tolerance: avoid null content
  if (assistantMessage.content == null) assistantMessage.content = "";

  const hasToolCalls = Array.isArray(assistantMessage.tool_calls) && assistantMessage.tool_calls.length > 0;

  return {
    id: makeChatCmplId(),
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    // Use the model Cursor requested, not the Azure/Anthropic resolved version string
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



// Root endpoint
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
        config: {
            apiKeyConfigured: !!CONFIG.AZURE_API_KEY
        },
    });
});

// Health check endpoint
app.get("/health", (req, res) => {
    console.log("[HEALTH] Health check requested");
    res.json({
        status: "ok",
        timestamp: new Date().toISOString(),
        apiKeyConfigured: !!CONFIG.AZURE_API_KEY,
        port: CONFIG.PORT,
    });
});

app.post("/chat/completions", requireAuth, async (req, res) => {
  console.log("[REQUEST /chat/completions]", new Date().toISOString());
  console.log("Model:", req.body?.model, "Stream:", req.body?.stream);

  try {
    if (!CONFIG.AZURE_API_KEY) {
      return res.status(500).json({ error: { message: "Azure API key not configured", type: "configuration_error" } });
    }
    if (!CONFIG.AZURE_ENDPOINT) {
      return res.status(500).json({ error: { message: "Azure endpoint not configured", type: "configuration_error" } });
    }
    if (!req.body) {
      return res.status(400).json({ error: { message: "Invalid request: empty body", type: "invalid_request_error" } });
    }

    const hasMessages = req.body.messages && Array.isArray(req.body.messages);
    const hasRoleContent = req.body.role && req.body.content;
    const hasInput = req.body.input && (Array.isArray(req.body.input) || typeof req.body.input === "string");
    const hasContent = req.body.content;

    if (!hasMessages && !hasRoleContent && !hasInput && !hasContent) {
      return res.status(400).json({
        error: {
          message: "Invalid request: must include messages, role/content, input, or content field",
          type: "invalid_request_error",
        },
      });
    }

    const toolsPresent = Array.isArray(req.body?.tools) && req.body.tools.length > 0;

    // Allow streaming ONLY when no tools are present (tool streaming not implemented here)
    const wantStream = req.body.stream === true;
    const isStreaming = wantStream && !toolsPresent;

    // Transform request to Anthropic format
    const anthropicRequest = transformRequest(req.body);
    anthropicRequest.stream = isStreaming;

    if (!isStreaming) {
      // Non-streaming JSON path
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

      if (response.status >= 400) {
        const msg = response.data?.error?.message || response.data?.message || "Azure API error";
        return res.status(response.status).json({ error: { message: msg, type: "api_error", code: response.status } });
      }

      const openAIResponse = transformResponse(response.data, req.body?.model);
      // Cursor tolerance hardening
      const m = openAIResponse?.choices?.[0]?.message;
      if (m && m.content == null) m.content = "";
      if (openAIResponse?.choices?.[0] && m?.tool_calls?.length) openAIResponse.choices[0].finish_reason = "tool_calls";

      res.setHeader("Content-Type", "application/json; charset=utf-8");
      return res.status(200).json(openAIResponse);
    }

    // Streaming SSE path (OpenAI-compatible)
    const response = await axios.post(CONFIG.AZURE_ENDPOINT, anthropicRequest, {
      headers: {
        "Content-Type": "application/json",
        "x-api-key": CONFIG.AZURE_API_KEY,
        "anthropic-version": CONFIG.ANTHROPIC_VERSION,
      },
      timeout: 120000,
      responseType: "stream",
      validateStatus: (s) => s < 600,
    });

    if (response.status >= 400) {
      // Read stream error body (best effort)
      let errorBuf = "";
      await new Promise((resolve) => {
        response.data.on("data", (c) => (errorBuf += c.toString()));
        response.data.on("end", resolve);
        response.data.on("error", resolve);
      });
      return res.status(response.status).json({
        error: {
          message: errorBuf || "Azure API error",
          type: "api_error",
          code: response.status,
        },
      });
    }

    res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
    res.setHeader("Cache-Control", "no-cache, no-transform");
    res.setHeader("Connection", "keep-alive");
    res.flushHeaders?.();

    const requestedModel = req.body?.model || "claude-opus-4-5";
    const openaiId = "chatcmpl-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 10);
    const created = Math.floor(Date.now() / 1000);

    // OpenAI streaming: send role first (many clients expect this)
    let sentRole = false;

    function sendChunk(delta, finish_reason = null) {
      const chunk = {
        id: openaiId,
        object: "chat.completion.chunk",
        created,
        model: requestedModel,
        choices: [
          {
            index: 0,
            delta,
            finish_reason,
          },
        ],
      };
      res.write(`data: ${JSON.stringify(chunk)}\n\n`);
    }

    // initial role chunk
    sendChunk({ role: "assistant" }, null);
    sentRole = true;

    let buffer = "";

    response.data.on("data", (chunk) => {
      buffer += chunk.toString();
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const data = line.slice(6).trim();

        if (!data || data === "[DONE]") continue;

        let evt;
        try {
          evt = JSON.parse(data);
        } catch {
          continue;
        }

        // Anthropic/Azure streaming events
        if (evt.type === "content_block_delta") {
          const text = evt.delta?.text;
          if (typeof text === "string" && text.length) {
            if (!sentRole) {
              sendChunk({ role: "assistant" }, null);
              sentRole = true;
            }
            sendChunk({ content: text }, null);
          }
        }

        if (evt.type === "message_stop") {
          // final chunk
          sendChunk({}, "stop");
          res.write("data: [DONE]\n\n");
          res.end();
        }
      }
    });

    response.data.on("end", () => {
      // If stream ends without message_stop, still close properly
      try {
        sendChunk({}, "stop");
        res.write("data: [DONE]\n\n");
      } catch {}
      res.end();
    });

    response.data.on("error", (err) => {
      console.error("[ERROR] Stream error:", err?.message || err);
      try {
        res.write(`data: ${JSON.stringify({ error: { message: "Stream error", type: "stream_error" } })}\n\n`);
        res.write("data: [DONE]\n\n");
      } catch {}
      res.end();
    });
  } catch (error) {
    console.error("[ERROR] Exception in /chat/completions:", error.message);
    return res.status(500).json({ error: { message: error.message, type: "proxy_error" } });
  }
});


// Main proxy endpoint (OpenAI-style)
app.post("/v1/chat/completions", requireAuth, async (req, res) => {
  console.log("[REQUEST /v1/chat/completions]", new Date().toISOString());
  // Avoid logging full body in production; it can be huge and may contain sensitive data
  // console.log("Body:", JSON.stringify(req.body, null, 2));
  console.log("Model:", req.body?.model, "Stream:", req.body?.stream);

  try {
    // Validate Azure config
    if (!CONFIG.AZURE_API_KEY || CONFIG.AZURE_API_KEY === "YOUR_ACTUAL_API_KEY_HERE") {
      console.error("[ERROR] Azure API key not configured");
      return res.status(500).json({
        error: {
          message: "Azure API key not configured. Set AZURE_API_KEY environment variable.",
          type: "configuration_error",
        },
      });
    }
    if (!CONFIG.AZURE_ENDPOINT) {
      console.error("[ERROR] Azure endpoint not configured");
      return res.status(500).json({
        error: { message: "Azure endpoint not configured", type: "configuration_error" },
      });
    }

    // Force non-streaming (Cursor/provider clients sometimes send stream:true; this proxy
    // does not fully implement OpenAI-compatible streaming + tool deltas)
    req.body.stream = false;
    const isStreaming = false;

    // Transform request
    let anthropicRequest;
    try {
      anthropicRequest = transformRequest(req.body);
      anthropicRequest.stream = false;
    } catch (transformError) {
      console.error("[ERROR] Failed to transform request:", transformError);
      return res.status(400).json({
        error: {
          message: "Failed to transform request: " + transformError.message,
          type: "transform_error",
        },
      });
    }

    console.log("[AZURE] Calling Azure Anthropic API...");
    // console.log("Transformed request:", JSON.stringify(anthropicRequest, null, 2));

    // Call Azure Anthropic API (always JSON)
    const response = await axios.post(CONFIG.AZURE_ENDPOINT, anthropicRequest, {
      headers: {
        "Content-Type": "application/json",
        "x-api-key": CONFIG.AZURE_API_KEY,
        "anthropic-version": CONFIG.ANTHROPIC_VERSION,
      },
      timeout: 120000,
      responseType: "json",
      validateStatus: (status) => status < 600,
    });

    console.log("[AZURE] Response status:", response.status);

    // Handle Azure error responses
    if (response.status >= 400) {
      let errorMessage = "Azure API error";
      let errorType = "api_error";

      if (response.data) {
        if (typeof response.data === "string") {
          try {
            const parsed = JSON.parse(response.data);
            errorMessage = parsed?.error?.message || parsed?.message || errorMessage;
            errorType = parsed?.error?.type || errorType;
          } catch {
            errorMessage = response.data;
          }
        } else if (response.data.error) {
          errorMessage = response.data.error.message || errorMessage;
          errorType = response.data.error.type || errorType;
        } else if (response.data.message) {
          errorMessage = response.data.message;
        }
      }

      console.error("[ERROR] Azure returned error:", response.status, errorMessage);
      console.error("[ERROR] Transformed request that failed:", JSON.stringify(anthropicRequest, null, 2));

      return res.status(response.status).json({
        error: { message: errorMessage, type: errorType, code: response.status },
      });
    }

    // Transform response to OpenAI chat.completion JSON
    try {
      const openAIResponse = transformResponse(response.data, req.body?.model);
      res.setHeader("Content-Type", "application/json; charset=utf-8");

      // Cursor/provider tolerance hardening
      const msg = openAIResponse?.choices?.[0]?.message;
      if (msg && msg.content == null) msg.content = "";

      if (openAIResponse?.choices?.[0] && msg?.tool_calls?.length) {
        openAIResponse.choices[0].finish_reason = "tool_calls";
      }

      console.log("[RESPONSE] Sending response to client");
      return res.json(openAIResponse);
    } catch (transformError) {
      console.error("[ERROR] Failed to transform response:", transformError);
      console.error("[ERROR] Raw response data:", JSON.stringify(response.data));
      return res.status(500).json({
        error: {
          message: "Failed to transform response: " + transformError.message,
          type: "transform_error",
        },
      });
    }
  } catch (error) {
    console.error("[ERROR] Exception in /v1/chat/completions:", error.message);

    if (error.response) {
      console.error("[ERROR] Azure API error:", error.response.status, error.response.statusText);
      return res.status(error.response.status).json({
        error: {
          message: error.response.data?.error?.message || error.message,
          type: error.response.data?.error?.type || "api_error",
          code: error.response.status,
        },
      });
    } else if (error.request) {
      console.error("[ERROR] No response from Azure API");
      return res.status(503).json({
        error: {
          message: "Unable to reach Azure Anthropic API: " + error.message,
          type: "connection_error",
        },
      });
    } else {
      console.error("[ERROR] Request setup error:", error.message);
      return res.status(500).json({
        error: { message: error.message, type: "proxy_error" },
      });
    }
  }
});


// Anthropic-native endpoint for direct compatibility
app.post("/v1/messages", async (req, res) => {
    console.log("[REQUEST /v1/messages]", new Date().toISOString());
    console.log("Body:", JSON.stringify(req.body, null, 2));

    try {
        // Validate API key
        if (!CONFIG.AZURE_API_KEY) {
            console.error("[ERROR] Azure API key not configured");
            throw new Error("Azure API key not configured");
        }

        const isStreaming = req.body.stream === true;
        console.log(`[AZURE] Calling Azure Anthropic API... (streaming: ${isStreaming})`);

        // Call Azure Anthropic API
        const response = await axios.post(CONFIG.AZURE_ENDPOINT, req.body, {
            headers: {
                "Content-Type": "application/json",
                "x-api-key": CONFIG.AZURE_API_KEY,
                "anthropic-version": req.headers["anthropic-version"] || CONFIG.ANTHROPIC_VERSION,
            },
            timeout: 120000,
            responseType: isStreaming ? "stream" : "json",
        });

        if (isStreaming) {
            // Set headers for SSE streaming
            res.setHeader("Content-Type", "text/event-stream");
            res.setHeader("Cache-Control", "no-cache");
            res.setHeader("Connection", "keep-alive");

            console.log("[AZURE] Streaming response...");

            // Pipe the stream directly to response
            response.data.pipe(res);

            response.data.on("end", () => {
                console.log("[AZURE] Stream ended");
            });

            response.data.on("error", (error) => {
                console.error("[ERROR] Stream error:", error);
                if (!res.headersSent) {
                    res.status(500).json({
                        error: {
                            message: "Streaming error",
                            type: "stream_error",
                        },
                    });
                }
            });
        } else {
            console.log("[AZURE] Response received successfully");
            // Return Anthropic response directly
            res.json(response.data);
        }
    } catch (error) {
        console.error("[ERROR]", error.message);

        if (error.response) {
            console.error("[ERROR] Azure API error:", {
                status: error.response.status,
                data: error.response.data,
            });

            res.status(error.response.status).json(error.response.data);
        } else if (error.request) {
            console.error("[ERROR] No response from Azure API");
            res.status(503).json({
                error: {
                    message: "Unable to reach Azure Anthropic API",
                    type: "connection_error",
                },
            });
        } else {
            console.error("[ERROR] Request setup error");
            res.status(500).json({
                error: {
                    message: error.message,
                    type: "proxy_error",
                },
            });
        }
    }
});

// Catch-all for any Anthropic API requests
app.all("/anthropic/*", async (req, res) => {
    console.log("[CATCH-ALL /anthropic/*]", req.method, req.path);

    try {
        if (!CONFIG.AZURE_API_KEY) {
            throw new Error("Azure API key not configured");
        }

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
        });

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
            error: {
                message: error.message,
                type: "proxy_error",
            },
        });
    }
});

// Catch-all for root /v1/* Anthropic-style requests
app.post("/v1/*", async (req, res) => {
    console.log("[CATCH-ALL /v1/*]", req.path);
    console.log("This request did not match specific handlers, proxying to Azure...");

    try {
        if (!CONFIG.AZURE_API_KEY) {
            throw new Error("Azure API key not configured");
        }

        const isStreaming = req.body?.stream === true;

        const response = await axios.post(CONFIG.AZURE_ENDPOINT, req.body, {
            headers: {
                "Content-Type": "application/json",
                "x-api-key": CONFIG.AZURE_API_KEY,
                "anthropic-version": req.headers["anthropic-version"] || CONFIG.ANTHROPIC_VERSION,
            },
            timeout: 120000,
            responseType: isStreaming ? "stream" : "json",
        });

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
            error: {
                message: error.message,
                type: "proxy_error",
            },
        });
    }
});
// OpenAI-compatible models endpoints (Cursor often calls this to validate provider)
app.get("/v1/models", requireAuth, (req, res) => {
  const now = Math.floor(Date.now() / 1000);
  res.json({
    object: "list",
    data: [
      {
        id: "claude-opus-4-5",
        object: "model",
        created: now,
        owned_by: "proxy",
      },
    ],
  });
});

app.get("/models", requireAuth, (req, res) => {
  const now = Math.floor(Date.now() / 1000);
  res.json({
    object: "list",
    data: [
      {
        id: "claude-opus-4-5",
        object: "model",
        created: now,
        owned_by: "proxy",
      },
    ],
  });
});

// 404 handler
app.use((req, res) => {
    console.log("[404] Route not found:", req.method, req.path);
    res.status(404).json({
        error: {
            message: "Endpoint not found. Available endpoints: GET /, GET /health, POST /chat/completions, POST /v1/chat/completions, POST /v1/messages",
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
