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
    content: textParts.length ? textParts.join("") : null,
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


// Transform Anthropic response to OpenAI format
function transformResponse(anthropicResponse) {
    const { content, id, model, stop_reason, usage } = anthropicResponse;

    return {
        id: id,
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: [
            {
                index: 0,
                message: {
                    role: "assistant",
                    content: content[0].text,
                },
                finish_reason: stop_reason,
            },
        ],
        usage: {
            prompt_tokens: usage.input_tokens,
            completion_tokens: usage.output_tokens,
            total_tokens: usage.input_tokens + usage.output_tokens,
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

// /chat/completions endpoint (without /v1 prefix) - Cursor uses this!
app.post("/chat/completions", requireAuth, async (req, res) => {
    console.log("[REQUEST /chat/completions]", new Date().toISOString());
    // Only log essential info to avoid Railway rate limit
    console.log("Model:", req.body?.model, "Stream:", req.body?.stream);
    // console.log('Body:', JSON.stringify(req.body, null, 2));
    // console.log('Headers:', JSON.stringify(req.headers, null, 2));

    try {
        if (!CONFIG.AZURE_API_KEY) {
            console.error("[ERROR] Azure API key not configured");
            return res.status(500).json({
                error: {
                    message: "Azure API key not configured",
                    type: "configuration_error",
                },
            });
        }

        if (!CONFIG.AZURE_ENDPOINT) {
            console.error("[ERROR] Azure endpoint not configured");
            return res.status(500).json({
                error: {
                    message: "Azure endpoint not configured",
                    type: "configuration_error",
                },
            });
        }

        // Validate request body
        if (!req.body) {
            console.error("[ERROR] Invalid request body - body is null or undefined");
            return res.status(400).json({
                error: {
                    message: "Invalid request: empty body",
                    type: "invalid_request_error",
                },
            });
        }

        // Check if request has valid content fields
        const hasMessages = req.body.messages && Array.isArray(req.body.messages);
        const hasRoleContent = req.body.role && req.body.content;
        const hasInput = req.body.input && (Array.isArray(req.body.input) || typeof req.body.input === "string");
        const hasContent = req.body.content;

        if (!hasMessages && !hasRoleContent && !hasInput && !hasContent) {
            console.error("[ERROR] Invalid request body - no valid content field found");
            console.error("[ERROR] Request body keys:", Object.keys(req.body));
            return res.status(400).json({
                error: {
                    message: "Invalid request: must include messages, role/content, input, or content field",
                    type: "invalid_request_error",
                },
            });
        }

        console.log("[DEBUG] Request format detected:", {
            hasMessages,
            hasRoleContent,
            hasInput,
            hasContent,
        });

        const isStreaming = req.body.stream === true;
        console.log(`[AZURE] Streaming mode: ${isStreaming}`);

        // Transform request to Anthropic format
        let anthropicRequest;
        try {
            anthropicRequest = transformRequest(req.body);
            console.log("[AZURE] Request transformed successfully");
            console.log("[AZURE] Using model/deployment:", anthropicRequest.model);
            // console.log('[AZURE] Transformed request:', JSON.stringify(anthropicRequest, null, 2));
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

        const response = await axios.post(CONFIG.AZURE_ENDPOINT, anthropicRequest, {
            headers: {
                "Content-Type": "application/json",
                "x-api-key": CONFIG.AZURE_API_KEY,
                "anthropic-version": CONFIG.ANTHROPIC_VERSION,
            },
            timeout: 120000,
            responseType: isStreaming ? "stream" : "json",
            validateStatus: function (status) {
                return status < 600; // Don't throw on any status < 600
            },
        });

        console.log("[AZURE] Response status:", response.status);

        // Handle error responses from Azure
        if (response.status >= 400) {
            console.error("[ERROR] Azure returned error status:", response.status);

            // Try to extract error message from response
            let errorMessage = "Azure API error";
            let errorType = "api_error";

            if (response.data) {
                // If response is a stream, we need to read it
                if (isStreaming && typeof response.data.pipe === "function") {
                    let errorBuffer = "";
                    await new Promise((resolve) => {
                        response.data.on("data", (chunk) => {
                            errorBuffer += chunk.toString();
                        });
                        response.data.on("end", () => {
                            resolve();
                        });
                        response.data.on("error", (err) => {
                            console.error("[ERROR] Error reading error stream:", err);
                            resolve();
                        });
                    });

                    try {
                        const parsed = JSON.parse(errorBuffer);
                        errorMessage = parsed?.error?.message || parsed?.message || errorMessage;
                        errorType = parsed?.error?.type || errorType;
                    } catch (e) {
                        errorMessage = errorBuffer || errorMessage;
                    }
                } else if (typeof response.data === "string") {
                    try {
                        const parsed = JSON.parse(response.data);
                        errorMessage = parsed?.error?.message || parsed?.message || errorMessage;
                        errorType = parsed?.error?.type || errorType;
                    } catch (e) {
                        errorMessage = response.data;
                    }
                } else if (response.data.error) {
                    errorMessage = response.data.error.message || errorMessage;
                    errorType = response.data.error.type || errorType;
                } else if (response.data.message) {
                    errorMessage = response.data.message;
                }
            }

            console.error("[ERROR] Azure error message:", errorMessage);
            console.error("[ERROR] Transformed request that failed:", JSON.stringify(anthropicRequest, null, 2));

            return res.status(response.status).json({
                error: {
                    message: errorMessage,
                    type: errorType,
                    code: response.status,
                },
            });
        }

        if (isStreaming) {
            res.setHeader("Content-Type", "text/event-stream");
            res.setHeader("Cache-Control", "no-cache");
            res.setHeader("Connection", "keep-alive");

            console.log("[AZURE] Streaming response...");

            let buffer = "";

            response.data.on("data", (chunk) => {
                buffer += chunk.toString();
                const lines = buffer.split("\n");
                buffer = lines.pop();

                for (const line of lines) {
                    if (line.startsWith("data: ")) {
                        const data = line.slice(6).trim();
                        if (data === "[DONE]") {
                            res.write("data: [DONE]\n\n");
                            continue;
                        }

                        try {
                            const anthropicEvent = JSON.parse(data);

                            if (anthropicEvent.type === "content_block_delta") {
                                const openaiChunk = {
                                    id: anthropicEvent.id || "chatcmpl-" + Date.now(),
                                    object: "chat.completion.chunk",
                                    created: Math.floor(Date.now() / 1000),
                                    model: req.body.model || "claude-opus-4-5",
                                    choices: [
                                        {
                                            index: 0,
                                            delta: {
                                                content: anthropicEvent.delta?.text || "",
                                            },
                                            finish_reason: null,
                                        },
                                    ],
                                };
                                res.write(`data: ${JSON.stringify(openaiChunk)}\n\n`);
                            } else if (anthropicEvent.type === "message_stop") {
                                const openaiChunk = {
                                    id: "chatcmpl-" + Date.now(),
                                    object: "chat.completion.chunk",
                                    created: Math.floor(Date.now() / 1000),
                                    model: req.body.model || "claude-opus-4-5",
                                    choices: [
                                        {
                                            index: 0,
                                            delta: {},
                                            finish_reason: "stop",
                                        },
                                    ],
                                };
                                res.write(`data: ${JSON.stringify(openaiChunk)}\n\n`);
                                res.write("data: [DONE]\n\n");
                            }
                        } catch (e) {
                            console.error("[ERROR] Failed to parse streaming chunk:", e);
                        }
                    }
                }
            });

            response.data.on("end", () => {
                console.log("[AZURE] Stream ended");
                res.end();
            });

            response.data.on("error", (error) => {
                console.error("[ERROR] Stream error:", error);
                if (!res.headersSent) {
                    res.status(500).json({
                        error: {
                            message: "Stream error: " + error.message,
                            type: "stream_error",
                        },
                    });
                } else {
                    res.end();
                }
            });
        } else {
            console.log("[AZURE] Response received successfully");

            try {
                const openAIResponse = transformResponse(response.data);
                console.log("[RESPONSE] Sending response to client");
                res.json(openAIResponse);
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
        }
    } catch (error) {
        console.error("[ERROR] Exception in /chat/completions:", error.message);
        // console.error('[ERROR] Stack:', error.stack);

        if (error.response) {
            console.error("[ERROR] Azure API error:", error.response.status, error.response.statusText);
            // Don't log full response data - too verbose

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
                error: {
                    message: error.message,
                    type: "proxy_error",
                },
            });
        }
    }
});

// Main proxy endpoint
app.post("/v1/chat/completions", requireAuth, async (req, res) => {
    console.log("[REQUEST]", new Date().toISOString());
    console.log("Body:", JSON.stringify(req.body, null, 2));

    try {
        // Validate API key
        if (!CONFIG.AZURE_API_KEY || CONFIG.AZURE_API_KEY === "YOUR_ACTUAL_API_KEY_HERE") {
            console.error("[ERROR] Azure API key not configured");
            throw new Error("Azure API key not configured. Set AZURE_API_KEY environment variable.");
        }

        const isStreaming = req.body.stream === true;
        console.log(`[AZURE] Streaming mode: ${isStreaming}`);

        // Transform request
        const anthropicRequest = transformRequest(req.body);
        console.log("[AZURE] Calling Azure Anthropic API...");
        console.log("Transformed request:", JSON.stringify(anthropicRequest, null, 2));

        // Call Azure Anthropic API
        const response = await axios.post(CONFIG.AZURE_ENDPOINT, anthropicRequest, {
            headers: {
                "Content-Type": "application/json",
                "x-api-key": CONFIG.AZURE_API_KEY,
                "anthropic-version": CONFIG.ANTHROPIC_VERSION,
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

            let buffer = "";

            response.data.on("data", (chunk) => {
                buffer += chunk.toString();
                const lines = buffer.split("\n");
                buffer = lines.pop(); // Keep incomplete line in buffer

                for (const line of lines) {
                    if (line.startsWith("data: ")) {
                        const data = line.slice(6).trim();
                        if (data === "[DONE]") {
                            res.write("data: [DONE]\n\n");
                            continue;
                        }

                        try {
                            const anthropicEvent = JSON.parse(data);

                            // Transform Anthropic streaming events to OpenAI format
                            if (anthropicEvent.type === "content_block_delta") {
                                const openaiChunk = {
                                    id: anthropicEvent.id || "chatcmpl-" + Date.now(),
                                    object: "chat.completion.chunk",
                                    created: Math.floor(Date.now() / 1000),
                                    model: req.body.model || "claude-opus-4-5",
                                    choices: [
                                        {
                                            index: 0,
                                            delta: {
                                                content: anthropicEvent.delta?.text || "",
                                            },
                                            finish_reason: null,
                                        },
                                    ],
                                };
                                res.write(`data: ${JSON.stringify(openaiChunk)}\n\n`);
                            } else if (anthropicEvent.type === "message_stop") {
                                const openaiChunk = {
                                    id: "chatcmpl-" + Date.now(),
                                    object: "chat.completion.chunk",
                                    created: Math.floor(Date.now() / 1000),
                                    model: req.body.model || "claude-opus-4-5",
                                    choices: [
                                        {
                                            index: 0,
                                            delta: {},
                                            finish_reason: "stop",
                                        },
                                    ],
                                };
                                res.write(`data: ${JSON.stringify(openaiChunk)}\n\n`);
                                res.write("data: [DONE]\n\n");
                            }
                        } catch (e) {
                            console.error("[ERROR] Failed to parse streaming chunk:", e);
                        }
                    }
                }
            });

            response.data.on("end", () => {
                console.log("[AZURE] Stream ended");
                res.end();
            });

            response.data.on("error", (error) => {
                console.error("[ERROR] Stream error:", error);
                res.end();
            });
        } else {
            console.log("[AZURE] Response received successfully");

            // Transform response
            const openAIResponse = transformResponse(response.data);
            console.log("[RESPONSE] Sending response to client");

            res.json(openAIResponse);
        }
    } catch (error) {
        console.error("[ERROR]", error.message);

        if (error.response) {
            console.error("[ERROR] Azure API error:", {
                status: error.response.status,
                data: error.response.data,
            });

            res.status(error.response.status).json({
                error: {
                    message: error.response.data.error?.message || "Azure API error",
                    type: error.response.data.error?.type || "api_error",
                    code: error.response.status,
                },
            });
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
