import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime as dt
import logging
from typing import Literal, Optional

import discord
import httpx
from openai import AsyncOpenAI
import yaml
import json
import subprocess
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

VISION_MODEL_TAGS = ("gpt-4", "claude-3", "gemini", "pixtral", "llava", "vision", "vl")
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")

ALLOWED_FILE_TYPES = ("image", "text")

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1

MAX_MESSAGE_NODES = 100


def get_config(filename="config.yaml"):
    with open(filename, "r") as file:
        return yaml.safe_load(file)


cfg = get_config()

if client_id := cfg["client_id"]:
    logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/api/oauth2/authorize?client_id={client_id}&permissions=412317273088&scope=bot\n")

intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(cfg["status_message"] or "github.com/jakobdylanc/llmcord")[:128])
discord_client = discord.Client(intents=intents, activity=activity)

httpx_client = httpx.AsyncClient()

msg_nodes = {}
last_task_time = 0
mcp_servers = {}  # Store MCP server processes


@dataclass
class MsgNode:
    text: Optional[str] = None
    images: list = field(default_factory=list)

    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None

    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False

    parent_msg: Optional[discord.Message] = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


async def start_mcp_server(name, server_config):
    command = server_config.get('command')
    args = server_config.get('args', [])
    env = {**os.environ, **server_config.get('env', {})}
    
    try:
        # Log la commande exacte pour le débogage
        cmd_str = f"{command} {' '.join(args)}"
        logging.info(f"Starting MCP server: {name} with command: {cmd_str}")
        logging.info(f"Environment variables: JIRA_URL={env.get('JIRA_URL')}, JIRA_USERNAME={env.get('JIRA_USERNAME')}")
        
        process = await asyncio.create_subprocess_exec(
            command, *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )
        
        # Lecture des erreurs potentielles au démarrage
        stderr_data = await asyncio.wait_for(process.stderr.readline(), timeout=2.0)
        if stderr_data:
            logging.warning(f"MCP server {name} stderr: {stderr_data.decode().strip()}")
        
        mcp_servers[name] = process
        logging.info(f"Started MCP server: {name} (pid: {process.pid})")
        
        # Vérifier si le serveur est prêt avec une requête ping simple
        ping_request = {
            "jsonrpc": "2.0",
            "method": "ping",
            "id": 0
        }
        process.stdin.write((json.dumps(ping_request) + "\n").encode())
        await process.stdin.drain()
        
        try:
            ping_response = await asyncio.wait_for(process.stdout.readline(), timeout=5.0)
            logging.info(f"MCP server {name} ping response: {ping_response.decode().strip()}")
        except asyncio.TimeoutError:
            logging.warning(f"MCP server {name} did not respond to ping")
            
    except Exception as e:
        logging.error(f"Failed to start MCP server {name}: {e}")


async def mcp_request(server_name, tool_name, **params):
    if server_name not in mcp_servers:
        return {"error": f"MCP server {server_name} not found"}
    
    request = {
        "jsonrpc": "2.0",
        "method": "execute",
        "params": {
            "name": tool_name,
            "parameters": params
        },
        "id": 1
    }
    
    process = mcp_servers[server_name]
    request_str = json.dumps(request) + "\n"
    
    try:
        # Ajout d'un timeout et meilleure gestion des erreurs
        process.stdin.write(request_str.encode())
        await asyncio.wait_for(process.stdin.drain(), timeout=5.0)
        
        # Attendre la réponse avec timeout
        response_line = await asyncio.wait_for(process.stdout.readline(), timeout=10.0)
        if not response_line:
            logging.error(f"MCP server {server_name} returned empty response")
            return {"error": f"Empty response from MCP server {server_name}"}
            
        response_data = json.loads(response_line.decode())
        logging.info(f"MCP response for {tool_name}: {json.dumps(response_data, indent=2)}")
        return response_data
    except asyncio.TimeoutError:
        logging.error(f"Timeout when calling MCP server {server_name}")
        return {"error": f"Timeout when calling MCP server {server_name}"}
    except Exception as e:
        logging.error(f"Error in MCP request to {server_name}: {str(e)}")
        return {"error": f"MCP request failed: {str(e)}"}


@discord_client.event
async def on_ready():
    logging.info(f"Logged in as {discord_client.user}")
    
    # Start MCP servers if enabled
    cfg = get_config()
    if cfg.get('mcp', {}).get('enabled', False):
        for name, server_config in cfg.get('mcp', {}).get('servers', {}).items():
            await start_mcp_server(name, server_config)


# Helper function to handle Anthropic API directly for streaming when needed
async def stream_anthropic_response(api_key, base_url, model, messages, extra_body):
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    # Convert messages from OpenAI format to Anthropic format
    anthropic_messages = []
    system_content = None
    
    # Remove "anthropic/" prefix from model name if present
    if "/" in model:
        model = model.split("/", 1)[1]
    
    for msg in messages:
        if msg.get("role") == "system":
            system_content = msg.get("content")
        else:
            # Handle different content formats
            content = msg.get("content")
            if isinstance(content, list):
                # Handle content blocks (text and images)
                anthropic_content = []
                for block in content:
                    if block.get("type") == "text":
                        anthropic_content.append({
                            "type": "text",
                            "text": block.get("text", "")
                        })
                    elif block.get("type") == "image_url":
                        image_url = block.get("image_url", {}).get("url", "")
                        # Handle base64 images
                        if image_url.startswith("data:"):
                            media_type = image_url.split(";")[0].split(":")[1]
                            base64_data = image_url.split(",")[1]
                            anthropic_content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64_data
                                }
                            })
                        else:
                            anthropic_content.append({
                                "type": "image",
                                "source": {
                                    "type": "url",
                                    "url": image_url
                                }
                            })
                anthropic_messages.append({
                    "role": msg.get("role"),
                    "content": anthropic_content
                })
            else:
                # Simple text content
                anthropic_messages.append({
                    "role": msg.get("role"),
                    "content": [{"type": "text", "text": str(content)}]
                })
    
    payload = {
        "model": model,
        "messages": anthropic_messages,
        "stream": True,
        "max_tokens": extra_body.get("max_tokens", 4096),
        "temperature": extra_body.get("temperature", 1.0)
    }
    
    if system_content:
        payload["system"] = system_content
    
    # Handle tool calling if configured
    if "tools" in extra_body:
        tools = []
        for tool in extra_body["tools"]:
            if tool.get("type") == "function":
                function_info = tool.get("function", {})
                tools.append({
                    "name": function_info.get("name", ""),
                    "description": function_info.get("description", ""),
                    "input_schema": function_info.get("parameters", {})
                })
        if tools:
            payload["tools"] = tools
    
    logging.info(f"Anthropic API request payload: {json.dumps(payload, indent=2)}")
    
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", f"{base_url}/messages", json=payload, headers=headers) as response:
            if response.status_code != 200:
                # Log the error details
                error_content = await response.aread()
                try:
                    error_json = json.loads(error_content)
                    logging.error(f"Anthropic API error: {error_json}")
                except:
                    logging.error(f"Anthropic API error (raw): {error_content.decode('utf-8', errors='replace')}")
            
            response.raise_for_status()
            
            async for chunk in response.aiter_text():
                # Anthropic sends messages prefixed with "data: "
                for line in chunk.split("\n"):
                    if line.startswith("data: "):
                        if line == "data: [DONE]":
                            break
                        
                        # Extract the JSON data
                        try:
                            json_str = line[6:]  # Remove "data: " prefix
                            data = json.loads(json_str)
                            
                            content_delta = ""
                            if data.get("type") == "content_block_delta":
                                if data.get("delta", {}).get("type") == "text_delta":
                                    content_delta = data.get("delta", {}).get("text", "")
                            
                            # Extract finish reason if present
                            stop_reason = None
                            if data.get("type") == "message_stop":
                                stop_reason = "stop"
                            
                            # Yield a chunk object that mimics the OpenAI format
                            class Choice:
                                def __init__(self):
                                    self.delta = type('obj', (object,), {
                                        'content': content_delta,
                                        'tool_calls': None  # Add tool call support if needed
                                    })
                                    self.finish_reason = stop_reason
                            
                            class Chunk:
                                def __init__(self):
                                    self.choices = [Choice()]
                            
                            yield Chunk()
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            logging.error(f"Error processing Anthropic chunk: {e}")
                            continue


@discord_client.event
async def on_message(new_msg):
    global msg_nodes, last_task_time

    is_dm = new_msg.channel.type == discord.ChannelType.private

    if (not is_dm and discord_client.user not in new_msg.mentions) or new_msg.author.bot:
        return

    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(id for id in (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None)) if id)

    cfg = get_config()

    allow_dms = cfg["allow_dms"]
    permissions = cfg["permissions"]

    (allowed_user_ids, blocked_user_ids), (allowed_role_ids, blocked_role_ids), (allowed_channel_ids, blocked_channel_ids) = (
        (perm["allowed_ids"], perm["blocked_ids"]) for perm in (permissions["users"], permissions["roles"], permissions["channels"])
    )

    allow_all_users = not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
    is_good_user = allow_all_users or new_msg.author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
    is_bad_user = not is_good_user or new_msg.author.id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)

    allow_all_channels = not allowed_channel_ids
    is_good_channel = allow_dms if is_dm else allow_all_channels or any(id in allowed_channel_ids for id in channel_ids)
    is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)

    if is_bad_user or is_bad_channel:
        return

    provider, model = cfg["model"].split("/", 1)
    base_url = cfg["providers"][provider]["base_url"]
    api_key = cfg["providers"][provider].get("api_key", "sk-no-key-required")
    openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    accept_images = any(x in model.lower() for x in VISION_MODEL_TAGS)
    accept_usernames = any(x in provider.lower() for x in PROVIDERS_SUPPORTING_USERNAMES)

    max_text = cfg["max_text"]
    max_images = cfg["max_images"] if accept_images else 0
    max_messages = cfg["max_messages"]

    use_plain_responses = cfg["use_plain_responses"]
    max_message_length = 2000 if use_plain_responses else (4096 - len(STREAMING_INDICATOR))

    # Build message chain and set user warnings
    messages = []
    user_warnings = set()
    curr_msg = new_msg

    while curr_msg != None and len(messages) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            if curr_node.text == None:
                cleaned_content = curr_msg.content.removeprefix(discord_client.user.mention).lstrip()

                good_attachments = {type: [att for att in curr_msg.attachments if att.content_type and type in att.content_type] for type in ALLOWED_FILE_TYPES}

                curr_node.text = "\n".join(
                    ([cleaned_content] if cleaned_content else [])
                    + [embed.description for embed in curr_msg.embeds if embed.description]
                    + [(await httpx_client.get(att.url)).text for att in good_attachments["text"]]
                )

                curr_node.images = [
                    dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode((await httpx_client.get(att.url)).content).decode('utf-8')}"))
                    for att in good_attachments["image"]
                ]

                curr_node.role = "assistant" if curr_msg.author == discord_client.user else "user"

                curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None

                curr_node.has_bad_attachments = len(curr_msg.attachments) > sum(len(att_list) for att_list in good_attachments.values())

                try:
                    if (
                        curr_msg.reference == None
                        and discord_client.user.mention not in curr_msg.content
                        and (prev_msg_in_channel := ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0])
                        and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply)
                        and prev_msg_in_channel.author == (discord_client.user if curr_msg.channel.type == discord.ChannelType.private else curr_msg.author)
                    ):
                        curr_node.parent_msg = prev_msg_in_channel
                    else:
                        is_public_thread = curr_msg.channel.type == discord.ChannelType.public_thread
                        parent_is_thread_start = is_public_thread and curr_msg.reference == None and curr_msg.channel.parent.type == discord.ChannelType.text

                        if parent_msg_id := curr_msg.channel.id if parent_is_thread_start else getattr(curr_msg.reference, "message_id", None):
                            if parent_is_thread_start:
                                curr_node.parent_msg = curr_msg.channel.starter_message or await curr_msg.channel.parent.fetch_message(parent_msg_id)
                            else:
                                curr_node.parent_msg = curr_msg.reference.cached_message or await curr_msg.channel.fetch_message(parent_msg_id)

                except (discord.NotFound, discord.HTTPException):
                    logging.exception("Error fetching next message in the chain")
                    curr_node.fetch_parent_failed = True

            if curr_node.images[:max_images]:
                content = ([dict(type="text", text=curr_node.text[:max_text])] if curr_node.text[:max_text] else []) + curr_node.images[:max_images]
            else:
                content = curr_node.text[:max_text]

            if content != "":
                message = dict(content=content, role=curr_node.role)
                if accept_usernames and curr_node.user_id != None:
                    message["name"] = str(curr_node.user_id)

                messages.append(message)

            if len(curr_node.text) > max_text:
                user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
            if len(curr_node.images) > max_images:
                user_warnings.add(f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message" if max_images > 0 else "⚠️ Can't see images")
            if curr_node.has_bad_attachments:
                user_warnings.add("⚠️ Unsupported attachments")
            if curr_node.fetch_parent_failed or (curr_node.parent_msg != None and len(messages) == max_messages):
                user_warnings.add(f"⚠️ Only using last {len(messages)} message{'' if len(messages) == 1 else 's'}")

            curr_msg = curr_node.parent_msg

    logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}")

    if system_prompt := cfg["system_prompt"]:
        system_prompt_extras = [f"Today's date: {dt.now().strftime('%B %d %Y')}."]
        if accept_usernames:
            system_prompt_extras.append("User's names are their Discord IDs and should be typed as '<@ID>'.")

        full_system_prompt = "\n".join([system_prompt] + system_prompt_extras)
        messages.append(dict(role="system", content=full_system_prompt))

    # Configure tools if MCP is enabled
    extra_body = cfg["extra_api_parameters"].copy()
    if cfg.get('mcp', {}).get('enabled', False):
        extra_body.update({
            "tools": [{
                "type": "function",
                "function": {
                    "name": "jira_search",
                    "description": "Search for Jira issues using JQL",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "jql": {
                                "type": "string",
                                "description": "JQL query string"
                            }
                        },
                        "required": ["jql"]
                    }
                }
            }, {
                "type": "function",
                "function": {
                    "name": "jira_get_issue",
                    "description": "Get details of a specific Jira issue",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "issue_key": {
                                "type": "string",
                                "description": "Jira issue key (e.g., PROJECT-123)"
                            }
                        },
                        "required": ["issue_key"]
                    }
                }
            }, {
                "type": "function",
                "function": {
                    "name": "jira_create_issue",
                    "description": "Create a new issue in Jira",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "project_key": {
                                "type": "string",
                                "description": "Project key where the issue will be created"
                            },
                            "issue_type": {
                                "type": "string",
                                "description": "Type of issue (e.g., Bug, Task, Story)"
                            },
                            "summary": {
                                "type": "string",
                                "description": "Issue summary/title"
                            },
                            "description": {
                                "type": "string",
                                "description": "Issue description in markdown format"
                            }
                        },
                        "required": ["project_key", "issue_type", "summary"]
                    }
                }
            }]
        })

    # Generate and send response message(s) (can be multiple if response is long)
    curr_content = finish_reason = edit_task = None
    response_msgs = []
    response_contents = []

    embed = discord.Embed()
    for warning in sorted(user_warnings):
        embed.add_field(name=warning, value="", inline=False)

    kwargs = dict(model=model, messages=messages[::-1], stream=True, extra_body=extra_body)
    try:
        async with new_msg.channel.typing():
            # Use a special handler for Anthropic to avoid compatibility issues
            if provider == "anthropic":
                completion_stream = stream_anthropic_response(api_key, base_url, model, messages[::-1], extra_body)
            else:
                completion_stream = await openai_client.chat.completions.create(**kwargs)
                
            tool_calls_buffer = {}
            
            async for curr_chunk in completion_stream:
                if finish_reason != None:
                    break

                finish_reason = curr_chunk.choices[0].finish_reason

                # Check for tool calls in the response
                if hasattr(curr_chunk.choices[0].delta, 'tool_calls') and curr_chunk.choices[0].delta.tool_calls:
                    for tool_call in curr_chunk.choices[0].delta.tool_calls:
                        if tool_call.index not in tool_calls_buffer:
                            tool_calls_buffer[tool_call.index] = {
                                'name': tool_call.function.name if hasattr(tool_call.function, 'name') else '',
                                'arguments': tool_call.function.arguments if hasattr(tool_call.function, 'arguments') else ''
                            }
                        else:
                            if hasattr(tool_call.function, 'name') and tool_call.function.name:
                                tool_calls_buffer[tool_call.index]['name'] = tool_call.function.name
                            if hasattr(tool_call.function, 'arguments') and tool_call.function.arguments:
                                tool_calls_buffer[tool_call.index]['arguments'] += tool_call.function.arguments

                prev_content = curr_content or ""
                curr_content = curr_chunk.choices[0].delta.content or ""

                new_content = prev_content if finish_reason == None else (prev_content + curr_content)

                if response_contents == [] and new_content == "":
                    continue

                if start_next_msg := response_contents == [] or len(response_contents[-1] + new_content) > max_message_length:
                    response_contents.append("")

                response_contents[-1] += new_content

                if not use_plain_responses:
                    ready_to_edit = (edit_task == None or edit_task.done()) and dt.now().timestamp() - last_task_time >= EDIT_DELAY_SECONDS
                    msg_split_incoming = finish_reason == None and len(response_contents[-1] + curr_content) > max_message_length
                    is_final_edit = finish_reason != None or msg_split_incoming
                    is_good_finish = finish_reason != None and finish_reason.lower() in ("stop", "end_turn")

                    if start_next_msg or ready_to_edit or is_final_edit:
                        if edit_task != None:
                            await edit_task

                        embed.description = response_contents[-1] if is_final_edit else (response_contents[-1] + STREAMING_INDICATOR)
                        embed.color = EMBED_COLOR_COMPLETE if msg_split_incoming or is_good_finish else EMBED_COLOR_INCOMPLETE

                        if start_next_msg:
                            reply_to_msg = new_msg if response_msgs == [] else response_msgs[-1]
                            response_msg = await reply_to_msg.reply(embed=embed, silent=True)
                            response_msgs.append(response_msg)

                            msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                            await msg_nodes[response_msg.id].lock.acquire()
                        else:
                            edit_task = asyncio.create_task(response_msgs[-1].edit(embed=embed))

                        last_task_time = dt.now().timestamp()

            # Process any tool calls
            if tool_calls_buffer and cfg.get('mcp', {}).get('enabled', False):
                tool_results = []
                for index, tool_call in tool_calls_buffer.items():
                    if tool_call['name'].startswith("jira_"):
                        try:
                            function_args = json.loads(tool_call['arguments'])
                            logging.info(f"Executing tool call: {tool_call['name']} with args: {function_args}")
                            
                            result = await mcp_request("atlassian", tool_call['name'], **function_args)
                            tool_results.append({
                                "tool_call_id": str(index),
                                "role": "tool",
                                "content": json.dumps(result)
                            })
                        except Exception as e:
                            logging.error(f"Error executing tool call: {e}")
                            tool_results.append({
                                "tool_call_id": str(index),
                                "role": "tool",
                                "content": json.dumps({"error": str(e)})
                            })
                
                # If we have tool results, make another call to get a final response
                if tool_results:
                    logging.info(f"Sending tool results back to model: {tool_results}")
                    follow_up_messages = messages[::-1] + tool_results
                    
                    # Add a placeholder to show tool execution is happening
                    if not use_plain_responses and response_msgs:
                        placeholder_embed = discord.Embed(
                            description="Processing Jira data...",
                            color=EMBED_COLOR_INCOMPLETE
                        )
                        await response_msgs[-1].edit(embed=placeholder_embed)
                    
                    # Get final response with tool results
                    if provider == "anthropic":
                        # For Anthropic, use a non-streaming call to handle tool results
                        headers = {
                            "x-api-key": api_key,
                            "anthropic-version": "2023-06-01",
                            "content-type": "application/json"
                        }
                        
                        # Remove "anthropic/" prefix from model name if present
                        model_name = model
                        if "/" in model_name:
                            model_name = model_name.split("/", 1)[1]
                        
                        # Convert OpenAI format to Anthropic format
                        anthropic_messages = []
                        system_content = None
                        
                        for msg in follow_up_messages:
                            if msg.get("role") == "system":
                                system_content = msg.get("content")
                            elif msg.get("role") == "tool":
                                # Format des résultats d'outil pour Anthropic Claude 3.5+
                                try:
                                    tool_result_content = msg.get("content", "{}")
                                    tool_call_id = msg.get("tool_call_id", "unknown")
                                    
                                    # S'assurer que le contenu est une chaîne JSON valide
                                    if isinstance(tool_result_content, dict):
                                        tool_result_content = json.dumps(tool_result_content)
                                    
                                    anthropic_messages.append({
                                        "role": "assistant",
                                        "content": [{
                                            "type": "tool_use",
                                            "id": tool_call_id
                                        }]
                                    })
                                    
                                    anthropic_messages.append({
                                        "role": "user", 
                                        "content": [{
                                            "type": "tool_result",
                                            "tool_use_id": tool_call_id,
                                            "content": tool_result_content
                                        }]
                                    })
                                    
                                    logging.info(f"Formatted tool result for Anthropic: {json.dumps(anthropic_messages[-2:], indent=2)}")
                                except Exception as e:
                                    logging.error(f"Error formatting tool result: {e}")
                                    anthropic_messages.append({
                                        "role": "user",
                                        "content": [{
                                            "type": "text",
                                            "text": f"Error processing tool result: {msg.get('content')}"
                                        }]
                                    })
                            else:
                                # Handle different content formats
                                content = msg.get("content")
                                if isinstance(content, list):
                                    # Handle content blocks (text and images)
                                    anthropic_content = []
                                    for block in content:
                                        if block.get("type") == "text":
                                            anthropic_content.append({
                                                "type": "text",
                                                "text": block.get("text", "")
                                            })
                                        elif block.get("type") == "image_url":
                                            image_url = block.get("image_url", {}).get("url", "")
                                            # Handle base64 images
                                            if image_url.startswith("data:"):
                                                media_type = image_url.split(";")[0].split(":")[1]
                                                base64_data = image_url.split(",")[1]
                                                anthropic_content.append({
                                                    "type": "image",
                                                    "source": {
                                                        "type": "base64",
                                                        "media_type": media_type,
                                                        "data": base64_data
                                                    }
                                                })
                                            else:
                                                anthropic_content.append({
                                                    "type": "image",
                                                    "source": {
                                                        "type": "url",
                                                        "url": image_url
                                                    }
                                                })
                                    anthropic_messages.append({
                                        "role": msg.get("role"),
                                        "content": anthropic_content
                                    })
                                else:
                                    # Simple text content
                                    anthropic_messages.append({
                                        "role": msg.get("role"),
                                        "content": [{"type": "text", "text": str(content)}]
                                    })
                        
                        anthropic_payload = {
                            "model": model_name,
                            "messages": anthropic_messages,
                            "max_tokens": extra_body.get("max_tokens", 4096),
                            "temperature": extra_body.get("temperature", 1.0)
                        }
                        
                        if system_content:
                            anthropic_payload["system"] = system_content
                        
                        logging.info(f"Anthropic non-streaming API request: {json.dumps(anthropic_payload, indent=2)}")
                        
                        async with httpx.AsyncClient() as client:
                            try:
                                response = await client.post(
                                    f"{base_url}/messages",
                                    json=anthropic_payload,
                                    headers=headers
                                )
                                
                                if response.status_code != 200:
                                    error_text = response.text
                                    logging.error(f"Anthropic API error: {error_text}")
                                    
                                response.raise_for_status()
                                result = response.json()
                                logging.info(f"Anthropic API response: {json.dumps(result, indent=2)}")
                                
                                # Extract the text content from the response
                                final_content = ""
                                for content_block in result.get("content", []):
                                    if content_block.get("type") == "text":
                                        final_content += content_block.get("text", "")
                            except Exception as e:
                                logging.error(f"Error in Anthropic API call: {str(e)}")
                                final_content = f"Error generating response: {str(e)}"
                    else:
                        # For other providers, use the OpenAI client
                        follow_up_kwargs = dict(model=model, messages=follow_up_messages, extra_body=cfg["extra_api_parameters"])
                        follow_up_response = await openai_client.chat.completions.create(**follow_up_kwargs)
                        final_content = follow_up_response.choices[0].message.content
                    
                    response_contents = [final_content]
                    
                    # Update the message with the final response
                    if not use_plain_responses and response_msgs:
                        final_embed = discord.Embed(
                            description=final_content,
                            color=EMBED_COLOR_COMPLETE
                        )
                        await response_msgs[-1].edit(embed=final_embed)
                    else:
                        # For plain text responses
                        for content in response_contents:
                            reply_to_msg = new_msg if response_msgs == [] else response_msgs[-1]
                            response_msg = await reply_to_msg.reply(content=content, suppress_embeds=True)
                            response_msgs.append(response_msg)
                            
                            msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                            await msg_nodes[response_msg.id].lock.acquire()

            elif use_plain_responses:
                for content in response_contents:
                    reply_to_msg = new_msg if response_msgs == [] else response_msgs[-1]
                    response_msg = await reply_to_msg.reply(content=content, suppress_embeds=True)
                    response_msgs.append(response_msg)

                    msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                    await msg_nodes[response_msg.id].lock.acquire()

    except Exception as e:
        logging.exception(f"Error while generating response: {str(e)}")

    for response_msg in response_msgs:
        msg_nodes[response_msg.id].text = "".join(response_contents)
        msg_nodes[response_msg.id].lock.release()

    # Delete oldest MsgNodes (lowest message IDs) from the cache
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                msg_nodes.pop(msg_id, None)


async def main():
    await discord_client.start(cfg["bot_token"])


asyncio.run(main())
