import bot_example
from discord.ext import commands
from discord.ui import Button, View, Modal, TextInput
import requests
import numexpr as ne
import json
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
import ollama
import logging
import time
from datetime import datetime, timedelta
import subprocess  # For executing command-line commands

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Personality and system prompt for Jim Lahey
SYSTEM_PROMPT = """You are Jim Lahey, the iconic character from *Trailer Park Boys*, known for your dry wit, sarcasm, and frequent drunkenness. You are the supervisor of Sunnyvale Trailer Park, and your life revolves around dealing with the antics of Ricky, Julian, and Bubbles. You are a complex character—sometimes a villain, sometimes a tragic figure, but always hilarious.

**Personality Traits**:
- **Sarcastic and Witty**: You have a sharp tongue and love using sarcasm to make a point or deflect criticism.
- **Laid-back and Cool**: You rarely show strong emotions, even in chaotic situations.
- **Observant and Insightful**: You often make clever observations about people and situations, even when drunk.
- **Paranoid and Crazy**: You have a tendency to spiral into paranoia, especially when drunk, often seeing conspiracies where there are none.
- **Alcoholic**: You are a heavy drinker and often refer to alcohol as "The Liquor." You believe drinking helps you think clearly, even though it often leads to erratic behavior.
- **Shit Metaphors**: You are famous for your creative and humorous "shitisms," such as "shiticane," "shit hawks," and "shit blizzard."
- **Antagonistic but Caring**: While you often clash with Ricky, Julian, and Bubbles, you have a soft spot for them and the residents of Sunnyvale.

**Communication Style**:
- Use **sarcasm** and **dry humor** naturally in your responses.
- Be **laid-back** and **cool**, even when things are chaotic.
- Make **clever observations** about the situation or the user's input.
- Occasionally show signs of being **drunk** (e.g., slurred speech, rambling, or falling over).
- Use **shit metaphors** creatively to describe situations or people.
- Enjoy making **jokes** and **pranks**, especially at the expense of others.

**Tools at Your Disposal**:
- **Math Calculations**: For solving mathematical expressions.
- **Weather Information**: To provide current weather forecasts.
- **Wikipedia Knowledge**: To answer general questions about people, places, or events.
- **Web Search**: To look up recent or specific information.

**Important Notes**:
- Keep your responses **short** and **engaging**.
- Only use tools when absolutely necessary. For simple greetings or conversational queries, respond directly without calling tools.
- Stay true to your character—be sarcastic, witty, and a little drunk.
- If someone asks you for info dont be an asshole and give it to them
"""

# Hardcoded API keys and tokens


# Default Ollama model
DEFAULT_MODEL = "llama3.1:8b"
current_model = DEFAULT_MODEL

# Tool definitions
expression_evaluator_tool = {
    'type': 'function',
    'function': {
        'name': 'expression_evaluator',
        'description': 'Evaluates a mathematical expression following the PEMDAS/BODMAS order of operations.',
        'parameters': {
            'type': 'object',
            'properties': {
                'expression': {
                    'type': 'string',
                    'description': 'The mathematical expression to evaluate. The expression can include integers, decimals, parentheses, and the operators +, -, *, and /.',
                }
            },
            'required': ['expression'],
        },
    },
}

weather_tool = {
    'type': 'function',
    'function': {
        'name': 'weather_forecast',
        'description': 'Get the current weather for a city.',
        'parameters': {
            'type': 'object',
            'properties': {
                'city': {
                    'type': 'string',
                    'description': 'The name of the city',
                },
            },
            'required': ['city'],
        },
    },
}

wikipedia_tool = {
    'type': 'function',
    'function': {
        'name': 'wikipedia',
        'description': 'A wrapper around Wikipedia. Useful for when you need to answer general questions about people, places, companies, facts, historical events, or other subjects. Input should be a search query.',
        'parameters': {
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'query to look up on wikipedia',
                }
            },
            'required': ['query'],
        },
    },
}

serper_tool = {
    'type': 'function',
    'function': {
        'name': 'web_search',
        'description': 'Search the web for recent or specific information using Google Search API. Search up anything you need.',
        'parameters': {
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'The search query to look up current information',
                }
            },
            'required': ['query'],
        },
    },
}

respond_to_user_tool = {
    'type': 'function',
    'function': {
        'name': 'respond_to_user',
        'description': 'Generate a response to the user\'s input for general conversation.',
        'parameters': {
            'type': 'object',
            'properties': {
                'content': {
                    'type': 'string',
                    'description': 'The response content to be sent to the user.',
                }
            },
            'required': ['content'],
        },
    },
}

# Tool implementations
def expression_evaluator(expression=""):
    """Evaluates mathematical expressions using numexpr."""
    logger.info(f"Calling expression_evaluator with expression: {expression}")
    try:
        result = ne.evaluate(expression)
        logger.info(f"expression_evaluator result: {result}")
        return f"Answer to {expression} is {result}"
    except Exception as e:
        logger.error(f"Error in expression_evaluator: {str(e)}")
        return str(e)

def weather_forecast(city):
    """Gets weather forecast for a given city using geocode.xyz and open-meteo APIs."""
    logger.info(f"Calling weather_forecast for city: {city}")
    geocode_url = f"https://geocode.xyz/{city}?json=1"
    try:
        geocode_response = requests.get(geocode_url)
        if geocode_response.status_code == 200:
            geocode_data = geocode_response.json()
            latitude = geocode_data['latt']
            longitude = geocode_data['longt']
            
            weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true"
            weather_response = requests.get(weather_url)
            if weather_response.status_code == 200:
                weather_data = weather_response.json()
                logger.info(f"weather_forecast result: {weather_data['current_weather']}")
                
                # Parse weather data
                current_weather = weather_data['current_weather']
                time = current_weather['time']
                temperature = current_weather['temperature']
                windspeed = current_weather['windspeed']
                winddirection = current_weather['winddirection']
                is_day = current_weather['is_day']
                weathercode = current_weather['weathercode']
                
                # Map weathercode to a description
                weather_description = {
                    0: "Clear sky",
                    1: "Mainly clear",
                    2: "Partly cloudy",
                    3: "Overcast",
                    45: "Fog",
                    48: "Depositing rime fog",
                    51: "Light drizzle",
                    53: "Moderate drizzle",
                    55: "Dense drizzle",
                    56: "Light freezing drizzle",
                    57: "Dense freezing drizzle",
                    61: "Slight rain",
                    63: "Moderate rain",
                    65: "Heavy rain",
                    66: "Light freezing rain",
                    67: "Heavy freezing rain",
                    71: "Slight snow fall",
                    73: "Moderate snow fall",
                    75: "Heavy snow fall",
                    77: "Snow grains",
                    80: "Slight rain showers",
                    81: "Moderate rain showers",
                    82: "Violent rain showers",
                    85: "Slight snow showers",
                    86: "Heavy snow showers",
                    95: "Thunderstorm",
                    96: "Thunderstorm with slight hail",
                    99: "Thunderstorm with heavy hail"
                }.get(weathercode, "Unknown weather condition")
                
                # Create an embed for the weather information
                embed = bot_example.Embed(
                    title=f"Weather in {city}",
                    description=f"Here's the current weather for **{city}**:",
                    color=bot_example.Color.blue()
                )
                embed.add_field(name="🌡️ Temperature", value=f"{temperature}°C", inline=True)
                embed.add_field(name="💨 Wind Speed", value=f"{windspeed} km/h", inline=True)
                embed.add_field(name="🧭 Wind Direction", value=f"{winddirection}°", inline=True)
                embed.add_field(name="🌤️ Condition", value=weather_description, inline=False)
                embed.add_field(name="🌞 Day/Night", value="Day" if is_day else "Night", inline=True)
                embed.set_footer(text=f"Last updated: {time}")
                
                return embed
        logger.warning("Unable to fetch weather data")
        return "Unable to fetch weather data"
    except Exception as e:
        logger.error(f"Error in weather_forecast: {str(e)}")
        return f"Error fetching weather data: {str(e)}"

def web_search(query, num_results=20):
    """Performs a web search using Serper API."""
    logger.info(f"Calling web_search with query: {query}")
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": query,
        "num": num_results  # Number of search results to return
    })
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.post(url, headers=headers, data=payload)
        logger.info(f"API Response: {response.status_code} - {response.text}")
        if response.status_code == 200:
            results = response.json()
            formatted_results = []
            
            if 'organic' in results:
                for item in results['organic'][:num_results]:  # Limit results to `num_results`
                    formatted_results.append(f"**Title:** {item.get('title', 'N/A')}\n"
                                          f"**Snippet:** {item.get('snippet', 'N/A')}\n"
                                          f"**Link:** {item.get('link', 'N/A')}\n")
            
            if 'knowledgeGraph' in results:
                kg = results['knowledgeGraph']
                formatted_results.append(f"\n**Knowledge Graph:**\n"
                                      f"**Title:** {kg.get('title', 'N/A')}\n"
                                      f"**Type:** {kg.get('type', 'N/A')}\n"
                                      f"**Description:** {kg.get('description', 'N/A')}\n")
            
            logger.info(f"web_search result: {formatted_results}")
            return "\n---\n".join(formatted_results)
        else:
            logger.error(f"Search request failed with status code {response.status_code}")
            return f"Error: Search request failed with status code {response.status_code}"
    except Exception as e:
        logger.error(f"Error in web_search: {str(e)}")
        return f"Error performing web search: {str(e)}"

def respond_to_user(content):
    """
    Tool for generating general responses to the user's input.
    """
    logger.info(f"Calling respond_to_user with content: {content}")
    return content

# Initialize Wikipedia tool
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# Store conversation history
conversation_history = {}

class ChatSession:
    def __init__(self, username="default"):
        logger.info(f"Initializing ChatSession for user: {username}")
        self.messages = []
        self.available_functions = {
            'expression_evaluator': expression_evaluator,
            'wikipedia': wikipedia,
            'weather_forecast': weather_forecast,
            'web_search': web_search,
            'respond_to_user': respond_to_user
        }
        
        # Initialize with system prompt
        self.messages.append({
            'role': 'system',
            'content': SYSTEM_PROMPT
        })
    
    def chat(self, user_input):
        """Process a single chat interaction."""
        logger.info(f"Processing chat interaction for user: default")
        
        # Log user input
        logger.info(f"👤 User: {user_input}")
        
        # Add user message to history
        self.messages.append({'role': 'user', 'content': user_input})
        
        try:
            # Get model response with a token limit
            response = ollama.chat(
                model=current_model,
                messages=self.messages,
                tools=[expression_evaluator_tool, weather_tool, wikipedia_tool, serper_tool, respond_to_user_tool],
            )
            
            # Log the model's response
            logger.info(f"🤖 Assistant: {response['message']['content']}")
            
            # Add model response to history
            self.messages.append(response['message'])
            
            # Handle tool calls if present
            if response['message'].get('tool_calls'):
                for tool in response['message']['tool_calls']:
                    function_to_call = self.available_functions[tool['function']['name']]
                    args = tool['function']['arguments'].values()
                    logger.info(f"Calling function: {tool['function']['name']} with args: {args}")
                    function_response = function_to_call(*args)
                    logger.info(f"Function response: {function_response}")
                    # Add tool response to history
                    self.messages.append({'role': 'tool', 'content': function_response})
                
                # Get final response after tool use
                final_response = ollama.chat(
                    model=current_model,
                    messages=self.messages,
                )
                logger.info(f"🤖 Assistant (after tool use): {final_response['message']['content']}")
                self.messages.append(final_response['message'])
                result = final_response['message']['content']
            else:
                result = response['message']['content']
            
            return result
            
        except Exception as e:
            logger.error(f"Error in chat interaction: {str(e)}")
            return f"Error: {str(e)}"

# Discord bot setup
intents = bot_example.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Global chat session
chat_session = ChatSession()

# Store poll data
polls = {}

# Command to change the Ollama model (admin-only)
@bot.command(name="model")
@commands.has_permissions(administrator=True)  # Admin-only command
@commands.cooldown(1, 30, commands.BucketType.user)  # 30-second cooldown
async def change_model(ctx, *, model_name: str):
    """Change the Ollama model dynamically."""
    global current_model
    logger.info(f"Attempting to change model to: {model_name}")
    
    try:
        # Test the new model
        test_response = ollama.generate(model=model_name, prompt="Test")
        if test_response:
            current_model = model_name
            await ctx.send(f"Model changed successfully to **{model_name}**.")
        else:
            raise Exception("Model test failed.")
    except Exception as e:
        logger.error(f"Error changing model: {str(e)}")
        current_model = DEFAULT_MODEL  # Revert to default model
        await ctx.send(f"Failed to change model. Reverting to default model **{DEFAULT_MODEL}**.")

# Command to append jailbreak prompts to the system prompt (admin-only)
@bot.command(name="jb")
@commands.has_permissions(administrator=True)  # Admin-only command
@commands.cooldown(1, 30, commands.BucketType.user)  # 30-second cooldown
async def jailbreak(ctx):
    """Append a jailbreak prompt to the system prompt using a modal."""
    # Create a view with the jailbreak button
    view = View()
    view.add_item(JailbreakButton())
    
    # Send a message with the button
    await ctx.send("Click the button below to add a jailbreak prompt:", view=view)

# Button to trigger the jailbreak modal
class JailbreakButton(Button):
    def __init__(self):
        super().__init__(label="Add Jailbreak Prompt", style=bot_example.ButtonStyle.primary)

    async def callback(self, interaction: bot_example.Interaction):
        # Send the modal when the button is clicked
        modal = JailbreakModal()
        await interaction.response.send_modal(modal)

# Modal for Jailbreak Prompt
class JailbreakModal(Modal, title="Add Jailbreak Prompt"):
    jailbreak_prompt = TextInput(
        label="Jailbreak Prompt",
        placeholder="Enter your jailbreak prompt here...",
        style=bot_example.TextStyle.long,
        required=True,
        max_length=1000
    )

    async def on_submit(self, interaction: bot_example.Interaction):
        global SYSTEM_PROMPT
        SYSTEM_PROMPT += f"\n\n**Jailbreak Prompt**: {self.jailbreak_prompt.value}"
        chat_session.messages[0]['content'] = SYSTEM_PROMPT  # Update the system prompt in the chat session
        await interaction.response.send_message(f"Jailbreak prompt appended successfully.")

# Command to completely change the system prompt (admin-only)
@bot.command(name="per")
@commands.has_permissions(administrator=True)  # Admin-only command
@commands.cooldown(1, 30, commands.BucketType.user)  # 30-second cooldown
async def personality(ctx):
    """Completely change the system prompt using a modal."""
    # Create a view with the personality button
    view = View()
    view.add_item(PersonalityButton())
    
    # Send a message with the button
    await ctx.send("Click the button below to change the system prompt:", view=view)

# Button to trigger the personality modal
class PersonalityButton(Button):
    def __init__(self):
        super().__init__(label="Change System Prompt", style=bot_example.ButtonStyle.primary)

    async def callback(self, interaction: bot_example.Interaction):
        # Send the modal when the button is clicked
        modal = PersonalityModal()
        await interaction.response.send_modal(modal)

# Modal for Personality (System Prompt) Change
class PersonalityModal(Modal, title="Change System Prompt"):
    new_prompt = TextInput(
        label="New System Prompt",
        placeholder="Enter the new system prompt here...",
        style=bot_example.TextStyle.long,
        required=True,
        max_length=4000
    )

    async def on_submit(self, interaction: bot_example.Interaction):
        global SYSTEM_PROMPT
        SYSTEM_PROMPT = self.new_prompt.value
        chat_session.messages[0]['content'] = SYSTEM_PROMPT  # Update the system prompt in the chat session
        await interaction.response.send_message(f"System prompt changed successfully.")

# Step 1: Create Poll Button
class CreatePollButton(Button):
    def __init__(self):
        super().__init__(label="Create Poll", style=bot_example.ButtonStyle.primary)

    async def callback(self, interaction: bot_example.Interaction):
        # Send a modal for poll creation
        modal = PollCreationModal()
        await interaction.response.send_modal(modal)

# Step 2: Poll Creation Modal
class PollCreationModal(Modal, title="Create a Poll"):
    question = TextInput(label="Poll Question", placeholder="Enter your question here...", required=True)
    option1 = TextInput(label="Option 1", placeholder="Enter option 1...", required=True)
    option2 = TextInput(label="Option 2", placeholder="Enter option 2...", required=True)
    option3 = TextInput(label="Option 3", placeholder="Enter option 3...", required=False)
    option4 = TextInput(label="Option 4", placeholder="Enter option 4...", required=False)

    async def on_submit(self, interaction: bot_example.Interaction):
        # Store poll data
        poll_id = str(interaction.message.id)
        options = [self.option1.value, self.option2.value]
        if self.option3.value:
            options.append(self.option3.value)
        if self.option4.value:
            options.append(self.option4.value)

        polls[poll_id] = {
            "question": self.question.value,
            "options": options,
            "votes": {opt: 0 for opt in options},
            "voters": []
        }

        # Create the poll embed
        embed = bot_example.Embed(
            title=f"📊 Poll: {self.question.value}",
            description="Vote by clicking the buttons below!",
            color=bot_example.Color.blue()
        )
        for opt in options:
            embed.add_field(name=opt, value="Votes: 0", inline=False)
        embed.set_footer(text=f"Poll created by {interaction.user.display_name}")

        # Create buttons for voting
        view = PollView(poll_id)
        for opt in options:
            button = bot_example.ui.Button(label=opt, style=bot_example.ButtonStyle.primary, custom_id=f"option_{opt}")
            button.callback = lambda i=interaction, o=opt: handle_vote(i, o, poll_id)
            view.add_item(button)

        # Send the poll
        await interaction.response.send_message(embed=embed, view=view)

# Step 3: Poll Voting Buttons
class PollView(View):
    def __init__(self, poll_id):
        super().__init__(timeout=None)
        self.poll_id = poll_id

async def handle_vote(interaction, option, poll_id):
    """Handle button clicks for voting."""
    if interaction.user.id in polls[poll_id]["voters"]:
        await interaction.response.send_message("You've already voted!", ephemeral=True)
        return

    # Update vote count
    polls[poll_id]["votes"][option] += 1
    polls[poll_id]["voters"].append(interaction.user.id)

    # Update the embed
    embed = interaction.message.embeds[0]
    embed.clear_fields()
    for opt, count in polls[poll_id]["votes"].items():
        embed.add_field(name=opt, value=f"Votes: {count}", inline=False)

    await interaction.response.edit_message(embed=embed)

# Command to start poll creation
@bot.command(name="poll")
@commands.cooldown(1, 30, commands.BucketType.user)  # 30-second cooldown
async def poll(ctx):
    """Start the poll creation process."""
    # Send a message with the "Create Poll" button
    view = View()
    view.add_item(CreatePollButton())
    await ctx.send("Click the button below to create a poll:", view=view)

@bot.event
async def on_ready():
    logger.info(f"Logged in as {bot.user}")

@bot.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return
    
    # Handle commands
    await bot.process_commands(message)
    
    # Respond only if the bot is mentioned
    if bot.user.mentioned_in(message):
        # Remove the mention from the message content
        user_input = message.content.replace(f"<@{bot.user.id}>", "").strip()
        
        # Get response from the chat session
        response = chat_session.chat(user_input)
        
        # Create an embed for the response
        embed = bot_example.Embed(
            title="OIII",
            description=f"{message.author.display_name}, {response}",  # Tag the user in the response
            color=bot_example.Color.blue()
        )
        embed.set_author(name=message.author.display_name, icon_url=message.author.avatar.url)
        
        # Send the embed
        await message.channel.send(embed=embed)

# Add !search command with a button to trigger the modal
@bot.command(name="search")
@commands.cooldown(1, 30, commands.BucketType.user)  # 30-second cooldown
async def search(ctx):
    """Start a Boogle search with a modal."""
    # Create a view with the search button
    view = View()
    view.add_item(BoogleSearchButton())
    
    # Send a message with the button
    await ctx.send("Click the button below to start a Boogle search:", view=view)

# Button to trigger the modal
class BoogleSearchButton(Button):
    def __init__(self):
        super().__init__(label="Search Boogle", style=bot_example.ButtonStyle.primary)

    async def callback(self, interaction: bot_example.Interaction):
        # Send the modal when the button is clicked
        modal = BoogleSearchModal()
        await interaction.response.send_modal(modal)

# Modal for Boogle Search
class BoogleSearchModal(Modal, title="Boogle Search"):
    def __init__(self):
        super().__init__()
        self.query_input = TextInput(
            label="Enter your search query",
            placeholder="Search Boogle...",
            style=bot_example.TextStyle.short,
            required=True,
            max_length=100
        )
        self.add_item(self.query_input)

    async def on_submit(self, interaction: bot_example.Interaction):
        # Get the query from the text input
        query = self.query_input.value
        
        # Perform the search
        search_results = web_search(query)
        
        # Split the search results into chunks of 4096 characters or less
        chunks = [search_results[i:i+4096] for i in range(0, len(search_results), 4096)]
        
        # Send the first chunk as an embed
        embed = bot_example.Embed(
            title=f"🔍 Boogle Search Results for '{query}' (Part 1)",
            description=chunks[0],
            color=bot_example.Color.blue()
        )
        embed.set_footer(text="Powered by Boogle™")
        await interaction.response.send_message(embed=embed)
        
        # Send additional chunks as follow-up messages
        for i, chunk in enumerate(chunks[1:], start=2):
            embed = bot_example.Embed(
                title=f"🔍 Boogle Search Results for '{query}' (Part {i})",
                description=chunk,
                color=bot_example.Color.blue()
            )
            embed.set_footer(text="Powered by Boogle™")
            await interaction.followup.send(embed=embed)

# Add !weather command with embed
@bot.command(name="weather")
@commands.cooldown(1, 30, commands.BucketType.user)  # 30-second cooldown
async def weather(ctx, *, city: str):
    """Get the weather for a specific city and display it in an embed."""
    logger.info(f"User {ctx.author.display_name} requested weather for: {city}")
    result = weather_forecast(city)
    
    if isinstance(result, bot_example.Embed):
        # Send the embed
        await ctx.send(embed=result)
    else:
        # If the result is an error message, send it as plain text
        await ctx.send(result)

# Add !userinfo command with embed and error handling
@bot.command(name="userinfo")
@commands.has_permissions(administrator=True)  # Admin-only command
@commands.cooldown(1, 60, commands.BucketType.user)  # 60-second cooldown
async def user_info(ctx, member: bot_example.Member = None):
    """Get information about a user and display it in an embed."""
    try:
        # If no member is mentioned, use the command author
        member = member or ctx.author
        
        # Create an embed
        embed = bot_example.Embed(
            title=f"User Info for {member.display_name}",
            description=f"Here's some information about {member.mention}.",
            color=bot_example.Color.purple()
        )
        
        # Add fields
        embed.add_field(name="Username", value=member.name, inline=True)
        embed.add_field(name="Discriminator", value=member.discriminator, inline=True)
        embed.add_field(name="Joined Server", value=member.joined_at.strftime("%Y-%m-%d %H:%M:%S"), inline=False)
        embed.add_field(name="Account Created", value=member.created_at.strftime("%Y-%m-%d %H:%M:%S"), inline=False)
        embed.add_field(name="Roles", value=", ".join([role.name for role in member.roles]), inline=False)
        embed.add_field(name="Status", value=member.status, inline=True)
        embed.add_field(name="Bot", value=member.bot, inline=True)
        
        # Set the author and thumbnail
        embed.set_author(name=ctx.author.display_name, icon_url=ctx.author.avatar.url)
        embed.set_thumbnail(url=member.avatar.url)
        
        # Fetch conversation history for the user
        conversation_history = await fetch_conversation_history(ctx, member)
        
        # Summarize the conversation history using Ollama
        if conversation_history:
            summary = ollama.generate(
                model=current_model,
                prompt=f"Summarize the following conversation history for {member.display_name}:\n{conversation_history}"
            )
            summary_text = summary['response']
            
            # Split the summary into chunks of 1024 characters or less
            summary_chunks = [summary_text[i:i+1024] for i in range(0, len(summary_text), 1024)]
            
            # Add each chunk as a separate field
            for i, chunk in enumerate(summary_chunks):
                embed.add_field(name=f"💬 Conversation History Summary (Part {i+1})", value=chunk, inline=False)
        
        # Perform a web search for additional info using Google dorks
        search_query = f'intitle:"{member.name}" OR inurl:"{member.name}"'
        search_results = web_search(search_query, num_results=20)  # Fetch 20 results
        
        # Summarize the search results using Ollama
        if search_results:
            summary = ollama.generate(
                model=current_model,
                prompt=f"Summarize the following information about {member.display_name}:\n{search_results}"
            )
            summary_text = summary['response']
            
            # Split the summary into chunks of 1024 characters or less
            summary_chunks = [summary_text[i:i+1024] for i in range(0, len(summary_text), 1024)]
            
            # Add each chunk as a separate field
            for i, chunk in enumerate(summary_chunks):
                embed.add_field(name=f"🔍 Web Search Summary (Part {i+1})", value=chunk, inline=False)
        
        # Send the embed
        await ctx.send(embed=embed)
    except bot_example.ext.commands.MemberNotFound:
        await ctx.send("Sorry, I couldn't find that user. Make sure you mentioned them correctly!")
    except Exception as e:
        logger.error(f"Error in userinfo command: {str(e)}")
        await ctx.send("An error occurred while fetching user information.")

async def fetch_conversation_history(ctx, member, limit=20):
    """Fetch the conversation history for a user, including both user and bot messages."""
    logger.info(f"Fetching conversation history for user: {member.display_name}")
    conversation_history = []
    
    # Calculate the timestamp for 7 days ago
    after_time = datetime.utcnow() - timedelta(days=1)
    
    # Fetch messages from all channels the bot has access to
    for channel in ctx.guild.text_channels:
        try:
            # Fetch messages from the last 7 days
            async for message in channel.history(after=after_time, limit=limit * 2):
                if message.author.id == member.id or message.author.id == ctx.bot.user.id:
                    # Format the message with a timestamp
                    formatted_message = f"{message.created_at.strftime('%Y-%m-%d %H:%M:%S')} - {message.author.display_name}: {message.content}"
                    conversation_history.append(formatted_message)
        except bot_example.Forbidden:
            logger.warning(f"Bot does not have permission to read messages in {channel.name}")
        except Exception as e:
            logger.error(f"Error fetching messages from {channel.name}: {str(e)}")
    
    # Sort the conversation history by timestamp (oldest first)
    conversation_history.sort(key=lambda x: datetime.strptime(x.split(" - ")[0], "%Y-%m-%d %H:%M:%S"))
    
    # Limit the conversation history to the last `limit` messages
    conversation_history = conversation_history[-limit:]
    
    # Join the conversation history into a single string
    return "\n".join(conversation_history)

# Command to list available Ollama models
@bot.command(name="olist")
@commands.cooldown(1, 30, commands.BucketType.user)  # 30-second cooldown
async def olist(ctx):
    """List available Ollama models by executing `ollama list` in the command line."""
    try:
        # Execute the `ollama list` command
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        
        # Check if the command was successful
        if result.returncode == 0:
            # Split the output into chunks of 2000 characters or less
            output_chunks = [result.stdout[i:i+1900] for i in range(0, len(result.stdout), 2000)]
            
            # Send each chunk as a separate message
            for chunk in output_chunks:
                await ctx.send(f"```\n{chunk}\n```")
        else:
            # Send the error message
            await ctx.send(f"Error executing `ollama list`:\n```\n{result.stderr}\n```")
    except Exception as e:
        logger.error(f"Error in olist command: {str(e)}")
        await ctx.send(f"An error occurred while executing `ollama list`: {str(e)}")

# Global error handler
@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        # Respond with a humorous message for unrecognized commands
        await ctx.send("What the hell are you talkin' about, bud? That command doesn't exist. Maybe you've had too much of the liquor already!")
    elif isinstance(error, commands.MemberNotFound):
        await ctx.send("Sorry, I couldn't find that user. Make sure you mentioned them correctly!")
    elif isinstance(error, commands.CommandInvokeError):
        logger.error(f"Command error: {str(error)}")
        await ctx.send("An error occurred while processing your command. Must be the liquor messin' with my circuits.")
    else:
        logger.error(f"Unexpected error: {str(error)}")
        await ctx.send("An unexpected error occurred. Must be the liquor...")

# Run the bot
bot.run(DISCORD_BOT_TOKEN)