import json
import ollama
from duckduckgo_search import DDGS

# Define local tool function
def search_web(query: str) -> str:
    """Searches the live web using DuckDuckGo and returns snippet results."""
    print(f"Searching the web for: '{query}'...")
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=3)]
            if not results:
                return "No results found."
            return json.dumps(results)
    except Exception as e:
        return f"Error during search: {str(e)}"

# Map string name to the function
available_tools = {
    "search_web": search_web
}

def run_local_agent(user_prompt: str):
    print(f"Starting Local Agentic Task using your GPU...")
    
    # Define the conversation state
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. If you need information you don't know "
                "or that requires live web data, call the search_web tool. "
                "Otherwise, answer directly."
            )
        },
        {"role": "user", "content": user_prompt}
    ]

    # Send the request to local hosted Llama model - explicitly pass Python function into the 'tools' parameter
    response = ollama.chat(
        model="qwen2.5:7b",
        messages=messages,
        tools=[search_web] 
    )
    
    response_message = response['message']
    messages.append(response_message)

    # Check if local model decided it needs to use the tool
    if response_message.get('tool_calls'):
        for tool_call in response_message['tool_calls']:
            tool_name = tool_call['function']['name']
            tool_args = tool_call['function']['arguments']
            
            # Execute the local tool
            if tool_name in available_tools:
                tool_output = available_tools[tool_name](query=tool_args.get("query"))
                
                # Append the results back to the local context window
                messages.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": tool_output
                })
        
        # Let the local model process the data it just searched for
        print("Local model is synthesizing search results...")
        final_response = ollama.chat(
            model="qwen2.5:7b",
            messages=messages
        )
        
        print("\n=== Local Agent Final Report ===")
        print(final_response['message']['content'])
        
    else:
        print("\n=== Local Agent Direct Answer ===")
        print(response_message['content'])

if __name__ == "__main__":
    # Hard coded query while testing the local agent
    prompt = "What is your opinion on the current politcal issues in the Philippines? just give me a brief summary"
    run_local_agent(prompt)
