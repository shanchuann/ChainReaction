import os
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_openai import ChatOpenAI

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

# Create DeepSeek model with proper configuration
model = ChatOpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat",
    temperature=0.1,
    max_tokens=1000  # Increased for structured output
)

# Create agent with ToolStrategy for structured output
agent = create_agent(
    model=model,
    tools=[],  # No tools needed for extraction
    response_format=ToolStrategy(ContactInfo)
)

# Invoke the agent
result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"
        }
    ]
})

# Access the structured response
print(result["structured_response"])
# Output: ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')