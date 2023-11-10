# Threads

Details the beta features for creating and managing threads and messages within OpenAI's API, possibly for a customer support or collaborative work tool. It outlines how to:

Create threads with optional messages and metadata.
Retrieve, modify, or delete threads, including updating metadata.
Create messages within threads specifying the role, content, and associated files.
Retrieve, modify, or list messages, with the ability to paginate through messages in a thread.
Attach, retrieve, and list files within messages, useful for tools that need file access.
Each operation is supported by the OpenAI Python client, with code snippets demonstrating how to use these features. The API seems to be designed to handle threaded conversations, perhaps for a chatbot or a customer service application, allowing for structured and organized communication flows.

https://platform.openai.com/docs/api-reference/threads

# Files

The provided information outlines the functionality for handling files within the OpenAI API. Files can be uploaded and are represented by a File object which includes metadata such as file ID, size, creation timestamp, filename, and the purpose of the file. The API allows for listing all files, uploading new files with a maximum size of 512 MB, deleting files, retrieving information about a specific file, and fetching the content of a file. These functions are essential for utilizing OpenAI's features like Assistants and Fine-tuning, where files are used to provide data to the models. The API also supports pagination and filtering by purpose when listing files.

https://platform.openai.com/docs/api-reference/files


# Assistants

The Assistants API allows you to build assistants that can call models and use various tools to perform tasks. Assistants are defined by several properties including an ID, creation timestamp, name, description, model to use, instructions, and tools enabled. Tools can include code_interpreter, retrieval, and function. Assistants can be created, retrieved, modified, and deleted using specific API endpoints. Files can be attached to assistants, and there are also endpoints to manage these files, such as creating, retrieving, listing, and deleting assistant files. This API facilitates the creation of sophisticated assistants tailored to specific tasks or workflows, utilizing the capabilities of OpenAI models.

https://platform.openai.com/docs/api-reference/assistants

The Assistants API allows you to build AI assistants within your own applications. An Assistant has instructions and can leverage models, tools, and knowledge to respond to user queries. The Assistants API currently supports three types of tools: Code Interpreter, Retrieval, and Function calling. In the future, we plan to release more OpenAI-built tools, and allow you to provide your own tools on our platform.

You can explore the capabilities of the Assistants API using the Assistants playground or by building a step-by-step integration outlined in this guide. At a high level, a typical integration of the Assistants API has the following flow:

Create an Assistant in the API by defining its custom instructions and picking a model. If helpful, enable tools like Code Interpreter, Retrieval, and Function calling.
Create a Thread when a user starts a conversation.
Add Messages to the Thread as the user ask questions.
Run the Assistant on the Thread to trigger responses. This automatically calls the relevant tools.
The Assistants API is in beta and we are actively working on adding more functionality. Share your feedback in our Developer Forum!
This starter guide walks through the key steps to create and run an Assistant that uses Code Interpreter.

Step 1: Create an Assistant
An Assistant represents an entity that can be configured to respond to users’ Messages using several parameters like:

Instructions: how the Assistant and model should behave or respond
Model: you can specify any GPT-3.5 or GPT-4 models, including fine-tuned models. The Retrieval tool requires gpt-3.5-turbo-1106 and gpt-4-1106-preview models.
Tools: the API supports Code Interpreter and Retrieval that are built and hosted by OpenAI.
Functions: the API allows you to define custom function signatures, with similar behavior as our function calling feature.
In this example, we're creating an Assistant that is a personal math tutor, with the Code Interpreter tool enabled:

Calls to the Assistants API require that you pass a beta HTTP header. This is handled automatically if you’re using OpenAI’s official Python or Node.js SDKs.
OpenAI-Beta: assistants=v1
python

python
Upgrade to 
Python SDK v1.2
 with pip install --upgrade openai
assistant = client.beta.assistants.create(
    name="Math Tutor",
    instructions="You are a personal math tutor. Write and run code to answer math questions.",
    tools=[{"type": "code_interpreter"}],
    model="gpt-4-1106-preview"
)
Step 2: Create a Thread
A Thread represents a conversation. We recommend creating one Thread per user as soon as the user initiates the conversation. Pass any user-specific context and files in this thread by creating Messages.

python

python
Upgrade to 
Python SDK v1.2
 with pip install --upgrade openai
thread = client.beta.threads.create()
Threads don’t have a size limit. You can pass as many Messages as you want to a Thread. The API will ensure that requests to the model fit within the maximum context window, using relevant optimization techniques such as truncation.

Step 3: Add a Message to a Thread
A Message contains the user's text, and optionally, any files that the user uploads. Image files aren't supported today, but we plan to add support for them in the coming months.

python

python
Upgrade to 
Python SDK v1.2
 with pip install --upgrade openai
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="I need to solve the equation `3x + 11 = 14`. Can you help me?"
)
Now if you list Messages in Thread, you will see that this message is added to the thread on creation:

{
  "object": "list",
  "data": [
    {
      "created_at": 1696995451,
      "id": "msg_4rb1Skx3XgQZEe4PHVRFQhr0",
      "object": "thread.message",
      "thread_id": "thread_34p0sfdas0823smfv",
      "role": "user",
      "content": [{
        "type": "text",
        "text": {
          "value": "I need to solve the equation `3x + 11 = 14`. Can you help me?",
          "annotations": []
        }
          }],
        ...
Step 4: Run the Assistant
For the Assistant to respond to the user message, you need to create a Run. This makes the Assistant read the Thread and decide whether to call tools or simply use the model to best answer the user query. As the run progresses, the assistant appends Messages to the thread with the role="assistant" .

You can optionally pass additional instructions to the Assistant while creating the Run:

python

python
Upgrade to 
Python SDK v1.2
 with pip install --upgrade openai
run = client.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id=assistant.id,
  instructions="Please address the user as Jane Doe. The user has a premium account."
)
Step 5: Display the Assistant's Response
This creates a Run in a queued status. You can periodically retrieve the Run to check on its status to see if it has moved to completed.

python

python
Upgrade to 
Python SDK v1.2
 with pip install --upgrade openai
run = client.beta.threads.runs.retrieve(
  thread_id=thread.id,
  run_id=run.id
)
Once the Run completes, you can retrieve the Messages added by the Assistant to the Thread.

python

python
Upgrade to 
Python SDK v1.2
 with pip install --upgrade openai
messages = client.beta.threads.messages.list(
  thread_id=thread.id
)
And finally, display them to the user! During this Run, the Assistant added two new Messages to the Thread.

ROLE	CONTENT
user	I need to solve the equation 3x + 11 = 14. Can you help me?
assistant	Certainly, Jane Doe. To solve the equation (3x + 11 = 14) for (x), you'll want to isolate (x) on one side of the equation. Here's how you can do that:
Subtract 11 from both sides of the equation to get (3x = 3).
Then, divide both sides by 3 to solve for (x).
Let me calculate the value of (x) for you.
assistant	The solution to the equation (3x + 11 = 14) is (x = 1).
You can also retrieve the Run Steps of this Run if you'd like to explore or display the inner workings of the Assistant and its tools.