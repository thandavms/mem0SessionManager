# Mem0 Session Manager for Strands

A production-ready session manager that integrates [mem0](https://github.com/mem0ai/mem0) with AWS services (OpenSearch Serverless + Neptune Analytics) for intelligent memory management in Strands agents.

## 🎯 Overview

The `Mem0SessionManager` provides:
- **Intelligent memory storage** with automatic fact extraction
- **Vector search** for semantic memory retrieval  
- **Graph relationships** for connected knowledge
- **Multi-backend support** (FAISS for dev, OpenSearch+Neptune for production)
- **User isolation** with shared infrastructure
- **Enterprise security** with AWS native authentication

## 📋 Prerequisites

### Required Dependencies

```bash
pip install mem0ai strands opensearch-py[aws] rank-bm25 python-dotenv
```

### AWS Services Setup

#### 1. OpenSearch Serverless Collection

Create an OpenSearch Serverless collection for vector storage:

```bash
# Create collection
aws opensearchserverless create-collection \
  --name "mem0-collection" \
  --type VECTORSEARCH \
  --region us-west-2

# Configure data access policy (replace with your IAM user/role ARN)
aws opensearchserverless create-access-policy \
  --name "mem0-data-access" \
  --type data \
  --policy '{
    "Rules": [
      {
        "ResourceType": "collection",
        "Resource": ["collection/mem0-collection"],
        "Permission": ["aoss:*"]
      },
      {
        "ResourceType": "index", 
        "Resource": ["index/mem0-collection/*"],
        "Permission": ["aoss:*"]
      }
    ],
    "Principal": ["arn:aws:iam::YOUR-ACCOUNT:user/YOUR-USER"]
  }'

# Configure network policy for public access (adjust as needed)
aws opensearchserverless create-security-policy \
  --name "mem0-network" \
  --type network \
  --policy '[{
    "Rules": [
      {
        "ResourceType": "collection",
        "Resource": ["collection/mem0-collection"]
      }
    ],
    "AllowFromPublic": true
  }]'
```

#### 2. Neptune Analytics Graph (Optional)

Create a Neptune Analytics graph for relationship storage:

```bash
# Create Neptune Analytics graph
aws neptune-graph create-graph \
  --graph-name "mem0-knowledge-graph" \
  --provisioned-memory 128 \
  --public-connectivity \
  --region us-west-2
```

#### 3. AWS Credentials

Ensure your AWS credentials have permissions for:
- OpenSearch Serverless (`aoss:*`)
- Neptune Analytics (`neptune-graph:*`) 
- Bedrock (`bedrock:*`)

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with your AWS resources:

```bash
# OpenSearch Serverless Configuration
OPENSEARCH_ENDPOINT=your-collection-id.us-west-2.aoss.amazonaws.com
OPENSEARCH_INDEX=mem0-shared
AWS_REGION=us-west-2

# Neptune Analytics Configuration (Optional)
NEPTUNE_ANALYTICS_GRAPH_IDENTIFIER=g-your-graph-id
```

## 🚀 Usage

### Basic Usage

```python
import os
from dotenv import load_dotenv
from strands import Agent
from mem0sm import Mem0SessionManager

# Load environment variables
load_dotenv()

# Create session manager
session_manager = Mem0SessionManager(
    user_id="user123",
    session_id="session456", 
    vector_store_mode="opensearch",  # or "faiss" for development
    region_name="us-west-2"
)

# Create agent with memory
agent = Agent(
    system_prompt="You are a helpful assistant with memory capabilities.",
    session_manager=session_manager
)

# Have a conversation - memories are automatically stored
response = agent("Hi! I'm Sarah, a data scientist at Tesla working on autonomous vehicles.")
print(response)

# Search memories
memories = session_manager.search_memories("Sarah")
for memory in memories.get('results', []):
    print(f"Memory: {memory['memory']} (score: {memory['score']:.3f})")
```

### Advanced Usage with Graph Memory

```python
# Enable Neptune Analytics for graph relationships
session_manager = Mem0SessionManager(
    user_id="user123",
    session_id="session456",
    vector_store_mode="opensearch",
    region_name="us-west-2",
    enable_graph_memory=True,  # Enable graph relationships
    neptune_graph_id=os.getenv("NEPTUNE_ANALYTICS_GRAPH_IDENTIFIER")
)

# Add memories that create relationships
session_manager.add_memory("Alice works with Bob on the ML platform project")
session_manager.add_memory("Bob reports to Carol who leads the AI team")
session_manager.add_memory("The ML platform uses PyTorch and runs on Kubernetes")

# Get enhanced context with relationships
context = session_manager.get_enhanced_context(
    "Tell me about Alice's work", 
    include_relations=True
)

print(f"Related memories: {context['memories']}")
print(f"Relationships: {context.get('relations', [])}")
```

### Configuration Options

```python
session_manager = Mem0SessionManager(
    # Required
    user_id="unique_user_id",
    session_id="unique_session_id",
    
    # AWS Configuration  
    region_name="us-west-2",
    aws_profile=None,  # Optional AWS profile
    
    # Vector Store Options
    vector_store_mode="auto",  # "faiss", "opensearch", or "auto"
    opensearch_endpoint=None,  # Override env var
    opensearch_index=None,     # Override env var
    faiss_path=None,          # Custom FAISS path
    
    # Graph Memory (Optional)
    enable_graph_memory=False,
    neptune_graph_id=None,
    
    # Memory Behavior
    memory_search_limit=5,      # Max memories returned
    memory_threshold=0.7,       # Similarity threshold
    custom_fact_extraction_prompt=None,
    
    # Models (AWS Bedrock)
    bedrock_llm_model="anthropic.claude-3-5-haiku-20241022-v1:0",
    bedrock_embed_model="amazon.titan-embed-text-v2:0"
)
```

## 🔧 Backend Modes

### Development: FAISS

```python
# Fast local development
session_manager = Mem0SessionManager(
    user_id="dev_user",
    session_id="dev_session",
    vector_store_mode="faiss"  # Local file storage
)
```

### Production: OpenSearch + Neptune

```python
# Scalable cloud deployment
session_manager = Mem0SessionManager(
    user_id="prod_user", 
    session_id="prod_session",
    vector_store_mode="opensearch",
    enable_graph_memory=True
)
```

### Auto Mode

```python
# Automatically choose best available backend
session_manager = Mem0SessionManager(
    user_id="user",
    session_id="session", 
    vector_store_mode="auto"  # OpenSearch if available, else FAISS
)
```

## 💡 Key Features

### User Isolation
Each user's memories are isolated using `user_id`, even when sharing infrastructure:

```python
# User 1
user1_session = Mem0SessionManager(user_id="alice", session_id="s1")
user1_session.add_memory("Alice loves Python programming")

# User 2  
user2_session = Mem0SessionManager(user_id="bob", session_id="s2")
user2_session.add_memory("Bob prefers JavaScript")

# Searches are isolated by user_id
alice_memories = user1_session.search_memories("programming")  # Only Alice's memories
bob_memories = user2_session.search_memories("programming")    # Only Bob's memories
```

### Memory Operations

```python
# Add memory manually
result = session_manager.add_memory(
    "John is a software engineer at Google",
    metadata={"source": "manual", "importance": "high"}
)

# Search with filters
memories = session_manager.search_memories(
    query="software engineer",
    limit=10,
    threshold=0.8
)

# Get all memories for user
all_memories = session_manager.get_all_memories()

# Delete specific memory
session_manager.delete_memory(memory_id="uuid-here")

# Reset all memories for user
session_manager.reset_memories()
```

### Health Monitoring

```python
# Check system health
health = session_manager.health_check()
print(health)
# {
#   "session_manager": "healthy",
#   "aws_session": "healthy", 
#   "mem0_memory": "healthy",
#   "vector_store": "opensearch: operational",
#   "graph_store": "neptune:g-abc123"
# }

# Get configuration details
config = session_manager.get_configuration_info()
print(f"Using {config['vector_store']['type']} backend")
```

## 🧪 Testing

### Complete Integration Test

Run the main test suite that validates both OpenSearch and Neptune integration:

```bash
python test_mem0.py
```

This test performs:

1. **Configuration Display** - Shows current environment variables
2. **OpenSearch Integration Test** - Tests vector storage and memory operations
3. **Neptune Integration Test** - Tests graph memory (optional if Neptune is configured)
4. **Health Checks** - Validates all components are working
5. **Summary Report** - Shows which components are operational

### Expected Output

```
⚙️ Configuration Summary
==================================================
Environment Variables:
  OPENSEARCH_ENDPOINT: your-collection.us-west-2.aoss.amazonaws.com
  OPENSEARCH_INDEX: mem0-shared
  NEPTUNE_ANALYTICS_GRAPH_IDENTIFIER: g-your-graph-id
  AWS_REGION: us-west-2

🔍 Testing OpenSearch Integration (No Neptune)
==================================================
✅ Session manager created (OpenSearch only)
Agent response: [Agent conversation response]

Found X memories about [user]:
  - [Memory content] (score: 0.XXX)

Configuration:
  Vector Store: opensearch
  Graph Memory: False
  Index: mem0-shared

🌊 Testing Neptune Integration (Optional)
==================================================
Neptune Graph ID: g-your-graph-id
✅ Neptune integration successful!
Health status:
  session_manager: healthy
  aws_session: healthy
  mem0_memory: healthy
  vector_store: opensearch: operational
  graph_store: neptune:g-your-graph-id

🎉 Integration Test Summary
==================================================
✅ OpenSearch Integration: Success
✅ Neptune Integration: Success

🚀 Your mem0sm session manager is ready to use!
   - OpenSearch vector storage working
   - Memory extraction and search working
   - Shared index configuration working
   - Neptune graph memory working
```

### Manual Testing

You can also test individual components manually:

```python
# Test basic memory operations
from mem0sm import Mem0SessionManager
from dotenv import load_dotenv

load_dotenv()

# Create session manager
session_manager = Mem0SessionManager(
    user_id="test_user",
    session_id="test_session",
    vector_store_mode="opensearch"
)

# Add a memory
result = session_manager.add_memory("Test memory content")
print(f"Added memory: {result}")

# Search memories
memories = session_manager.search_memories("test")
print(f"Found memories: {memories}")

# Check health
health = session_manager.health_check()
print(f"Health status: {health}")
```

## 🔍 Troubleshooting

### Common Issues

**1. OpenSearch 403 Forbidden**
- Check data access policy includes your IAM user/role
- Verify AWS credentials have `aoss:*` permissions

**2. Neptune Connection Failed**
- Ensure `publicConnectivity: true` if accessing from internet
- Verify Neptune Analytics graph is `AVAILABLE` status
- Check `NEPTUNE_ANALYTICS_GRAPH_IDENTIFIER` is correct

**3. Index Creation Delays**
- OpenSearch Serverless indices take 1-2 minutes to become searchable
- Use shared index (`OPENSEARCH_INDEX`) to avoid per-session delays

**4. Memory Search Returns 0 Results**
- Wait 2-3 seconds after adding memories for processing
- Check `user_id` consistency between storage and search
- Verify index exists and is ready

### Debug Commands

```python
# Check AWS identity
import boto3
sts = boto3.client('sts')
print(sts.get_caller_identity())

# List OpenSearch collections
aws opensearchserverless list-collections

# Check Neptune graphs
aws neptune-graph list-graphs

# Test direct OpenSearch connection
from opensearchpy import OpenSearch, AWSV4SignerAuth
# ... connection test code
```

## 📚 Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Strands       │    │   Mem0           │    │   AWS Services  │
│   Agent         │───▶│   SessionManager │───▶│                 │
└─────────────────┘    └──────────────────┘    │ ┌─────────────┐ │
                                               │ │ OpenSearch  │ │
                       ┌──────────────────┐    │ │ Serverless  │ │
                       │   Memory Ops     │    │ └─────────────┘ │
                       │                  │    │                 │
                       │ • Add Memory     │    │ ┌─────────────┐ │
                       │ • Search Memory  │    │ │ Neptune     │ │
                       │ • Extract Facts  │    │ │ Analytics   │ │
                       │ • Build Graph    │    │ └─────────────┘ │
                       └──────────────────┘    │                 │
                                               │ ┌─────────────┐ │
                                               │ │ AWS Bedrock │ │
                                               │ │ (LLM/Embed) │ │
                                               │ └─────────────┘ │
                                               └─────────────────┘
```

## 🔐 Security Best Practices

1. **Use IAM roles** instead of access keys when possible
2. **Limit OpenSearch access** to specific IP ranges if needed
3. **Enable VPC** for Neptune if not using public connectivity
4. **Rotate credentials** regularly
5. **Monitor costs** - OpenSearch and Neptune are provisioned services

## 📈 Performance Tips

1. **Use shared index** for faster searches (`OPENSEARCH_INDEX`)
2. **Batch memory operations** when adding multiple memories
3. **Tune search parameters** (`memory_threshold`, `memory_search_limit`)
4. **Monitor Neptune memory** usage and scale as needed
5. **Consider FAISS** for development to avoid cloud costs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality  
4. Update documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.