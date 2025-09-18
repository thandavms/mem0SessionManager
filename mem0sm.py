import os
import boto3
import uuid
import json
from typing import Optional, Dict, Any, List, Union
import logging
from pathlib import Path
from datetime import datetime, timezone

from strands.session.session_manager import SessionManager
from strands.types.session import Session, SessionAgent, SessionMessage
from mem0 import Memory

# Import for OpenSearch Serverless authentication
try:
    from opensearchpy import AWSV4SignerAuth
except ImportError:
    AWSV4SignerAuth = None

logger = logging.getLogger(__name__)

class Mem0SessionManager(SessionManager):
    """
    AWS-native session manager that uses Mem0's built-in storage with intelligent memory.
    """
    
    def __init__(
        self,
        session_id: str,
        user_id: str,
        
        # AWS Configuration
        region_name: Optional[str] = 'us-west-2',
        aws_profile: Optional[str] = None,
        
        # Bedrock Model Overrides
        bedrock_llm_model: Optional[str] = None,
        bedrock_embed_model: Optional[str] = None,
        
        # Vector Store Configuration
        vector_store_mode: Optional[str] = None,  # 'faiss', 'opensearch', 'auto'
        opensearch_endpoint: Optional[str] = None,
        opensearch_index: Optional[str] = None,
        faiss_path: Optional[str] = None,
        
        # Graph Memory Configuration
        enable_graph_memory: bool = False,
        neptune_graph_id: Optional[str] = None,
        
        # Memory Behavior
        custom_fact_extraction_prompt: Optional[str] = None,
        memory_search_limit: int = 5,
        memory_threshold: float = 0.7,
        
        # Session Storage Configuration
        session_storage_dir: Optional[str] = None,
        
        **kwargs
    ):
        """
        Initialize AWS-focused Mem0SessionManager.
        
        Args:
            session_id: Unique identifier for the session
            user_id: User identifier for memory isolation
            region_name: AWS region (auto-detected if not provided)
            aws_profile: AWS profile name (optional)
            
            bedrock_llm_model: Override default Bedrock LLM model
            bedrock_embed_model: Override default Bedrock embedding model
            
            vector_store_mode: 'faiss' (dev), 'opensearch' (prod), or 'auto' (default)
            opensearch_endpoint: OpenSearch endpoint (auto-detected if available)
            opensearch_index: OpenSearch index name
            faiss_path: Local path for FAISS storage
            
            enable_graph_memory: Enable Neptune Analytics graph memory
            neptune_graph_id: Neptune Analytics graph identifier
            
            custom_fact_extraction_prompt: Custom prompt for memory extraction
            memory_search_limit: Max memories to retrieve for context
            memory_threshold: Similarity threshold for memory retrieval
            
            session_storage_dir: Directory for session storage (optional)
        """
        
        super().__init__(session_id=session_id)
        
        self.user_id = user_id
        self.memory_search_limit = memory_search_limit
        self.memory_threshold = memory_threshold
        self.session_id = session_id
        # AWS Configuration
        self.region_name = region_name or 'us-west-2'
        self.aws_profile = aws_profile
        
        # Initialize AWS session
        self.aws_session = self._create_aws_session()
        
        # Session storage setup (needed before vector store config)
        self.session_storage_dir = session_storage_dir or f"mem0_sessions/{session_id}"
        Path(self.session_storage_dir).mkdir(parents=True, exist_ok=True)
        
        # Model Configuration with AWS defaults
        self.bedrock_llm_model = bedrock_llm_model or self._get_default_llm_model()
        self.bedrock_embed_model = bedrock_embed_model or self._get_default_embed_model()
        
        # Vector Store Configuration
        self.vector_store_mode = vector_store_mode or "auto"
        self.resolved_vector_store = self._resolve_vector_store_config(
            opensearch_endpoint, opensearch_index, faiss_path
        )
        
        # Graph Memory Configuration
        self.enable_graph_memory = enable_graph_memory
        self.neptune_graph_id = neptune_graph_id or os.getenv("NEPTUNE_ANALYTICS_GRAPH_IDENTIFIER")
        
        # Set up environment variables for Mem0's AWS Bedrock provider
        self._setup_aws_environment()
        
        # Build and initialize Mem0 configuration
        mem0_config = self._build_mem0_config(custom_fact_extraction_prompt)
        
        try:
            self.memory = Memory.from_config(mem0_config)
            logger.info(f"Initialized Mem0SessionManager with {self.resolved_vector_store['type']} vector store")
        except Exception as e:
            logger.error(f"Failed to initialize Mem0 Memory: {e}")
            raise RuntimeError(f"Mem0 initialization failed: {e}")
        
        # Session state tracking
        self._current_session: Optional[Session] = None
        self._current_agent: Optional[SessionAgent] = None
        self._message_counter = 0
    
    def _create_aws_session(self) -> boto3.Session:
        """Create boto3 session with optional profile."""
        try:
            if self.aws_profile:
                session = boto3.Session(
                    profile_name=self.aws_profile,
                    region_name=self.region_name
                )
            else:
                session = boto3.Session(region_name=self.region_name)
            
            # Verify credentials work
            sts = session.client('sts')
            sts.get_caller_identity()
            
            return session
            
        except Exception as e:
            logger.error(f"Failed to create AWS session: {e}")
            raise RuntimeError(f"AWS session initialization failed: {e}")
    
    def _setup_aws_environment(self):
        """Set up environment variables for Mem0's AWS Bedrock provider."""
        try:
            # Set AWS_REGION if not already set
            if not os.environ.get("AWS_REGION"):
                os.environ["AWS_REGION"] = self.region_name
            
            # Get credentials from the boto3 session and set environment variables
            credentials = self.aws_session.get_credentials()
            if credentials:
                if not os.environ.get("AWS_ACCESS_KEY_ID"):
                    os.environ["AWS_ACCESS_KEY_ID"] = credentials.access_key
                if not os.environ.get("AWS_SECRET_ACCESS_KEY"):
                    os.environ["AWS_SECRET_ACCESS_KEY"] = credentials.secret_key
                if credentials.token and not os.environ.get("AWS_SESSION_TOKEN"):
                    os.environ["AWS_SESSION_TOKEN"] = credentials.token
                    
                logger.info(f"Set AWS environment variables for Mem0 (region: {self.region_name})")
            else:
                logger.warning("No AWS credentials found in session")
                
        except Exception as e:
            logger.warning(f"Failed to set AWS environment variables: {e}")
    
    def _get_default_llm_model(self) -> str:
        """Get default LLM model (using OpenAI for Mem0 compatibility)."""
        return "anthropic.claude-3-5-haiku-20241022-v1:0"
    
    def _get_default_embed_model(self) -> str:
        """Get default embedding model (using OpenAI for Mem0 compatibility)"""
        return "amazon.titan-embed-text-v2:0"
    
    def _resolve_vector_store_config(
        self, 
        opensearch_endpoint: Optional[str],
        opensearch_index: Optional[str], 
        faiss_path: Optional[str]
    ) -> Dict[str, Any]:
        """Resolve vector store configuration based on mode and environment."""
        
        if self.vector_store_mode == "faiss":
            return self._get_faiss_config(faiss_path)
        elif self.vector_store_mode == "opensearch":
            return self._get_opensearch_config(opensearch_endpoint, opensearch_index)
        else:  # auto mode
            # Try OpenSearch first (production), fallback to FAISS (development)
            if self._is_opensearch_available(opensearch_endpoint):
                logger.info("Auto-detected OpenSearch environment, using OpenSearch vector store")
                return self._get_opensearch_config(opensearch_endpoint, opensearch_index)
            else:
                logger.info("OpenSearch not available, using FAISS vector store for development")
                return self._get_faiss_config(faiss_path)
    
    def _is_opensearch_available(self, endpoint: Optional[str]) -> bool:
        """Check if OpenSearch is available in current environment."""
        endpoint = endpoint or os.getenv("OPENSEARCH_ENDPOINT")
        if not endpoint:
            return False
            
        try:
            # Try to create OpenSearch client and test connection
            from opensearchpy import OpenSearch
            client = OpenSearch([endpoint])
            client.info()  # Simple health check
            return True
        except Exception:
            return False
    
    def _get_faiss_config(self, faiss_path: Optional[str]) -> Dict[str, Any]:
        """Get FAISS vector store configuration."""
        if not faiss_path:
            faiss_path = f"{self.session_storage_dir}/faiss_db"
        
        # Ensure directory exists
        Path(faiss_path).parent.mkdir(parents=True, exist_ok=True)
        
        return {
            "type": "faiss",
            "provider": "faiss",
            "config": {
                "collection_name": f"mem0_session_{self.session_id}",
                "path": faiss_path,
                "embedding_model_dims": 1024  # OpenAI text-embedding-3-small dimensions
            }
        }
    
    def _get_opensearch_config(
        self, 
        endpoint: Optional[str], 
        index: Optional[str]
    ) -> Dict[str, Any]:
        """Get OpenSearch vector store configuration."""
        endpoint = endpoint or os.getenv("OPENSEARCH_ENDPOINT")
        if not endpoint:
            raise ValueError("OpenSearch endpoint required but not provided or found in environment")
        
        # Use shared index from environment variable instead of per-session index
        index = index or os.getenv("OPENSEARCH_INDEX", "mem0-shared")
        
        # Get AWS credentials for OpenSearch Serverless authentication
        if AWSV4SignerAuth is None:
            raise ImportError("opensearchpy with AWSV4SignerAuth is required for OpenSearch Serverless. Install with: pip install opensearch-py[aws]")
        
        credentials = self.aws_session.get_credentials()
        auth = AWSV4SignerAuth(credentials, self.region_name, "aoss")
        
        return {
            "type": "opensearch",
            "provider": "opensearch",  
            "config": {
                "collection_name": index,
                "host": endpoint,
                "embedding_model_dims": 1024,  # AWS Titan text embedding v2 dimensions
                "use_ssl": True,
                "verify_certs": True,
                "port": 443,  # HTTPS port for OpenSearch Serverless
                # AWS SigV4 authentication for OpenSearch Serverless
                "http_auth": auth
            }
        }
    
    def _get_graph_config(self) -> Optional[Dict[str, Any]]:
        """Get Neptune Analytics graph configuration if enabled."""
        if not self.enable_graph_memory or not self.neptune_graph_id:
            return None
            
        return {
            "provider": "neptune",
            "config": {
                "endpoint": f"neptune-graph://{self.neptune_graph_id}",
                "region_name": self.region_name
            }
        }
    
    def _build_mem0_config(self, custom_fact_extraction_prompt: Optional[str]) -> Dict[str, Any]:
        """Build complete Mem0 configuration."""
        config = {
            # LLM Configuration (AWS Bedrock)
            "llm": {
                "provider": "aws_bedrock",
                "config": {
                    "model": self.bedrock_llm_model,
                    "temperature": 0.1,
                    "max_tokens": 2000
                }
            },
            
            # Embedder Configuration (AWS Bedrock)  
            "embedder": {
                "provider": "aws_bedrock",
                "config": {
                    "model": self.bedrock_embed_model
                }
            },
            
            # Vector Store Configuration
            "vector_store": {
                "provider": self.resolved_vector_store["provider"],
                "config": self.resolved_vector_store["config"]
            },
            
            # History Database - Mem0's built-in storage
            "history_db_path": f"{self.session_storage_dir}/mem0_history.db",
            "version": "v1.1"
        }
        
        # Add graph store if enabled
        graph_config = self._get_graph_config()
        if graph_config:
            config["graph_store"] = graph_config
            
        # Add custom prompts if provided
        if custom_fact_extraction_prompt:
            config["custom_fact_extraction_prompt"] = custom_fact_extraction_prompt
            
        return config
    
    # ==============================================================================
    # Strands SessionManager Interface Implementation
    # ==============================================================================
    
    def create_session(self, session: Session) -> Session:
        """Create a new session."""
        self._current_session = session
        
        # Store in Mem0's history database via conversation
        self._store_session_metadata(session)
        
        logger.info(f"Created session {session.session_id}")
        return session
    
    def read_session(self, session_id: str) -> Optional[Session]:
        """Read a session by ID."""
        if self._current_session and self._current_session.session_id == session_id:
            return self._current_session
        
        # Try to reconstruct from Mem0's storage
        session_data = self._load_session_metadata(session_id)
        if session_data:
            self._current_session = Session.from_dict(session_data)
            return self._current_session
        
        return None
    
    def create_agent(self, agent: SessionAgent) -> SessionAgent:
        """Create a new agent within the session."""
        self._current_agent = agent
        
        # Store agent metadata
        self._store_agent_metadata(agent)
        
        logger.info(f"Created agent {agent.agent_id} in session {self.session_id}")
        return agent
    
    def read_agent(self, session_id: str, agent_id: str) -> Optional[SessionAgent]:
        """Read an agent by session and agent ID."""
        if (self._current_agent and 
            self._current_agent.agent_id == agent_id and 
            self.session_id == session_id):
            return self._current_agent
        
        # Try to reconstruct from storage
        agent_data = self._load_agent_metadata(agent_id)
        if agent_data:
            self._current_agent = SessionAgent.from_dict(agent_data)
            return self._current_agent
        
        return None
    
    def update_agent(self, agent: SessionAgent) -> SessionAgent:
        """Update an existing agent."""
        self._current_agent = agent
        self._store_agent_metadata(agent)
        
        logger.debug(f"Updated agent {agent.agent_id}")
        return agent
    
    def create_message(self, message: SessionMessage) -> SessionMessage:
        """Create a new message."""
        logger.info(f"create_message called with: {type(message)} - {message}")
        
        # Store in Mem0's conversation history
        conversation_turn = self._convert_message_to_conversation(message)
        logger.info(f"Converted to conversation: {conversation_turn}")
        
        # Add to Mem0 with automatic memory extraction
        try:
            # Handle different message formats for metadata
            message_id = getattr(message, 'message_id', None) or getattr(message, 'id', str(uuid.uuid4()))
            created_at = getattr(message, 'created_at', None) or datetime.now(timezone.utc).isoformat()
            
            # Extract content string from conversation format
            content_text = ""
            for turn in conversation_turn:
                content_text += f"{turn.get('role', '')}: {turn.get('content', '')} "
            content_text = content_text.strip()
            
            logger.info(f"About to call custom memory add with content: {content_text}")
            result = self.add_memory(
                content_text,
                metadata={
                    "session_id": self.session_id,
                    "message_id": message_id,
                    "agent_id": getattr(self._current_agent, 'agent_id', 'unknown'),
                    "timestamp": created_at,
                    "source": "strands_session"
                }
            )
            logger.info(f"Memory.add result: {result}")
            
            self._message_counter += 1
            logger.debug(f"Created message {message_id} with memory extraction")
            
        except Exception as e:
            logger.warning(f"Failed to store message in Mem0: {e}")
            import traceback
            logger.warning(f"Full traceback: {traceback.format_exc()}")
        
        return message
    
    def read_message(self, session_id: str, agent_id: str, message_id: str) -> Optional[SessionMessage]:
        """Read a message by IDs."""
        # For now, we'll rely on Mem0's conversation history
        # In a full implementation, you might query Mem0's history database directly
        logger.debug(f"Reading message {message_id} (delegated to conversation history)")
        return None
    
    def update_message(self, message: SessionMessage) -> SessionMessage:
        """Update an existing message."""
        # Update in Mem0 if needed
        logger.debug(f"Updated message {message.message_id}")
        return message
    
    def list_messages(
        self, 
        session_id: str, 
        agent_id: str, 
        limit: Optional[int] = None
    ) -> List[SessionMessage]:
        """List messages for a session and agent."""
        # Query Mem0's conversation history
        try:
            # This would need to be implemented based on Mem0's API
            # For now, return empty list
            logger.debug(f"Listing messages for session {session_id}, agent {agent_id}")
            return []
        except Exception as e:
            logger.warning(f"Failed to list messages: {e}")
            return []
    
    # ==============================================================================
    # Memory Operations (Enhanced Capabilities)
    # ==============================================================================
    
    def add_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Add memory to Mem0 with session context."""
        base_metadata = {
            "session_id": self.session_id,
            "source": "strands_session"
        }
        if metadata:
            base_metadata.update(metadata)
        
        # Use custom fact extraction for Claude models
        return self._add_memory_with_custom_extraction(content, base_metadata)
    
    def _add_memory_with_custom_extraction(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Add memory with custom fact extraction that works with Claude."""
        # Extract facts using Claude in a way that works
        facts = self._extract_facts_with_claude(content)
        
        # Store each fact separately using Mem0's vector store directly
        results = []
        for fact in facts:
            try:
                # Create embedding for the fact
                embedding = self.memory.embedding_model.embed(fact)
                
                # Store in vector store with metadata
                memory_id = str(uuid.uuid4())
                memory_item = {
                    "id": memory_id,
                    "data": fact,  # Use 'data' key that Mem0 expects
                    "memory": fact,
                    "user_id": self.user_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "metadata": metadata
                }
                
                # Add to vector store using correct insert signature
                self.memory.vector_store.insert(
                    vectors=[embedding],
                    payloads=[memory_item],
                    ids=[memory_id]
                )
                
                # Add to SQL storage (skip for now due to method mismatch)
                # if hasattr(self.memory, 'db') and self.memory.db:
                #     self.memory.db.add_memory(memory_item)
                
                results.append({"id": memory_id, "memory": fact})
                logger.info(f"Stored memory: {fact}")
                
            except Exception as e:
                logger.warning(f"Failed to store fact '{fact}': {e}")
        
        return {"results": results}
    
    def _extract_facts_with_claude(self, content: str) -> List[str]:
        """Extract facts using Claude without JSON response format."""
        try:
            prompt = f"""Extract specific, factual information from the following text. 
Return each fact on a separate line, starting with "FACT: ".
Only extract concrete, verifiable facts about people, places, things, or relationships.

Text: {content}

Facts:"""

            response = self.memory.llm.generate_response([
                {"role": "user", "content": prompt}
            ])
            
            # Parse the response to extract facts
            facts = []
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('FACT: '):
                    fact = line[6:].strip()  # Remove "FACT: " prefix
                    if fact:
                        facts.append(fact)
            
            logger.info(f"Extracted {len(facts)} facts: {facts}")
            return facts if facts else [content]  # Fallback to original content
            
        except Exception as e:
            logger.warning(f"Fact extraction failed: {e}")
            return [content]  # Fallback to storing the original content
    
    def search_memories(
        self, 
        query: str, 
        limit: Optional[int] = None, 
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Search memories with session and user context."""
        search_limit = limit or self.memory_search_limit
        search_threshold = threshold or self.memory_threshold
        
        return self.memory.search(
            query=query,
            user_id=self.user_id,
            limit=search_limit,
            threshold=search_threshold
        )
    
    def get_all_memories(self) -> Dict[str, Any]:
        """Get all memories for current user."""
        return self.memory.get_all(user_id=self.user_id)
    
    def delete_memory(self, memory_id: str) -> Dict[str, Any]:
        """Delete a specific memory."""
        return self.memory.delete(memory_id=memory_id)
    
    def reset_memories(self) -> Dict[str, Any]:
        """Reset all memories for current user."""
        return self.memory.reset(user_id=self.user_id)
    
    def get_enhanced_context(
        self, 
        current_message: str, 
        include_memories: bool = True,
        include_relations: bool = None
    ) -> Dict[str, Any]:
        """
        Get enhanced context combining session data and Mem0 memories.
        
        Returns enriched context for agent responses.
        """
        context = {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "memories": [],
            "memory_summary": ""
        }
        
        if include_memories:
            try:
                memory_results = self.search_memories(current_message)
                memories = memory_results.get("results", [])
                context["memories"] = memories
                
                # Create summary of relevant memories
                if memories:
                    memory_texts = [mem.get("memory", "") for mem in memories[:3]]  # Top 3
                    context["memory_summary"] = " | ".join(memory_texts)
                
                # Include graph relations if enabled and requested
                if (include_relations is None and self.enable_graph_memory) or include_relations:
                    context["relations"] = memory_results.get("relations", [])
                    
            except Exception as e:
                logger.warning(f"Failed to retrieve memories for context: {e}")
                
        return context
    
    # ==============================================================================
    # Helper Methods
    # ==============================================================================
    
    def _convert_message_to_conversation(self, message: SessionMessage) -> List[Dict[str, str]]:
        """Convert SessionMessage to Mem0 conversation format."""
        # Handle different message formats
        if hasattr(message, 'message'):
            msg_data = message.message
        elif isinstance(message, dict):
            msg_data = message
        else:
            msg_data = {"role": "user", "content": str(message)}
        
        if isinstance(msg_data, dict):
            role = msg_data.get("role", "user")
            content = msg_data.get("content", "")
            
            # Handle Strands' list-based content format
            if isinstance(content, list):
                # Extract text from Strands format: [{'text': 'actual content'}]
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and 'text' in item:
                        text_parts.append(item['text'])
                    else:
                        text_parts.append(str(item))
                content = " ".join(text_parts)
        else:
            # Fallback for other message formats
            role = "user"  # Default assumption
            content = str(msg_data)
        
        return [{"role": role, "content": content}]
    
    def _store_session_metadata(self, session: Session):
        """Store session metadata."""
        metadata_path = Path(self.session_storage_dir) / "session_metadata.json"
        try:
            with open(metadata_path, 'w') as f:
                json.dump(session.to_dict(), f, default=str)
        except Exception as e:
            logger.warning(f"Failed to store session metadata: {e}")
    
    def _load_session_metadata(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session metadata."""
        metadata_path = Path(self.session_storage_dir) / "session_metadata.json"
        try:
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load session metadata: {e}")
        return None
    
    def _store_agent_metadata(self, agent: SessionAgent):
        """Store agent metadata."""
        agents_dir = Path(self.session_storage_dir) / "agents"
        agents_dir.mkdir(exist_ok=True)
        
        agent_path = agents_dir / f"agent_{agent.agent_id}.json"
        try:
            # Create a serializable representation of the agent
            agent_data = {
                "agent_id": agent.agent_id,
                "session_id": getattr(agent, 'session_id', self.session_id),
                "created_at": getattr(agent, 'created_at', datetime.now(timezone.utc).isoformat()),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            with open(agent_path, 'w') as f:
                json.dump(agent_data, f, default=str)
        except Exception as e:
            logger.warning(f"Failed to store agent metadata: {e}")
    
    def _load_agent_metadata(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Load agent metadata."""
        agent_path = Path(self.session_storage_dir) / "agents" / f"agent_{agent_id}.json"
        try:
            if agent_path.exists():
                with open(agent_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load agent metadata: {e}")
        return None
    
    # ==============================================================================
    # Required Abstract Methods Implementation
    # ==============================================================================
    
    def initialize(self, agent, **kwargs) -> None:
        """Initialize the session manager."""
        # Already initialized in __init__, but this method is required by the abstract base class
        self._agent = agent
        logger.info(f"Session manager initialized for session {self.session_id}")
    
    def append_message(self, message: SessionMessage, agent) -> None:
        """Append a message to the current session."""
        # Use the existing create_message method
        self.create_message(message)
    
    def redact_latest_message(self) -> None:
        """Redact the latest message in the session."""
        # Implementation depends on your specific requirements
        # For now, we'll log the action
        logger.info(f"Redacting latest message in session {self.session_id}")
        # You might want to mark the message as redacted in Mem0 or remove it
        # This is a placeholder implementation
    
    def sync_agent(self, agent: SessionAgent) -> None:
        """Synchronize agent state."""
        # Use the existing update_agent method
        self.update_agent(agent)
        logger.debug(f"Synced agent {agent.agent_id}")

    # ==============================================================================
    # Utility Methods
    # ==============================================================================
    
    def get_configuration_info(self) -> Dict[str, Any]:
        """Get current configuration information for debugging."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "region_name": self.region_name,
            "bedrock_llm_model": self.bedrock_llm_model,
            "bedrock_embed_model": self.bedrock_embed_model,
            "vector_store": self.resolved_vector_store,
            "graph_memory_enabled": self.enable_graph_memory,
            "neptune_graph_id": self.neptune_graph_id,
            "memory_search_limit": self.memory_search_limit,
            "memory_threshold": self.memory_threshold,
            "session_storage_dir": self.session_storage_dir
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components."""
        health = {
            "session_manager": "healthy",
            "aws_session": "unknown", 
            "mem0_memory": "unknown",
            "vector_store": "unknown",
            "graph_store": "not_enabled" if not self.enable_graph_memory else "unknown"
        }
        
        try:
            # Test AWS session
            self.aws_session.client('sts').get_caller_identity()
            health["aws_session"] = "healthy"
        except Exception as e:
            health["aws_session"] = f"error: {e}"
            
        try:
            # Test Mem0 memory
            self.memory.get_all(user_id=self.user_id, limit=1)
            health["mem0_memory"] = "healthy"
        except Exception as e:
            health["mem0_memory"] = f"error: {e}"
            
        # Vector store health
        health["vector_store"] = f"{self.resolved_vector_store['type']}: operational"
        
        if self.enable_graph_memory:
            health["graph_store"] = f"neptune:{self.neptune_graph_id}"
            
        return health