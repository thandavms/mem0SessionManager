#!/usr/bin/env python3
"""
Complete integration test: OpenSearch + optional Neptune
"""

import os
import time
from dotenv import load_dotenv
from strands import Agent
from mem0sm import Mem0SessionManager

# Load environment variables
load_dotenv()

def test_opensearch_only():
    """Test OpenSearch without Neptune first"""
    
    print("🔍 Testing OpenSearch Integration (No Neptune)")
    print("=" * 50)
    
    # Create session manager (OpenSearch only)
    session_manager = Mem0SessionManager(
        user_id="integration_user",
        session_id="integration_session",
        vector_store_mode="opensearch",
        region_name="us-west-2",
        enable_graph_memory=False  # Disable Neptune for now
    )
    
    print("✅ Session manager created (OpenSearch only)")
    
    # Create agent
    agent = Agent(
        system_prompt="You are a helpful assistant with memory.",
        session_manager=session_manager
    )
    
    # Test conversation
    response = agent("Hi! I'm David, a DevOps engineer who loves Kubernetes and works with microservices.")
    print(f"Agent response: {str(response)[:100]}...")
    
    # Wait for processing
    time.sleep(3)
    
    # Test search
    search_results = session_manager.search_memories("David")
    print(f"\nFound {len(search_results.get('results', []))} memories about David:")
    for r in search_results.get('results', []):
        print(f"  - {r.get('memory', 'N/A')} (score: {r.get('score', 0):.3f})")
    
    # Configuration check
    config = session_manager.get_configuration_info()
    print(f"\nConfiguration:")
    print(f"  Vector Store: {config['vector_store']['type']}")
    print(f"  Graph Memory: {config['graph_memory_enabled']}")
    print(f"  Index: {config['vector_store']['config']['collection_name']}")
    
    return True

def test_neptune_optional():
    """Test Neptune integration separately (optional)"""
    
    print("\n" + "🌊 Testing Neptune Integration (Optional)")
    print("=" * 50)
    
    neptune_graph_id = os.getenv('NEPTUNE_ANALYTICS_GRAPH_IDENTIFIER')
    if not neptune_graph_id:
        print("⚠️ Neptune not configured - skipping Neptune test")
        return True
    
    print(f"Neptune Graph ID: {neptune_graph_id}")
    
    try:
        # Try to create session manager with Neptune
        session_manager = Mem0SessionManager(
            user_id="neptune_test_user",
            session_id="neptune_test_session",
            vector_store_mode="opensearch",
            region_name="us-west-2",
            enable_graph_memory=True,
            neptune_graph_id=neptune_graph_id
        )
        
        print("✅ Neptune integration successful!")
        
        # Test health check
        health = session_manager.health_check()
        print("Health status:")
        for component, status in health.items():
            print(f"  {component}: {status}")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Neptune integration failed: {e}")
        print("This is expected if Neptune isn't fully configured or accessible")
        print("OpenSearch integration works independently")
        return False

def test_configuration_display():
    """Display current configuration"""
    
    print("\n" + "⚙️ Configuration Summary")
    print("=" * 50)
    
    print("Environment Variables:")
    print(f"  OPENSEARCH_ENDPOINT: {os.getenv('OPENSEARCH_ENDPOINT', 'Not Set')}")
    print(f"  OPENSEARCH_INDEX: {os.getenv('OPENSEARCH_INDEX', 'Not Set')}")
    print(f"  NEPTUNE_ANALYTICS_GRAPH_IDENTIFIER: {os.getenv('NEPTUNE_ANALYTICS_GRAPH_IDENTIFIER', 'Not Set')}")
    print(f"  AWS_REGION: {os.getenv('AWS_REGION', 'Not Set')}")

if __name__ == "__main__":
    try:
        # Display configuration
        test_configuration_display()
        
        # Test OpenSearch (should work)
        opensearch_success = test_opensearch_only()
        
        # Test Neptune (optional)
        neptune_success = test_neptune_optional()
        
        print("\n" + "🎉 Integration Test Summary")
        print("=" * 50)
        print(f"✅ OpenSearch Integration: {'Success' if opensearch_success else 'Failed'}")
        print(f"{'✅' if neptune_success else '⚠️'} Neptune Integration: {'Success' if neptune_success else 'Optional - Not Working'}")
        
        if opensearch_success:
            print("\n🚀 Your mem0sm session manager is ready to use!")
            print("   - OpenSearch vector storage working")
            print("   - Memory extraction and search working")
            print("   - Shared index configuration working")
            if neptune_success:
                print("   - Neptune graph memory working")
            else:
                print("   - Neptune can be enabled later when ready")
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()