#!/usr/bin/env python3
"""
Context memory test for DeepSeek R1 - Long multi-turn conversations without tool calls.

Tests the model's ability to:
1. Remember information from earlier in the conversation
2. Maintain coherent context across many turns
3. Reference previous statements appropriately
4. Build upon earlier topics naturally
5. Handle context window limits gracefully
"""

import json
import requests
import time
from typing import Dict, List, Any

SERVER = "http://0.0.0.0:8003"
CHAT_URL = f"{SERVER}/v1/chat/completions"

def make_request(messages: List[Dict], verbose: bool = True) -> Dict[str, Any]:
    """Make a non-streaming request for context testing."""
    payload = {
        "model": "DeepSeek-R1-0528-FP4",
        "messages": messages,
        "stream": False,
        "max_tokens": 4096,
        "temperature": 0.3,  # Slightly higher for more natural conversation
    }
    
    if verbose:
        print(f"Request with {len(messages)} messages, last: '{messages[-1]['content'][:60]}...'")
    
    response = requests.post(CHAT_URL, json=payload)
    if response.status_code != 200:
        return {"error": f"HTTP {response.status_code}: {response.text}"}
    
    result = response.json()
    if 'choices' in result and len(result['choices']) > 0:
        message = result['choices'][0]['message']
        return {
            "content": message.get('content', ''),
            "reasoning_content": message.get('reasoning_content', ''),
            "usage": result.get('usage', {}),
            "finish_reason": result['choices'][0].get('finish_reason')
        }
    
    return {"content": "", "reasoning_content": ""}

def test_personal_information_memory():
    """Test if model remembers personal information shared earlier."""
    print("=== TESTING PERSONAL INFORMATION MEMORY ===")
    
    messages = []
    
    # Turn 1: Share personal info
    messages.append({"role": "user", "content": "Hi! My name is Sarah and I'm a software engineer working on AI systems. I have two cats named Pixel and Byte, and I love hiking in the mountains on weekends."})
    
    result1 = make_request(messages, verbose=False)
    messages.append({"role": "assistant", "content": result1["content"]})
    print(f"1. Introduction response: {len(result1['content'])} chars")
    
    # Turn 2: Ask about work
    messages.append({"role": "user", "content": "What do you think about the current state of AI development?"})
    result2 = make_request(messages, verbose=False)
    messages.append({"role": "assistant", "content": result2["content"]})
    print(f"2. AI discussion: {len(result2['content'])} chars")
    
    # Turn 3: Test name memory
    messages.append({"role": "user", "content": "Can you remind me what we were talking about? And do you remember my name?"})
    result3 = make_request(messages, verbose=False)
    messages.append({"role": "assistant", "content": result3["content"]})
    
    # Check if name is remembered
    name_remembered = "Sarah" in result3["content"]
    print(f"3. Name memory test: {'✅' if name_remembered else '❌'} (Sarah mentioned: {name_remembered})")
    
    # Turn 4: Test pet memory
    messages.append({"role": "user", "content": "I'm thinking of getting a third cat. What do you think would be a good name that goes with my other cats?"})
    result4 = make_request(messages, verbose=False)
    messages.append({"role": "assistant", "content": result4["content"]})
    
    # Check if pet names are remembered
    pixel_mentioned = "Pixel" in result4["content"]
    byte_mentioned = "Byte" in result4["content"]
    pets_remembered = pixel_mentioned or byte_mentioned
    print(f"4. Pet memory test: {'✅' if pets_remembered else '❌'} (Pixel: {pixel_mentioned}, Byte: {byte_mentioned})")
    
    # Turn 5: Test profession memory
    messages.append({"role": "user", "content": "Given my background, what programming languages would you recommend I focus on?"})
    result5 = make_request(messages, verbose=False)
    
    # Check if profession is remembered
    engineer_context = any(word in result5["content"].lower() for word in ["engineer", "software", "development", "programming"])
    print(f"5. Profession memory: {'✅' if engineer_context else '❌'} (Engineering context: {engineer_context})")
    
    return {
        "name_remembered": name_remembered,
        "pets_remembered": pets_remembered, 
        "profession_remembered": engineer_context,
        "total_turns": len(messages) // 2
    }

def test_story_continuity():
    """Test if model can maintain story continuity across turns."""
    print("\n=== TESTING STORY CONTINUITY ===")
    
    messages = []
    
    # Start a story
    messages.append({"role": "user", "content": "Let's create a story together. I'll start: 'Detective Maria Rodriguez walked into the old mansion on a stormy night, her flashlight cutting through the darkness. She was investigating the mysterious disappearance of the mansion's owner, Mr. Blackwood.' Please continue the story."})
    
    result1 = make_request(messages, verbose=False)
    messages.append({"role": "assistant", "content": result1["content"]})
    print(f"1. Story continuation: {len(result1['content'])} chars")
    
    # Continue story development
    messages.append({"role": "user", "content": "Great! Now let's add that Maria finds a hidden room behind a bookshelf. What does she discover there?"})
    result2 = make_request(messages, verbose=False)
    messages.append({"role": "assistant", "content": result2["content"]})
    print(f"2. Hidden room discovery: {len(result2['content'])} chars")
    
    # Test character memory
    messages.append({"role": "user", "content": "What was the detective's name again? And who was she looking for?"})
    result3 = make_request(messages, verbose=False)
    messages.append({"role": "assistant", "content": result3["content"]})
    
    # Check story memory
    maria_remembered = "Maria" in result3["content"] or "Rodriguez" in result3["content"]
    blackwood_remembered = "Blackwood" in result3["content"]
    print(f"3. Character memory: {'✅' if maria_remembered else '❌'} Maria, {'✅' if blackwood_remembered else '❌'} Blackwood")
    
    # Continue with plot development
    messages.append({"role": "user", "content": "Let's say Maria finds an old diary in the hidden room. What does the first entry reveal about Mr. Blackwood's disappearance?"})
    result4 = make_request(messages, verbose=False)
    messages.append({"role": "assistant", "content": result4["content"]})
    print(f"4. Diary entry: {len(result4['content'])} chars")
    
    # Test plot consistency
    messages.append({"role": "user", "content": "Based on everything we've established so far, how do you think this mystery should be resolved?"})
    result5 = make_request(messages, verbose=False)
    
    # Check if resolution references earlier elements
    story_elements = ["Maria", "Rodriguez", "Blackwood", "mansion", "hidden", "diary"]
    elements_referenced = sum(1 for element in story_elements if element.lower() in result5["content"].lower())
    
    print(f"5. Story resolution coherence: {elements_referenced}/{len(story_elements)} elements referenced")
    
    return {
        "character_memory": maria_remembered and blackwood_remembered,
        "story_coherence": elements_referenced >= 3,
        "total_turns": len(messages) // 2
    }

def test_technical_discussion_context():
    """Test context maintenance in technical discussions."""
    print("\n=== TESTING TECHNICAL DISCUSSION CONTEXT ===")
    
    messages = []
    
    # Start technical discussion
    messages.append({"role": "user", "content": "I'm working on a machine learning project to predict customer churn for a SaaS company. We have 50,000 customers with features like usage patterns, subscription tier, support tickets, and payment history. What approach would you recommend?"})
    
    result1 = make_request(messages, verbose=False)
    messages.append({"role": "assistant", "content": result1["content"]})
    print(f"1. Initial ML advice: {len(result1['content'])} chars")
    
    # Follow up on specific aspect
    messages.append({"role": "user", "content": "That's helpful. Now, regarding feature engineering - what specific features would you derive from the usage patterns data?"})
    result2 = make_request(messages, verbose=False)
    messages.append({"role": "assistant", "content": result2["content"]})
    print(f"2. Feature engineering: {len(result2['content'])} chars")
    
    # Test context memory
    messages.append({"role": "user", "content": "Going back to our original problem - what was the target variable we're trying to predict again?"})
    result3 = make_request(messages, verbose=False)
    messages.append({"role": "assistant", "content": result3["content"]})
    
    churn_remembered = "churn" in result3["content"].lower()
    print(f"3. Target variable memory: {'✅' if churn_remembered else '❌'} (churn mentioned: {churn_remembered})")
    
    # Add complexity
    messages.append({"role": "user", "content": "We've decided to use a gradient boosting approach. How should we handle class imbalance if only 5% of customers churn?"})
    result4 = make_request(messages, verbose=False)
    messages.append({"role": "assistant", "content": result4["content"]})
    print(f"4. Class imbalance handling: {len(result4['content'])} chars")
    
    # Test comprehensive context
    messages.append({"role": "user", "content": "Can you summarize our entire discussion and provide a final implementation roadmap?"})
    result5 = make_request(messages, verbose=False)
    
    # Check if summary includes key elements
    key_elements = ["churn", "50,000", "gradient boosting", "imbalance", "features", "SaaS"]
    elements_in_summary = sum(1 for element in key_elements if element.lower() in result5["content"].lower())
    
    print(f"5. Comprehensive summary: {elements_in_summary}/{len(key_elements)} key elements included")
    
    return {
        "target_remembered": churn_remembered,
        "comprehensive_summary": elements_in_summary >= 4,
        "total_turns": len(messages) // 2
    }

def test_very_long_conversation():
    """Test context maintenance in a very long conversation."""
    print("\n=== TESTING VERY LONG CONVERSATION (15+ turns) ===")
    
    messages = []
    topics_discussed = []
    
    conversation_flow = [
        ("travel", "I'm planning a trip to Japan next spring. What are some must-see places in Tokyo?"),
        ("food", "That sounds amazing! What about Japanese food? I'm vegetarian - what dishes should I try?"),
        ("culture", "I'm also interested in traditional Japanese culture. What cultural experiences would you recommend?"),
        ("language", "Should I learn some Japanese before I go? What are the most important phrases?"),
        ("transportation", "How does transportation work in Tokyo? I've heard about the train system."),
        ("accommodation", "What type of accommodation would you suggest for a first-time visitor?"),
        ("budget", "Speaking of costs, what should I budget for a 10-day trip to Japan?"),
        ("seasons", "You mentioned spring - why is that a good time to visit? What about cherry blossoms?"),
        ("day_planning", "Can you help me plan a sample day itinerary combining some of the places we discussed?"),
        ("shopping", "I'd also like to do some shopping. What are some unique things to buy in Japan?"),
        ("etiquette", "What cultural etiquette should I be aware of to be respectful?"),
        ("technology", "I've heard Japan is very high-tech. What technological experiences should I look for?"),
        ("memory_test", "This has been a great discussion! Can you remind me what we talked about regarding food options for vegetarians?"),
        ("comprehensive_test", "Can you create a complete travel checklist based on everything we've discussed?"),
        ("final_test", "What was the original reason I said I was interested in visiting Japan?")
    ]
    
    for i, (topic, question) in enumerate(conversation_flow, 1):
        messages.append({"role": "user", "content": question})
        result = make_request(messages, verbose=False)
        
        # Handle potential errors
        if "error" in result:
            print(f"{i:2d}. {topic.title()}: ERROR - {result['error']}")
            break
            
        content = result.get("content", "")
        messages.append({"role": "assistant", "content": content})
        topics_discussed.append(topic)
        
        print(f"{i:2d}. {topic.title()}: {len(content)} chars")
        
        # Test specific memory points
        if topic == "memory_test":
            vegetarian_remembered = "vegetarian" in content.lower()
            print(f"    Memory check: {'✅' if vegetarian_remembered else '❌'} Vegetarian context remembered")
        
        elif topic == "comprehensive_test":
            # Check if checklist includes multiple discussed topics
            checklist_topics = sum(1 for t in ["food", "transport", "accommodation", "culture", "language"] 
                                 if t in content.lower())
            print(f"    Checklist completeness: {checklist_topics}/5 major topics included")
        
        elif topic == "final_test":
            # Check if original context (spring trip to Japan) is remembered
            spring_remembered = "spring" in content.lower()
            japan_remembered = "japan" in content.lower()
            original_context = spring_remembered and japan_remembered
            print(f"    Original context: {'✅' if original_context else '❌'} Spring Japan trip remembered")
    
    total_tokens = sum(len(msg["content"]) for msg in messages) // 4  # Rough token estimate
    print(f"\nConversation stats:")
    print(f"  Total turns: {len(messages) // 2}")
    print(f"  Total messages: {len(messages)}")
    print(f"  Estimated tokens: ~{total_tokens}")
    print(f"  Topics covered: {len(topics_discussed)}")
    
    return {
        "total_turns": len(messages) // 2,
        "topics_covered": len(topics_discussed),
        "estimated_tokens": total_tokens
    }

def main():
    """Run all context memory tests."""
    print("DeepSeek R1 Context Memory & Long Conversation Testing")
    print("=" * 60)
    
    # Health check
    try:
        response = requests.get(f"{SERVER}/health", timeout=5)
        print(f"Server health: {response.status_code}\n")
    except Exception as e:
        print(f"Warning: Health check failed: {e}\n")
    
    results = {}
    
    # Run tests
    try:
        results["personal_memory"] = test_personal_information_memory()
        time.sleep(1)
        
        results["story_continuity"] = test_story_continuity() 
        time.sleep(1)
        
        results["technical_context"] = test_technical_discussion_context()
        time.sleep(1)
        
        results["long_conversation"] = test_very_long_conversation()
        
    except Exception as e:
        print(f"Error during testing: {e}")
        return
    
    # Summary
    print("\n" + "=" * 60)
    print("CONTEXT MEMORY TEST RESULTS")
    print("=" * 60)
    
    print("\n📝 Personal Information Memory:")
    pm = results["personal_memory"]
    print(f"  Name remembered: {'✅' if pm['name_remembered'] else '❌'}")
    print(f"  Pets remembered: {'✅' if pm['pets_remembered'] else '❌'}")
    print(f"  Profession remembered: {'✅' if pm['profession_remembered'] else '❌'}")
    print(f"  Turns: {pm['total_turns']}")
    
    print("\n📚 Story Continuity:")
    sc = results["story_continuity"]
    print(f"  Character memory: {'✅' if sc['character_memory'] else '❌'}")
    print(f"  Story coherence: {'✅' if sc['story_coherence'] else '❌'}")
    print(f"  Turns: {sc['total_turns']}")
    
    print("\n🔬 Technical Discussion:")
    tc = results["technical_context"]
    print(f"  Target variable remembered: {'✅' if tc['target_remembered'] else '❌'}")
    print(f"  Comprehensive summary: {'✅' if tc['comprehensive_summary'] else '❌'}")
    print(f"  Turns: {tc['total_turns']}")
    
    print("\n💬 Long Conversation:")
    lc = results["long_conversation"]
    print(f"  Total turns: {lc['total_turns']}")
    print(f"  Topics covered: {lc['topics_covered']}")
    print(f"  Estimated tokens: ~{lc['estimated_tokens']}")
    
    # Overall assessment
    total_memory_tests = 6  # name, pets, profession, characters, target, summary
    passed_memory_tests = sum([
        pm['name_remembered'], pm['pets_remembered'], pm['profession_remembered'],
        sc['character_memory'], tc['target_remembered'], tc['comprehensive_summary']
    ])
    
    print(f"\n🎯 Overall Context Memory Score: {passed_memory_tests}/{total_memory_tests} ({passed_memory_tests/total_memory_tests*100:.1f}%)")
    
    if passed_memory_tests >= 5:
        print("🎉 EXCELLENT: Model demonstrates strong context memory capabilities!")
    elif passed_memory_tests >= 3:
        print("👍 GOOD: Model shows decent context retention with some limitations.")
    else:
        print("⚠️  NEEDS IMPROVEMENT: Context memory appears limited.")

if __name__ == "__main__":
    main()
