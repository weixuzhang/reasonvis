import requests
import json
import time

def test_single_scenario():
    """
    Test generating a single visualization scenario
    """
    # API Configuration
    api_key = "8beCPow2KcmVGufSecmUZrTQhVN2OnPb"  # Replace with your key
    url = "https://gptproxy.llmpaas.tencent.com/v1/chat/completions"  # Added v1 path
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": "gpt-4o",
        "messages": [{
            "role": "user",
            "content": """Generate a visualization scenario for business domain at reasoning level 2.
            Create a realistic business analysis scenario that includes:
            1. Initial data context
            2. A sequence of 4 progressive visualization queries
            3. Expected visualization types
            4. Key insights to be discovered

            Format the output as a JSON object with these fields:
            {
                "domain": "business",
                "level": 2,
                "context": "description of the business context",
                "queries": ["query1", "query2", "query3", "query4"],
                "vis_types": ["type1", "type2", "type3", "type4"],
                "insights": ["insight1", "insight2", ...]
            }"""
        }],
        "temperature": 0.7,
        "max_tokens": 1000
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        
        print("\nStatus Code:", response.status_code)
        print("\nResponse Headers:", response.headers)
        print("\nRaw Response:", response.text)
        
        if response.status_code == 200:
            result = response.json()
            print("\nParsed Response:")
            print(json.dumps(result, indent=2))
            
            # Extract the content from the response
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                try:
                    scenario = json.loads(content)
                    print("\nParsed Scenario:")
                    print(json.dumps(scenario, indent=2))
                except json.JSONDecodeError:
                    print("\nWarning: Could not parse scenario as JSON")
                    print("Content:", content)
        else:
            print(f"\nError: API request failed with status code {response.status_code}")
            print("Response:", response.text)
            
        return response.text

    except Exception as e:
        print(f"Error in API call: {str(e)}")
        return None

if __name__ == "__main__":
    print("=== Testing Single Scenario ===")
    single_result = test_single_scenario()